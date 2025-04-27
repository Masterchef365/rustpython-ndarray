use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn, SliceInfo, SliceInfoElem};
use rustpython_vm::{
    atomic_func,
    builtins::{PyInt, PyModule, PySlice, PyStr, PyTuple},
    class::PyClassImpl,
    convert::ToPyObject,
    protocol::PyMappingMethods,
    PyObject, PyObjectRef, PyRef, PyResult, TryFromObject, VirtualMachine,
};

use std::{
    fmt::Display,
    sync::{Arc, RwLock},
};

use crate::GenericArray;

pub type DynamicSlice = SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>;

/// Provides a sliced representation of an array, where the slices are deferred until needed.
#[derive(Debug, Clone)]
pub struct PyNdArray<T> {
    slices: Vec<DynamicSlice>,
    unsliced: Arc<RwLock<ndarray::ArrayD<T>>>,
}

impl<T> PyNdArray<T> {
    pub fn from_array(data: ndarray::ArrayD<T>) -> Self {
        Self {
            slices: vec![],
            unsliced: Arc::new(RwLock::new(data)),
        }
    }

    /// Borrow the entire array immutably to read it for a moment
    pub fn read<U>(&self, mut readfn: impl FnMut(ArrayViewD<'_, T>) -> U) -> U {
        let arr = self.unsliced.read().unwrap();

        let default_slice = vec![SliceInfoElem::from(..); arr.ndim()];
        let default_slice = DynamicSlice::try_from(default_slice).unwrap();

        let mut arr_slice = arr.slice(default_slice);

        for slice in &self.slices {
            arr_slice = arr_slice.slice_move(slice);
        }

        readfn(arr_slice)
    }

    /// Borrow the entire array mutably for a moment
    pub fn write<U>(&self, writefn: impl Fn(ArrayViewMutD<'_, T>) -> U) -> U {
        let mut arr = self.unsliced.write().unwrap();

        let default_slice = vec![SliceInfoElem::from(..); arr.ndim()];
        let default_slice = DynamicSlice::try_from(default_slice).unwrap();

        let mut arr_slice = arr.slice_mut(default_slice);

        for slice in &self.slices {
            arr_slice = arr_slice.slice_move(slice);
        }

        writefn(arr_slice)
    }
}

impl<T: ToPyObject + Copy> PyNdArray<T> {
    /// getitem, as implemented in the rustpython interface
    pub fn getitem(&self, needle: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let last_slice = py_index_to_sliceinfo(needle, vm)?;

        let value = self.read(|sliced| {
            sliced.slice(&last_slice).get([]).copied().ok_or_else(|| {
                vm.new_runtime_error(format!("Array has dimension {:?}", sliced.dim()))
            })
        })?;

        Ok(value.to_pyobject(vm))
    }
}

impl<T: TryFromObject + Copy> PyNdArray<T> where PyNdArray<T>: GenericArray {
    /// setitem, as implemented in the rustpython interface
    pub fn setitem(
        &self,
        needle: PyObjectRef,
        value: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        let last_slice = py_index_to_sliceinfo(needle, vm)?;

        if let Some(other_array) = value.downcast_ref::<<Self as GenericArray>::PyArray>() {
            self.write(|mut sliced| {
            });

            return Ok(());
        }

        let value: T = TryFromObject::try_from_object(vm, value)?;
        self.write(|mut sliced| {
            let mut sliced = sliced.slice_mut(&last_slice);
            let dim = sliced.dim();
            sliced.fill(value);
        });

        Ok(())
    }
}

impl<T: Display> Display for PyNdArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.read(|slice| write!(f, "{slice}"))
    }
}

/// Converts a PyInt to an isize
pub fn pyint_to_isize(int: &PyInt, vm: &VirtualMachine) -> PyResult<isize> {
    int.as_bigint()
        .try_into()
        .map_err(|e| vm.new_runtime_error(format!("{e}")))
}

/// Converts a PyObject to an isize
pub fn py_obj_elem_to_isize(obj: &PyObject, vm: &VirtualMachine) -> PyResult<isize> {
    let int: &PyInt = obj
        .downcast_ref::<PyInt>()
        .ok_or_else(|| vm.new_runtime_error("Indices must be isize".to_string()))?;
    pyint_to_isize(int, vm)
}

/// Converts a PyObject to a SliceInfoElem
pub fn py_index_elem_to_sliceinfo_elem(
    elem: PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<SliceInfoElem> {
    if let Some(int) = elem.downcast_ref::<PyInt>() {
        return Ok(SliceInfoElem::Index(pyint_to_isize(int, vm)?));
    }

    if let Some(slice) = elem.downcast_ref::<PySlice>() {
        let stop = py_obj_elem_to_isize(&slice.stop, vm)?;
        let start = slice
            .start
            .as_ref()
            .map(|start| py_obj_elem_to_isize(start, vm))
            .transpose()?;
        let step = slice
            .step
            .as_ref()
            .map(|step| py_obj_elem_to_isize(step, vm))
            .transpose()?;
        return Ok(SliceInfoElem::Slice {
            start: start.unwrap_or(stop),
            step: step.unwrap_or(1),
            end: Some(stop),
        });
    }

    Err(vm.new_runtime_error("Unrecognized index {elem:?}".to_string()))
}

/// Converts a PyObject to a DynamicSlice
pub fn py_index_to_sliceinfo(shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<DynamicSlice> {
    if let Ok(single) = py_index_elem_to_sliceinfo_elem(shape.clone(), vm) {
        return Ok(DynamicSlice::try_from(vec![single]).unwrap());
    }

    if let Some(tuple) = shape.downcast_ref::<PyTuple>() {
        let indices: Vec<SliceInfoElem> = tuple
            .iter()
            .map(|member| py_index_elem_to_sliceinfo_elem(member.clone(), vm))
            .collect::<PyResult<_>>()?;
        return Ok(DynamicSlice::try_from(indices).unwrap());
    }

    Err(vm.new_runtime_error("Unrecognized index {shape:?}".to_string()))
}

/// Converts a PyObject shape to a Vec<usize>
pub fn py_shape_to_rust(shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
    if let Some(int) = shape.downcast_ref::<PyInt>() {
        return Ok(vec![int
            .as_bigint()
            .try_into()
            .map_err(|e| vm.new_runtime_error(format!("{e}")))?]);
    }

    shape
        .downcast::<PyTuple>()
        .map_err(|_| vm.new_runtime_error("Shape must be integer tuple".into()))?
        .iter()
        .map(|pyobject| {
            Ok(pyobject
                .downcast_ref::<PyInt>()
                .ok_or_else(|| vm.new_runtime_error("Indices must be usize".into()))?
                .as_bigint()
                .try_into()
                .map_err(|e| vm.new_runtime_error(format!("{e}")))?)
        })
        .collect::<PyResult<_>>()
}


