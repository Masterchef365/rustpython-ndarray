use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn, SliceInfo, SliceInfoElem};
use rustpython_vm::{
    builtins::{PyInt, PyNone, PySlice, PyTuple},
    convert::ToPyObject,
    PyObject, PyObjectRef, PyResult, TryFromObject, VirtualMachine,
};

use std::{
    fmt::Display,
    sync::{Arc, RwLock},
};

use crate::GenericArray;

pub type DynamicSlice = SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>;

/// Provides a sliced representation of an array, where the slices are deferred until needed.
#[derive(Debug, Clone)]
pub struct SlicedArcArray<T> {
    slices: Vec<DynamicSlice>,
    unsliced: Arc<RwLock<ndarray::ArrayD<T>>>,
}

impl<T> SlicedArcArray<T> {
    pub fn from_array(data: ndarray::ArrayD<T>) -> Self {
        Self {
            slices: vec![],
            unsliced: Arc::new(RwLock::new(data)),
        }
    }

    /// Borrow the entire array immutably to read it for a moment
    pub fn read<U>(&self, mut readfn: impl FnMut(ArrayViewD<'_, T>) -> U) -> U {
        let arr = self.unsliced.read().unwrap();

        let mut arr_slice = arr.view();

        for slice in &self.slices {
            arr_slice = arr_slice.slice_move(slice);
        }

        readfn(arr_slice)
    }

    /// Borrow the entire array mutably for a moment
    pub fn write<U>(&self, writefn: impl Fn(ArrayViewMutD<'_, T>) -> U) -> U {
        let mut arr = self.unsliced.write().unwrap();

        let mut arr_slice = arr.view_mut();

        for slice in &self.slices {
            arr_slice = arr_slice.slice_move(slice);
        }

        writefn(arr_slice)
    }

    pub fn append_slice(&self, slice: DynamicSlice, vm: &VirtualMachine) -> PyResult<Self> {
        if let Err(e) = self.read(|sliced| sliced.bounds_check(&slice)) {
            return Err(vm.new_index_error(format!("Slice out of bounds; {e}")));
        }

        let mut slices = self.slices.clone();
        slices.push(slice);

        Ok(Self {
            slices,
            unsliced: self.unsliced.clone(),
        })
    }

    pub fn ndim(&self) -> usize {
        self.read(|sliced| sliced.ndim())
    }

    pub fn shape(&self) -> Vec<usize> {
        self.read(|sliced| sliced.shape().to_vec())
    }

    pub fn length(&self) -> usize {
        self.read(|sliced| sliced.shape().get(0).copied().unwrap_or(1))
    }
}

impl<T: Clone> SlicedArcArray<T> {
    pub fn sliced_copy(&self) -> Self {
        self.read(|sliced| Self::from_array(sliced.to_owned()))
    }
}

impl<T: Display> SlicedArcArray<T>
where
    SlicedArcArray<T>: GenericArray,
{
    pub fn repr(&self) -> String {
        format!("array({}, dtype='{}')", self, Self::DTYPE.stringy_key())
    }
}

impl<T: ToPyObject + Copy> SlicedArcArray<T> {
    /// getitem, as implemented in the rustpython interface
    pub fn getitem(&self, needle: PyObjectRef, vm: &VirtualMachine) -> PyResult
    where
        SlicedArcArray<T>: GenericArray,
    {
        let slice = py_index_to_sliceinfo(needle, vm)?;
        let sliced_self = self.append_slice(slice, vm)?;

        sliced_self.read(|sliced_array| {
            if sliced_array.ndim() == 0 {
                Ok(sliced_array.get([]).copied().unwrap().to_pyobject(vm))
            } else {
                Ok(sliced_self.cast().to_pyobject(vm))
            }
        })
    }
}

impl<T: TryFromObject + Copy> SlicedArcArray<T>
where
    SlicedArcArray<T>: GenericArray,
{
    /// Fills the slice `needle` with `value` (casted to T)
    pub fn fill(&self, needle: DynamicSlice, value: T, vm: &VirtualMachine) -> PyResult<()> {
        let sliced_self = self.append_slice(needle, vm)?;

        sliced_self.write(|mut sliced| {
            sliced.fill(value);
            Ok(())
        })
    }

    /*
    /// Fills the slice `needle` with `value` (casted to T)
    pub fn set_array(
        &self,
        needle: DynamicSlice,
        value: SlicedArcArray<T>,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
    }
    */

    pub fn assign_fn<F, U>(
        &self,
        slice: DynamicSlice,
        other: SlicedArcArray<T>,
        vm: &VirtualMachine,
        f: F,
    ) -> PyResult<U>
    where
        F: Fn(ArrayViewMutD<'_, T>, ArrayViewD<'_, T>, &VirtualMachine) -> PyResult<U>,
    {
        // Check if we're copying from a slice of ourself ...
        if Arc::ptr_eq(&self.unsliced, &other.unsliced) {
            // TODO: THIS IS MEMORY INTENSIVE AND SLOW!!
            let copied = other.read(|us| us.to_owned());
            self.append_slice(slice, vm)?.write(|other_us| {
                if other_us.shape() != copied.shape() {
                    return Err(vm.new_runtime_error(format!(
                        "Attempted to assign shape {:?} to shape {:?}",
                        copied.shape(),
                        other_us.shape(),
                    )));
                }

                f(other_us, copied.view(), vm)
            })
        } else {
            self.append_slice(slice, vm)?.write(|mut us| {
                other.read(|them| {
                    if us.shape() != them.shape() {
                        return Err(vm.new_runtime_error(format!(
                            "Attempted to assign shape {:?} to shape {:?}",
                            them.shape(),
                            us.shape(),
                        )));
                    }

                    f(us.view_mut(), them.view(), vm)
                })
            })
        }
    }
}

impl<T: Display> Display for SlicedArcArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.read(|slice| write!(f, "{slice}"))
    }
}

/// Converts a PyInt to an isize
pub fn pyint_to_isize(int: &PyInt, vm: &VirtualMachine) -> PyResult<isize> {
    int.as_bigint()
        .try_into()
        .map_err(|e| vm.new_runtime_error(format!("Bigint cast {e}")))
}

/// Converts a PyObject to an isize
pub fn py_obj_elem_to_isize(obj: &PyObject, vm: &VirtualMachine) -> PyResult<Option<isize>> {
    if obj.downcast_ref::<PyNone>().is_some() {
        return Ok(None);
    }

    let int: &PyInt = obj
        .downcast_ref::<PyInt>()
        .ok_or_else(|| vm.new_runtime_error("Indices must be isize".to_string()))?;

    pyint_to_isize(int, vm).map(Some)
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
            .and_then(|start| py_obj_elem_to_isize(start, vm).transpose())
            .transpose()?;
        let step = slice
            .step
            .as_ref()
            .and_then(|step| py_obj_elem_to_isize(step, vm).transpose())
            .transpose()?;
        return Ok(SliceInfoElem::Slice {
            start: start.unwrap_or(0),
            step: step.unwrap_or(1),
            end: stop,
        });
    }

    if let Some(_) = elem.downcast_ref::<PyNone>() {
        return Ok(SliceInfoElem::NewAxis);
    }

    Err(vm.new_runtime_error(format!("Unrecognized index {elem:?}")))
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

    Err(vm.new_runtime_error(format!("Unrecognized sliceinfo index {shape:?}")))
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
