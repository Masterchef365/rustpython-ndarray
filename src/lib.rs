#![allow(warnings)]

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn, SliceInfo, SliceInfoElem};
use rustpython_vm::{
    atomic_func,
    builtins::{PyInt, PyList, PyModule, PySlice, PyStr, PyTuple},
    class::PyClassImpl,
    convert::ToPyObject,
    object::PyObjectPayload,
    protocol::PyMappingMethods,
    types::AsMapping,
    PyObject, PyObjectRef, PyRef, PyResult, TryFromObject, VirtualMachine,
};

pub fn make_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    let mut module = pyndarray::make_module(vm);
    //module.set_attr("PyNdArrayFloat32", pyndarray::PyNdArrayFloat32::make_class(&vm.ctx), vm);
    pyndarray::PyNdArrayFloat32::make_class(&vm.ctx);
    pyndarray::PyNdArrayFloat64::make_class(&vm.ctx);

    module
}

use std::{
    borrow::Borrow, fmt::Display, str::FromStr, sync::{Arc, RwLock}
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float64,
}

#[rustpython_vm::pymodule]
pub mod pyndarray {
    use super::*;
    use builtins::{PyListRef, PyStr, PyStrRef};
    use function::{KwArgs, OptionalArg};
    use rustpython_vm::types::AsMapping;
    use rustpython_vm::*;

    macro_rules! build_pyarray {
        ($primitive:ident, $dtype:ident) => {
            #[derive(PyPayload, Clone, Debug)]
            #[pyclass(module = "pyndarray", name)]
            pub(crate) struct $dtype {
                pub(crate) arr: PyNdArray<$primitive>,
            }

            //#[pyclass]
            #[pyclass(with(AsMapping))]
            impl $dtype {
                #[pymethod(magic)]
                fn getitem(&self, needle: PyObjectRef, vm: &VirtualMachine) -> PyResult {
                    self.arr.getitem(needle, vm)
                }

                #[pymethod(magic)]
                fn setitem(
                    &self,
                    needle: PyObjectRef,
                    value: PyObjectRef,
                    vm: &VirtualMachine,
                ) -> PyResult<()> {
                    self.arr.setitem(needle, value, vm)
                }

                #[pymethod(magic)]
                fn str(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult<PyStrRef> {
                    Ok(vm.ctx.new_str(zelf.arr.to_string()))
                }
            }

            impl AsMapping for $dtype {
                fn as_mapping() -> &'static PyMappingMethods {
                    static AS_MAPPING: PyMappingMethods = PyMappingMethods {
                        subscript: atomic_func!(|mapping, needle, vm| {
                            $dtype::mapping_downcast(mapping).getitem(needle.to_pyobject(vm), vm)
                        }),
                        ass_subscript: atomic_func!(|mapping, needle, value, vm| {
                            let zelf = $dtype::mapping_downcast(mapping);
                            if let Some(value) = value {
                                zelf.setitem(needle.to_pyobject(vm), value, vm)
                            } else {
                                //zelf.internal_delitem(needle, vm)
                                Err(vm.new_exception_msg(
                                    vm.ctx.exceptions.runtime_error.to_owned(),
                                    "Arrays do not support delete".to_string(),
                                ))
                            }
                        }),
                        length: atomic_func!(|_mapping, vm| {
                            Err(vm.new_exception_msg(
                                vm.ctx.exceptions.runtime_error.to_owned(),
                                "Arrays do not support len()".to_string(),
                            ))
                        }),
                    };
                    &AS_MAPPING
                }
            }

            impl From<PyNdArray<$primitive>> for $dtype {
                fn from(arr: PyNdArray<$primitive>) -> Self {
                    Self { arr }
                }
            }
        };
    }

    build_pyarray!(f32, PyNdArrayFloat32);
    build_pyarray!(f64, PyNdArrayFloat64);

    #[pyfunction]
    fn zeros(shape: PyObjectRef, mut kw: KwArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let dtype = kw.pop_kwarg("dtype");

        let shape = py_shape_to_rust(shape.into(), vm)?;

        let dtype = dtype
            .map(|dtype| {
                DataType::from_pyobject(&dtype)
                    .ok_or_else(|| vm.new_runtime_error(format!("Unrecognized dtype {dtype:?}")))
            })
            .transpose()?;

        match dtype {
            Some(DataType::Float64) => Ok(PyNdArrayFloat64::from(PyNdArray::from_array(
                ndarray::ArrayD::zeros(shape),
            ))
            .to_pyobject(vm)),
            None | Some(DataType::Float32) => Ok(PyNdArrayFloat32::from(PyNdArray::from_array(
                ndarray::ArrayD::zeros(shape),
            ))
            .to_pyobject(vm)),
        }
    }
}

type DynamicSlice = SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn>;

fn pyint_to_isize(int: &PyInt, vm: &VirtualMachine) -> PyResult<isize> {
    int.as_bigint()
        .try_into()
        .map_err(|e| vm.new_runtime_error(format!("{e}")))
}

fn py_obj_elem_to_isize(obj: &PyObject, vm: &VirtualMachine) -> PyResult<isize> {
    let int: &PyInt = obj
        .downcast_ref::<PyInt>()
        .ok_or_else(|| vm.new_runtime_error("Indices must be isize".to_string()))?;
    pyint_to_isize(int, vm)
}

fn py_index_elem_to_sliceinfo_elem(
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
        })
    }

    Err(vm.new_runtime_error("Unrecognized index {elem:?}".to_string()))
}

fn py_index_to_sliceinfo(shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<DynamicSlice> {
    if let Some(int) = shape.downcast_ref::<PyInt>() {
        let idx: isize = int
            .as_bigint()
            .try_into()
            .map_err(|e| vm.new_runtime_error(format!("{e}")))?;
        return Ok(DynamicSlice::try_from(vec![SliceInfoElem::Index(idx)]).unwrap());
    }

    Err(vm.new_runtime_error("nah".to_string()))
}

fn py_shape_to_rust(shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
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

/// Provides a sliced representation of an array, where the slices are deferred until needed.
#[derive(Debug, Clone)]
pub struct PyNdArray<T> {
    pub slices: Vec<DynamicSlice>,
    pub data: Arc<RwLock<ndarray::ArrayD<T>>>,
}

impl<T> PyNdArray<T> {
    pub fn from_array(data: ndarray::ArrayD<T>) -> Self {
        Self {
            slices: vec![],
            data: Arc::new(RwLock::new(data)),
        }
    }

    pub fn read<U>(&self, mut readfn: impl FnMut(ArrayViewD<'_, T>) -> U) -> U {
        let mut arr = self.data.read().unwrap();

        let default_slice = vec![SliceInfoElem::from(..); arr.ndim()];
        let default_slice = DynamicSlice::try_from(default_slice).unwrap();

        let mut arr_slice = arr.slice(default_slice);

        for slice in &self.slices {
            arr_slice = arr_slice.slice_move(slice);
        }

        readfn(arr_slice)
    }

    pub fn write<U>(&self, mut writefn: impl Fn(ArrayViewMutD<'_, T>) -> U) -> U {
        let mut arr = self.data.write().unwrap();

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
    fn getitem(&self, needle: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let data = self.data.read().unwrap();
        /*
        for slice in self.slices {
            data.slice(&slice)
        }
        */
        let indices = py_shape_to_rust(needle, vm)?;
        let value = data
            .get(indices.as_slice())
            .ok_or_else(|| vm.new_runtime_error("Invalid index".into()))?;

        Ok(value.to_pyobject(vm))
    }
}

impl<T: TryFromObject + Copy> PyNdArray<T> {
    fn setitem(
        &self,
        needle: PyObjectRef,
        value: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        let indices = py_shape_to_rust(needle, vm)?;
        let mut data = self.data.write().unwrap();
        /*
        for slice in self.slices {
            data.slice(&slice)
        }
        */

        let cell = data
            .get_mut(indices.as_slice())
            .ok_or_else(|| vm.new_runtime_error("Invalid index".into()))?;
        *cell = TryFromObject::try_from_object(vm, value)?;

        Ok(())
    }
}

impl<T: Display> Display for PyNdArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.read(|slice| write!(f, "{slice}"))
    }
}

impl DataType {
    fn from_pyobject(obj: &PyObject) -> Option<Self> {
        // TODO: Casts from float and integer primitives
        match obj.downcast_ref::<PyStr>()?.as_str() {
            "float64" => Some(Self::Float64),
            "float32" => Some(Self::Float32),
            _ => None,
        }
    }
}

/*
use std::sync::{Arc, Mutex};
use ndarray::SliceInfoElem;
use pyndarray::PyNdArray;
use rustpython_vm::atomic_func;
use rustpython_vm::builtins::PyBaseExceptionRef;
use rustpython_vm::protocol::{PyMappingMethods, PyNumberMethods};
use rustpython_vm::types::{AsMapping, AsNumber};
use rustpython_vm::{
    builtins::{PyFloat, PyModule},
    PyObject, PyObjectRef, PyRef, PyResult, TryFromBorrowedObject, VirtualMachine,
};

pub fn make_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    pyndarray::make_module(vm)
}

mod py_slice_info_elem;
use py_slice_info_elem::PySliceInfoElem;

mod generic_array;
use generic_array::*;

#[rustpython_vm::pymodule]
pub mod pyndarray {
    use crate::{runtime_error, view};

    use crate::generic_array::{self, *};

    use std::sync::{Arc, Mutex};

    use ndarray::SliceInfoElem;
    use rustpython_vm::builtins::{PyListRef, PyStrRef};
    use rustpython_vm::types::{AsMapping, AsNumber};
    use rustpython_vm::{pyclass, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};

    #[pyfunction]
    fn array_from_list(
        data: PyListRef,
        shape: PyListRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        Ok(PyNdArray::from_array(GenericArrayData::from_array(
            data, shape, vm,
        )?))
    }

    /// Provides a sliced representation of an array, where the slices are deferred until needed.
    #[pyattr]
    #[derive(PyPayload, Clone)]
    #[pyclass(module = "pyndarray", name = "PyNdArray")]
    pub(crate) struct PyNdArray {
        pub(crate) data: Arc<Mutex<GenericArrayData>>,
        pub(crate) slices: Vec<Vec<SliceInfoElem>>,
    }

    impl std::fmt::Debug for PyNdArray {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.data.fmt(f)
        }
    }

    #[pyclass(with(AsMapping, AsNumber))]
    impl PyNdArray {
        #[pymethod(magic)]
        fn getitem(&self, needle: PyObjectRef, vm: &VirtualMachine) -> PyResult {
            self.internal_getitem(&*needle, vm)
        }

        #[pymethod(magic)]
        fn setitem(
            &self,
            needle: PyObjectRef,
            value: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<()> {
            self.internal_setitem(&*needle, value, vm)
        }

        #[pymethod(magic)]
        fn str(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult<PyStrRef> {
            let lck = zelf.data.as_ref().lock().unwrap();
            let data_view =
                generic_array::view(&lck, &zelf.slices).map_err(|e| runtime_error(e, vm))?;
            Ok(vm.ctx.new_str(match data_view {
                GenericArray::Float32(data) => format!("Float32 {}", data),
                GenericArray::Float64(data) => format!("Float64 {}", data),
            }))
        }

        #[pymethod]
        fn ndim(&self, vm: &VirtualMachine) -> PyResult<usize> {
            let lck = self.data.lock().unwrap();
            let data_view = view(&lck, &self.slices).map_err(|e| runtime_error(e, vm))?;
            Ok(data_view.ndim())
        }

        /*
        #[pymethod]
        fn shape(&self) -> Vec<usize> {
            self.data.lock().unwrap().view().shape().to_vec()
        }
        */

        #[pymethod(magic)]
        fn iadd(
            zelf: PyRef<Self>,
            other: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<PyRef<Self>> {
            zelf.internal_iadd(other, vm)?;
            Ok(zelf)
        }
    }
}

impl AsMapping for PyNdArray {
    fn as_mapping() -> &'static PyMappingMethods {
        static AS_MAPPING: PyMappingMethods = PyMappingMethods {
            subscript: atomic_func!(|mapping, needle, vm| {
                PyNdArray::mapping_downcast(mapping).internal_getitem(needle, vm)
            }),
            ass_subscript: atomic_func!(|mapping, needle, value, vm| {
                let zelf = PyNdArray::mapping_downcast(mapping);
                if let Some(value) = value {
                    zelf.internal_setitem(needle, value, vm)
                } else {
                    //zelf.internal_delitem(needle, vm)
                    Err(vm.new_exception_msg(
                        vm.ctx.exceptions.runtime_error.to_owned(),
                        "Arrays do not support delete".to_string(),
                    ))
                }
            }),
            length: atomic_func!(|_mapping, vm| {
                Err(vm.new_exception_msg(
                    vm.ctx.exceptions.runtime_error.to_owned(),
                    "Arrays do not support len()".to_string(),
                ))
            }),
        };
        &AS_MAPPING
    }
}

impl AsNumber for PyNdArray {
    fn as_number() -> &'static rustpython_vm::protocol::PyNumberMethods {
        static AS_MAPPING: PyNumberMethods = PyNumberMethods {
            inplace_add: Some(|a, b, vm| {
                PyNdArray::number_downcast(a.to_number()).internal_iadd(b.to_owned(), vm)?;
                Ok(a.to_owned())
            }),
            ..PyNumberMethods::NOT_IMPLEMENTED
        };
        &AS_MAPPING
    }
}

fn parse_indices(needle: &PyObject, vm: &VirtualMachine) -> PyResult<Vec<SliceInfoElem>> {
    let indices: Vec<PySliceInfoElem> =
        TryFromBorrowedObject::try_from_borrowed_object(vm, needle)?;

    Ok(indices.into_iter().map(|idx| idx.0).collect())
}

impl PyNdArray {
    fn from_array(inner: GenericArrayData) -> Self {
        Self {
            data: Arc::new(Mutex::new(inner)),
            slices: vec![],
        }
    }

    fn append_slice(&self, slice: Vec<SliceInfoElem>) -> Self {
        let mut slices = self.slices.clone();
        slices.push(slice);

        Self {
            data: self.data.clone(),
            slices,
        }
    }

    fn internal_getitem(&self, needle: &PyObject, vm: &VirtualMachine) -> PyResult {
        let slice = parse_indices(needle, vm)?;
        let with_appended_slice = self.append_slice(slice);

        let lck = self.data.lock().unwrap();
        let arr_view = view(&lck, &with_appended_slice.slices).map_err(|e| runtime_error(e, vm))?;

        if arr_view.ndim() == 0 {
            Ok(arr_view.item(vm))
        } else {
            Ok(vm.new_pyobj(with_appended_slice))
        }
    }

    fn internal_setitem(
        &self,
        needle: &PyObject,
        value: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        // Get the indices and slice ourself with them
        let slice = parse_indices(needle, vm)?;
        let with_appended_slice = self.append_slice(slice);

        let mut lck = self.data.lock().unwrap();

        // If we're assigning from a slice of ourself, make a silent clone
        // TODO: Slow(?)
        let mut self_clone: Option<GenericArrayData> = None;
        if let Ok(other) = value.clone().downcast::<PyNdArray>() {
            if Arc::ptr_eq(&self.data, &other.data) {
                self_clone = Some(lck.clone());
            }
        }

        // View the array at the slice
        let mut arr_view =
            view_mut(&mut lck, &with_appended_slice.slices).map_err(|e| runtime_error(e, vm))?;

        // If it's a scalar, fill with it
        if let Ok(number) = value.clone().downcast::<PyFloat>() {
            arr_view.fill(number.to_f64());
            return Ok(());
        }

        // If it's another array ....
        if let Ok(other) = value.clone().downcast::<PyNdArray>() {
            // If it's us, use the clone we made
            if let Some(self_clone) = self_clone {
                let self_arr_view =
                    view(&self_clone, &other.slices).map_err(|e| runtime_error(e, vm))?;
                arr_view.set_array(self_arr_view, vm)?;
            } else {
                // If it's another array, slice it
                let mut lck = other.data.lock().unwrap();
                let other_arr_view =
                    view(&mut lck, &other.slices).map_err(|e| runtime_error(e, vm))?;
                arr_view.set_array(other_arr_view, vm)?;
            }

            return Ok(());
        }

        return Err(vm.new_exception_msg(
            vm.ctx.exceptions.runtime_error.to_owned(),
            format!("Cannot set array to value of type {}", value.class().name())
        ));
    }

    fn internal_iadd(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        todo!()
    }
}

fn runtime_error(s: String, vm: &VirtualMachine) -> PyBaseExceptionRef {
    vm.new_exception_msg(vm.ctx.exceptions.runtime_error.to_owned(), s)
}
*/
