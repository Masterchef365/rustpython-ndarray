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
