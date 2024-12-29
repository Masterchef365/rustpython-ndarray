use rustpython_vm::{builtins::PyModule, PyRef, VirtualMachine};

pub fn make_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    rustpython_ndarray::make_module(vm)
}

#[rustpython_vm::pymodule]
pub mod rustpython_ndarray {
    use std::cell::RefCell;
    use std::rc::Rc;

    use ndarray::ArrayD;
    use rustpython_vm::builtins::{PyBaseExceptionRef, PyList, PyListRef, PyStrRef};
    use rustpython_vm::convert::ToPyObject;
    use rustpython_vm::object::Traverse;
    use rustpython_vm::protocol::PyNumber;
    use rustpython_vm::types::AsNumber;
    use rustpython_vm::{
        pyclass, PyObject, PyObjectRef, PyPayload, PyResult, TryFromBorrowedObject, TryFromObject,
        VirtualMachine,
    };

    #[pyfunction]
    fn array_from_list(
        data: PyListRef,
        shape: PyListRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        Ok(PyNdArray {
            inner: Rc::new(RefCell::new(PyNdArrayType::from_array(data, shape, vm)?)),
        })
    }

    #[pyattr]
    #[derive(PyPayload, Clone)]
    #[pyclass(module = "rustpython_ndarray", name = "PyNdArray")]
    struct PyNdArray {
        inner: Rc<RefCell<PyNdArrayType>>,
    }

    #[derive(Clone)]
    enum PyNdArrayType {
        Float32(ArrayD<f32>),
        Float64(ArrayD<f64>),
    }

    impl std::fmt::Debug for PyNdArray {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match &*self.inner.borrow() {
                PyNdArrayType::Float32(arr) => writeln!(f, "<PyNdArray f32 {:?}>", arr.dim()),
                PyNdArrayType::Float64(arr) => writeln!(f, "<PyNdArray f64 {:?}>", arr.dim()),
            }
        }
    }

    impl PyNdArrayType {
        fn from_array(data: PyListRef, shape: PyListRef, vm: &VirtualMachine) -> PyResult<Self> {
            let shape: Vec<usize> = TryFromObject::try_from_object(vm, shape.into())?;

            let data_f32: PyResult<Vec<f32>> =
                TryFromObject::try_from_object(vm, data.clone().into());

            if let Ok(data) = data_f32 {
                return Ok(Self::Float32(
                    ArrayD::from_shape_vec(&*shape, data).map_err(|e| {
                        vm.new_exception_msg(
                            vm.ctx.exceptions.runtime_error.to_owned(),
                            e.to_string(),
                        )
                    })?,
                ));
            }

            let data_f64: Vec<f64> = TryFromObject::try_from_object(vm, data.into())?;
            Ok(Self::Float64(
                ArrayD::from_shape_vec(shape, data_f64).map_err(|e| {
                    vm.new_exception_msg(vm.ctx.exceptions.runtime_error.to_owned(), e.to_string())
                })?,
            ))
        }

        fn get_item(&self, vm: &VirtualMachine, key: &[usize]) -> PyResult {
            match self {
                PyNdArrayType::Float32(data) => Self::get_item_internal(vm, data, key),
                PyNdArrayType::Float64(data) => Self::get_item_internal(vm, data, key),
            }
        }

        fn get_item_internal<T: ToPyObject + Copy>(
            vm: &VirtualMachine,
            data: &ArrayD<T>,
            key: &[usize],
        ) -> PyResult {
            Ok(vm.new_pyobj(*data.get(&*key).ok_or_else(|| {
                vm.new_exception_msg(
                    vm.ctx.exceptions.index_error.to_owned(),
                    format!(
                        "Index {key:?} was out of bounds for array {:?}",
                        data.shape()
                    )
                    .into(),
                )
            })?))
        }

        fn set_item(
            &mut self,
            vm: &VirtualMachine,
            key: &[usize],
            value: PyObjectRef,
        ) -> PyResult<()> {
            match self {
                PyNdArrayType::Float32(data) => Self::set_item_internal(
                    vm,
                    data,
                    key,
                    TryFromObject::try_from_object(vm, value)?,
                )?,
                PyNdArrayType::Float64(data) => Self::set_item_internal(
                    vm,
                    data,
                    key,
                    TryFromObject::try_from_object(vm, value)?,
                )?,
            }

            Ok(())
        }

        fn set_item_internal<T: ToPyObject + Copy>(
            vm: &VirtualMachine,
            data: &mut ArrayD<T>,
            key: &[usize],
            value: T,
        ) -> PyResult<()> {
            if let Some(data_val) = data.get_mut(&*key) {
                *data_val = value;
                Ok(())
            } else {
                Err(vm.new_exception_msg(
                    vm.ctx.exceptions.index_error.to_owned(),
                    format!(
                        "Index {key:?} was out of bounds for array {:?}",
                        data.shape()
                    )
                    .into(),
                ))
            }
        }
    }

    #[pyclass]
    impl PyNdArray {
        #[pymethod(name = "__getitem__")]
        fn get_item(&self, key: Vec<usize>, vm: &VirtualMachine) -> PyResult {
            self.inner.borrow().get_item(vm, &key)
        }

        #[pymethod(name = "__setitem__")]
        fn set_item(&self, key: Vec<usize>, value: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
            self.inner.borrow_mut().set_item(vm, &key, value)
        }
    }
}
