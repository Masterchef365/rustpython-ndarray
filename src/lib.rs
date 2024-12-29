use rustpython_vm::{builtins::PyModule, PyRef, VirtualMachine};

pub fn make_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    rustpython_ndarray::make_module(vm)
}

#[rustpython_vm::pymodule]
pub mod rustpython_ndarray {
    use ndarray::ArrayD;
    use rustpython_vm::builtins::{PyList, PyListRef, PyStrRef};
    use rustpython_vm::object::Traverse;
    use rustpython_vm::types::AsNumber;
    use rustpython_vm::{
        pyclass, PyObjectRef, PyPayload, PyResult, TryFromBorrowedObject, TryFromObject,
        VirtualMachine,
    };

    #[pyfunction]
    fn array_from_list(
        data: PyListRef,
        shape: PyListRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        Ok(PyNdArray {
            inner: PyNdArrayType::from_array(data, shape, vm)?,
        })
    }

    #[pyattr]
    #[derive(PyPayload, Clone)]
    #[pyclass(module = "rustpython_ndarray", name = "PyNdArray")]
    struct PyNdArray {
        inner: PyNdArrayType,
    }

    #[pyclass]
    impl PyNdArray {}

    #[derive(Clone)]
    enum PyNdArrayType {
        Float32(ArrayD<f32>),
        Float64(ArrayD<f64>),
    }

    impl std::fmt::Debug for PyNdArray {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match &self.inner {
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
    }
}
