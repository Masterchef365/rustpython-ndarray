#[rustpython_vm::pymodule]
pub mod rustpython_ndarray {
    use ndarray::ArrayD;
    use rustpython_vm::builtins::PyStrRef;
    use rustpython_vm::{pyclass, PyPayload};

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
            match self.inner {
                PyNdArrayType::Float32(_) => writeln!(f, "<PyNdArray(f32)>"),
                PyNdArrayType::Float64(_) => writeln!(f, "<PyNdArray(f64)>"),
            }
        }
    }

    #[pyfunction]
    fn do_thing(x: i32) -> i32 {
        x + 1
    }

    #[pyfunction]
    fn other_thing(s: PyStrRef) -> (String, usize) {
        let new_string = format!("hello from rust, {}!", s);
        let prev_len = s.as_str().len();
        (new_string, prev_len)
    }
}
