use num_traits::cast::ToPrimitive;

use ndarray::{ArrayD, ArrayViewD, SliceInfo, SliceInfoElem};
use rustpython_vm::{
    builtins::{PyInt, PyListRef, PyModule, PyNone, PySlice},
    convert::ToPyObject,
    PyObjectRef, PyRef, PyResult, TryFromObject, VirtualMachine,
};

pub fn make_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    rustpython_ndarray::make_module(vm)
}

#[derive(Clone)]
enum PyNdArrayType {
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
}

fn apply_array_view<'a, T>(arr: &'a ArrayD<T>, slices: &[Vec<SliceInfoElem>]) -> ArrayViewD<'a, T> {
    let mut view = arr.view();

    for slice in slices {
        let slice: SliceInfo<Vec<SliceInfoElem>, ndarray::Dim<ndarray::IxDynImpl>, ndarray::Dim<ndarray::IxDynImpl>> =
            unsafe { SliceInfo::new(slice.clone()) }.unwrap();
        view = view.slice(slice);
    }

    view
}

impl std::fmt::Debug for PyNdArrayType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PyNdArrayType::Float32(arr) => writeln!(f, "<PyNdArray f32 {:?}>", arr.dim()),
            PyNdArrayType::Float64(arr) => writeln!(f, "<PyNdArray f64 {:?}>", arr.dim()),
        }
    }
}

impl PyNdArrayType {
    fn from_array(data: PyListRef, shape: PyListRef, vm: &VirtualMachine) -> PyResult<Self> {
        let shape: Vec<usize> = TryFromObject::try_from_object(vm, shape.into())?;

        let data_f32: PyResult<Vec<f32>> = TryFromObject::try_from_object(vm, data.clone().into());

        if let Ok(data) = data_f32 {
            return Ok(Self::Float32(
                ArrayD::from_shape_vec(&*shape, data).map_err(|e| {
                    vm.new_exception_msg(vm.ctx.exceptions.runtime_error.to_owned(), e.to_string())
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
            PyNdArrayType::Float32(data) => Self::get_item_generic(vm, data, key),
            PyNdArrayType::Float64(data) => Self::get_item_generic(vm, data, key),
        }
    }

    fn get_item_generic<T: ToPyObject + Copy>(
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

    fn set_item(&mut self, vm: &VirtualMachine, key: &[usize], value: PyObjectRef) -> PyResult<()> {
        match self {
            PyNdArrayType::Float32(data) => {
                Self::set_item_internal(vm, data, key, TryFromObject::try_from_object(vm, value)?)?
            }
            PyNdArrayType::Float64(data) => {
                Self::set_item_internal(vm, data, key, TryFromObject::try_from_object(vm, value)?)?
            }
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

fn get_isize(obj: PyObjectRef, vm: &VirtualMachine) -> PyResult<isize> {
    let py_int = obj.downcast::<PyInt>().or_else(|_| {
        Err(vm.new_exception_msg(
            vm.ctx.exceptions.runtime_error.to_owned(),
            "Indices must be integers".to_string(),
        ))
    })?;

    py_int.as_bigint().to_isize().ok_or_else(|| {
        vm.new_exception_msg(
            vm.ctx.exceptions.runtime_error.to_owned(),
            "Index cannot convert to isize".to_string(),
        )
    })
}

#[derive(Clone, Copy)]
struct PySliceInfoElem {
    elem: SliceInfoElem,
}

impl TryFromObject for PySliceInfoElem {
    fn try_from_object(vm: &VirtualMachine, obj: PyObjectRef) -> PyResult<Self> {
        Ok(Self {
            elem: py_to_slice_info_elem(obj, vm)?,
        })
    }
}

fn py_to_slice_info_elem(obj: PyObjectRef, vm: &VirtualMachine) -> PyResult<SliceInfoElem> {
    if let Ok(index) = get_isize(obj.clone(), vm) {
        return Ok(SliceInfoElem::Index(index));
    }

    if let Ok(slice) = obj.clone().downcast::<PySlice>() {
        let start = slice.start.clone().map(|i| get_isize(i, vm)).transpose()?;
        let end = get_isize(slice.stop.clone(), vm)?;
        let step = slice.step.clone().map(|i| get_isize(i, vm)).transpose()?;

        // TODO: Is this right?
        return Ok(SliceInfoElem::Slice {
            start: start.unwrap_or(0),
            end: Some(end),
            step: step.unwrap_or(1),
        });
    }

    if let Ok(_) = obj.downcast::<PyNone>() {
        return Ok(SliceInfoElem::NewAxis);
    }

    Err(vm.new_exception_msg(
        vm.ctx.exceptions.runtime_error.to_owned(),
        "Invalid slice index type".to_string(),
    ))
}

#[rustpython_vm::pymodule]
pub mod rustpython_ndarray {
    use crate::{py_to_slice_info_elem, PySliceInfoElem};

    use super::PyNdArrayType;

    use std::cell::RefCell;
    use std::rc::Rc;

    use ndarray::{ArrayView, SliceInfoElem};
    use rustpython_vm::builtins::{PyFloat, PyListRef, PyStrRef};

    use rustpython_vm::protocol::{PyMappingMethods, PyNumberMethods};
    use rustpython_vm::types::{AsMapping, AsNumber};
    use rustpython_vm::{
        atomic_func, pyclass, PyObject, PyObjectRef, PyPayload, PyRef, PyResult,
        TryFromBorrowedObject, VirtualMachine,
    };

    #[pyfunction]
    fn array_from_list(
        data: PyListRef,
        shape: PyListRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        Ok(PyNdArray {
            inner: Rc::new(RefCell::new(PyNdArrayType::from_array(data, shape, vm)?)),
            slices: vec![],
        })
    }

    #[pyattr]
    #[derive(PyPayload, Clone)]
    #[pyclass(module = "rustpython_ndarray", name = "PyNdArray")]
    struct PyNdArray {
        inner: Rc<RefCell<PyNdArrayType>>,
        slices: Vec<Vec<SliceInfoElem>>,
    }

    impl std::fmt::Debug for PyNdArray {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.borrow().fmt(f)
        }
    }

    impl PyNdArray {
        fn add_slice(mut self, slice: Vec<PySliceInfoElem>) -> Self {
            let native_elems = slice.into_iter().map(|elem| elem.elem).collect();
            self.slices.push(native_elems);
            self
        }

        fn inner_getitem(&self, needle: &PyObject, vm: &VirtualMachine) -> PyResult {
            let indices: Vec<PySliceInfoElem> =
                TryFromBorrowedObject::try_from_borrowed_object(vm, needle)?;

            if indices
                .iter()
                .all(|idx| matches!(idx.elem, SliceInfoElem::Index(_)))
            {
                return Ok(todo!("Get single item"));
            }

            todo!("Try get slice")
        }

        fn inner_setitem(
            &self,
            needle: &PyObject,
            value: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<()> {
            let idx: Vec<usize> = TryFromBorrowedObject::try_from_borrowed_object(vm, needle)?;
            self.inner.borrow_mut().set_item(vm, &idx, value)
        }

        fn inner_iadd(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
            if let Ok(other) = other.clone().downcast::<PyFloat>() {
                match &mut *self.inner.borrow_mut() {
                    PyNdArrayType::Float32(data) => *data += other.to_f64() as f32,
                    PyNdArrayType::Float64(data) => *data += other.to_f64(),
                }
            }

            if let Ok(other) = other.clone().downcast::<PyNdArray>() {
                if Rc::ptr_eq(&other.inner, &self.inner) {
                    match &mut *self.inner.borrow_mut() {
                        PyNdArrayType::Float32(data) => *data *= 2.0,
                        PyNdArrayType::Float64(data) => *data *= 2.0,
                    }
                    return Ok(());
                }

                match (&mut *self.inner.borrow_mut(), &*other.inner.borrow()) {
                    (PyNdArrayType::Float32(data), PyNdArrayType::Float32(other)) => {
                        *data += other;
                    }
                    (PyNdArrayType::Float64(data), PyNdArrayType::Float64(other)) => {
                        *data += other;
                    }
                    _ => {
                        return Err(vm.new_exception_msg(
                            vm.ctx.exceptions.runtime_error.to_owned(),
                            "Array datatype mismatch".to_string(),
                        ))
                    }
                }
                Ok(())
            } else {
                Err(vm.new_exception_msg(
                    vm.ctx.exceptions.runtime_error.to_owned(),
                    format!(
                        "Cannot add {self:?} and {}",
                        other.obj_type().str(vm).unwrap()
                    ),
                ))
            }
        }
    }

    #[pyclass(with(AsMapping, AsNumber))]
    impl PyNdArray {
        #[pymethod(magic)]
        fn getitem(&self, needle: PyObjectRef, vm: &VirtualMachine) -> PyResult {
            self.inner_getitem(&*needle, vm)
        }

        #[pymethod(magic)]
        fn setitem(
            &self,
            needle: PyObjectRef,
            value: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<()> {
            self.inner_setitem(&*needle, value, vm)
        }

        #[pymethod(magic)]
        fn str(zelf: PyRef<Self>, vm: &VirtualMachine) -> PyResult<PyStrRef> {
            Ok(vm.ctx.new_str(format!("{:?}", zelf)))
        }

        #[pymethod(magic)]
        fn iadd(
            zelf: PyRef<Self>,
            other: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<PyRef<Self>> {
            zelf.inner_iadd(other, vm)?;
            Ok(zelf)
        }
    }

    impl AsMapping for PyNdArray {
        fn as_mapping() -> &'static PyMappingMethods {
            static AS_MAPPING: PyMappingMethods = PyMappingMethods {
                subscript: atomic_func!(|mapping, needle, vm| {
                    PyNdArray::mapping_downcast(mapping).inner_getitem(needle, vm)
                }),
                ass_subscript: atomic_func!(|mapping, needle, value, vm| {
                    let zelf = PyNdArray::mapping_downcast(mapping);
                    if let Some(value) = value {
                        zelf.inner_setitem(needle, value, vm)
                    } else {
                        //zelf.inner_delitem(needle, vm)
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
                    PyNdArray::number_downcast(a.to_number()).inner_iadd(b.to_owned(), vm)?;
                    Ok(a.to_owned())
                }),
                ..PyNumberMethods::NOT_IMPLEMENTED
            };
            &AS_MAPPING
        }
    }
}
