use std::sync::{Arc, Mutex};

use num_traits::cast::ToPrimitive;

use ndarray::{Dim, IxDynImpl, SliceInfoElem};
use rustpython_ndarray::PyNdArray;
use rustpython_vm::{
    builtins::{PyFloat, PyInt, PyListRef, PyModule, PyNone, PySlice},
    convert::ToPyObject,
    protocol::PyNumber,
    PyObject, PyObjectRef, PyRef, PyResult, TryFromBorrowedObject, TryFromObject, VirtualMachine,
};

pub fn make_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    rustpython_ndarray::make_module(vm)
}

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};

#[derive(Clone)]
enum GenericArray<F32, F64> {
    Float32(F32),
    Float64(F64),
}

type GenericArrayData = GenericArray<ArrayD<f32>, ArrayD<f64>>;
type GenericArrayDataView<'a> = GenericArray<ArrayViewD<'a, f32>, ArrayViewD<'a, f64>>;
type GenericArrayDataViewMut<'a> = GenericArray<ArrayViewMutD<'a, f32>, ArrayViewMutD<'a, f64>>;

impl std::fmt::Debug for GenericArrayData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericArray::Float32(arr) => writeln!(f, "<PyNdArray f32 {:?}>", arr.dim()),
            GenericArray::Float64(arr) => writeln!(f, "<PyNdArray f64 {:?}>", arr.dim()),
        }
    }
}

impl std::fmt::Debug for GenericArrayDataViewMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericArray::Float32(arr) => writeln!(f, "<PyNdArray f32 {:?}>", arr.dim()),
            GenericArray::Float64(arr) => writeln!(f, "<PyNdArray f64 {:?}>", arr.dim()),
        }
    }
}

impl std::fmt::Debug for GenericArrayDataView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenericArray::Float32(arr) => writeln!(f, "<PyNdArray f32 {:?}>", arr.dim()),
            GenericArray::Float64(arr) => writeln!(f, "<PyNdArray f64 {:?}>", arr.dim()),
        }
    }
}



impl GenericArrayDataViewMut<'_> {
    fn ndim(&self) -> usize {
        match self {
            GenericArray::Float32(f) => f.ndim(),
            GenericArray::Float64(f) => f.ndim(),
        }
    }

    fn set_item(&mut self, value: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        if let Some(other) = value.downcast_ref::<PyNdArray>() {
            let lck = other.data.lock().unwrap();
            let other_view = view(&lck, &other.slices);
            match (self, other_view) {
                (GenericArray::Float32(s), GenericArray::Float32(other)) => s.assign(&other),
                (GenericArray::Float64(s), GenericArray::Float64(other)) => s.assign(&other),
                (s, other) => return Err(vm.new_exception_msg(
                    vm.ctx.exceptions.runtime_error.to_owned(),
                    format!(
                        "Type mismatch, cannot assign {:?} to {:?}",
                        other, s,
                    ),
                )),
            }
        } else {
            if let Ok(scalar) = value.downcast::<PyFloat>() {
                let scalar = scalar.to_f64();
                match self {
                    GenericArray::Float32(f) => f.fill(scalar as f32),
                    GenericArray::Float64(f) => f.fill(scalar),
                };
            }
        }
        Ok(())
    }
}

impl GenericArrayDataView<'_> {
    fn ndim(&self) -> usize {
        match self {
            GenericArray::Float32(f) => f.ndim(),
            GenericArray::Float64(f) => f.ndim(),
        }
    }

    fn item(&self, vm: &VirtualMachine) -> PyObjectRef {
        assert_eq!(self.ndim(), 0);
        let idx = vec![0_usize; self.ndim()];
        match self {
            GenericArray::Float32(f) => vm.new_pyobj(f.get(idx.as_slice()).copied()),
            GenericArray::Float64(f) => vm.new_pyobj(f.get(idx.as_slice()).copied()),
        }
    }
}

impl GenericArrayData {
    fn view(&self) -> GenericArrayDataView<'_> {
        match self {
            GenericArray::Float32(data) => GenericArray::Float32(data.view()),
            GenericArray::Float64(data) => GenericArray::Float64(data.view()),
        }
    }

    fn view_mut(&mut self) -> GenericArrayDataViewMut<'_> {
        match self {
            GenericArray::Float32(data) => GenericArray::Float32(data.view_mut()),
            GenericArray::Float64(data) => GenericArray::Float64(data.view_mut()),
        }
    }

    fn item(&self, vm: &VirtualMachine) -> PyObjectRef {
        assert_eq!(self.ndim(), 0);
        let idx = vec![0_usize; self.ndim()];
        match self {
            GenericArrayData::Float32(f) => vm.new_pyobj(f.get(idx.as_slice()).copied()),
            GenericArrayData::Float64(f) => vm.new_pyobj(f.get(idx.as_slice()).copied()),
        }
    }

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

    fn ndim(&self) -> usize {
        match self {
            GenericArrayData::Float32(f) => f.ndim(),
            GenericArrayData::Float64(f) => f.ndim(),
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
        // TODO: Check for invalid types
        let start = slice.start.clone().and_then(|i| get_isize(i, vm).ok());
        let end = get_isize(slice.stop.clone(), vm).ok();
        let step = slice.step.clone().and_then(|i| get_isize(i, vm).ok());

        return Ok(SliceInfoElem::Slice {
            start: start.unwrap_or(0),
            end,
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
    use crate::{py_to_slice_info_elem, view, ArrayD, GenericArrayDataView, PySliceInfoElem};

    use super::GenericArrayData;

    use std::sync::{Arc, Mutex};

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
        Ok(PyNdArray::from_array(GenericArrayData::from_array(
            data, shape, vm,
        )?))
    }

    /// Provides a sliced representation of an array, where the slices are deferred until needed.
    #[pyattr]
    #[derive(PyPayload, Clone)]
    #[pyclass(module = "rustpython_ndarray", name = "PyNdArray")]
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
            let data_view = view(&lck, &zelf.slices);
            Ok(vm.ctx.new_str(match data_view {
                GenericArrayDataView::Float32(data) => format!("Float32 {}", data),
                GenericArrayDataView::Float64(data) => format!("Float64 {}", data),
            }))
        }

        #[pymethod]
        fn ndim(&self) -> usize {
            self.internal_ndim()
        }

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
}

fn generic_view<'a, T>(
    mut arr: ArrayViewD<'a, T>,
    slices: &[Vec<SliceInfoElem>],
) -> ArrayViewD<'a, T> {
    for slice in slices {
        arr = arr.slice_move(slice.as_slice());
    }
    arr
}

fn view<'a>(data: &'a GenericArrayData, slices: &[Vec<SliceInfoElem>]) -> GenericArrayDataView<'a> {
    match data {
        GenericArray::Float32(data) => GenericArray::Float32(generic_view(data.view(), slices)),
        GenericArray::Float64(data) => GenericArray::Float64(generic_view(data.view(), slices)),
    }
}

fn generic_view_mut<'a, T>(
    mut arr: ArrayViewMutD<'a, T>,
    slices: &[Vec<SliceInfoElem>],
) -> ArrayViewMutD<'a, T> {
    for slice in slices {
        arr = arr.slice_move(slice.as_slice());
    }
    arr
}

fn view_mut<'a>(
    data: &'a mut GenericArrayData,
    slices: &[Vec<SliceInfoElem>],
) -> GenericArrayDataViewMut<'a> {
    match data {
        GenericArray::Float32(data) => {
            GenericArray::Float32(generic_view_mut(data.view_mut(), slices))
        }
        GenericArray::Float64(data) => {
            GenericArray::Float64(generic_view_mut(data.view_mut(), slices))
        }
    }
}

fn parse_indices(needle: &PyObject, vm: &VirtualMachine) -> PyResult<Vec<SliceInfoElem>> {
    let indices: Vec<PySliceInfoElem> =
        TryFromBorrowedObject::try_from_borrowed_object(vm, needle)?;

    Ok(indices.into_iter().map(|idx| idx.elem).collect())
}

impl PyNdArray {
    fn from_array(inner: GenericArrayData) -> Self {
        Self {
            data: Arc::new(Mutex::new(inner)),
            slices: vec![],
        }
    }

    fn append_slice(&self, slice: Vec<SliceInfoElem>, vm: &VirtualMachine) -> PyResult<Self> {
        let mut slices = self.slices.clone();
        slices.push(slice);

        Ok(Self {
            data: self.data.clone(),
            slices,
        })
    }

    fn internal_ndim(&self) -> usize {
        self.data.lock().unwrap().ndim()
    }

    fn internal_getitem(&self, needle: &PyObject, vm: &VirtualMachine) -> PyResult {
        let slice = parse_indices(needle, vm)?;
        let with_appended_slice = self.append_slice(slice, vm)?;

        let lck = self.data.lock().unwrap();
        let arr_view = view(&lck, &with_appended_slice.slices);

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
        let slice = parse_indices(needle, vm)?;
        let with_appended_slice = self.append_slice(slice, vm)?;

        let mut lck = self.data.lock().unwrap();
        let mut arr_view = view_mut(&mut lck, &with_appended_slice.slices);

        arr_view.set_item(value, vm)
    }

    fn internal_iadd(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        /*
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
        */
        todo!()
    }
}
