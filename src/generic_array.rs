use std::sync::{Arc, Mutex};

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use ndarray::{Dim, IxDynImpl, SliceInfoElem};
use rustpython_vm::{
    builtins::{PyFloat, PyListRef},
    PyObjectRef, PyResult, TryFromObject, VirtualMachine,
};

use crate::rustpython_ndarray::PyNdArray;


#[derive(Clone)]
pub enum GenericArray<F32, F64> {
    Float32(F32),
    Float64(F64),
}

pub type GenericArrayData = GenericArray<ArrayD<f32>, ArrayD<f64>>;
pub type GenericArrayDataView<'a> = GenericArray<ArrayViewD<'a, f32>, ArrayViewD<'a, f64>>;
pub type GenericArrayDataViewMut<'a> = GenericArray<ArrayViewMutD<'a, f32>, ArrayViewMutD<'a, f64>>;


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
    pub fn ndim(&self) -> usize {
        match self {
            GenericArray::Float32(f) => f.ndim(),
            GenericArray::Float64(f) => f.ndim(),
        }
    }

    pub fn fill(&mut self, scalar: f64) {
        match self {
            GenericArray::Float32(f) => f.fill(scalar as f32),
            GenericArray::Float64(f) => f.fill(scalar),
        }
    }

    pub fn set_array(&mut self, source: GenericArrayDataView<'_>, vm: &VirtualMachine) -> PyResult<()> {
        match (self, source) {
            (GenericArray::Float32(s), GenericArray::Float32(other)) => Ok(s.assign(&other)),
            (GenericArray::Float64(s), GenericArray::Float64(other)) => Ok(s.assign(&other)),
            (s, other) => Err(vm.new_exception_msg(
                    vm.ctx.exceptions.runtime_error.to_owned(),
                    format!(
                        "Type mismatch, cannot assign {:?} to {:?}",
                        other, s,
                    ),
            )),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            GenericArray::Float32(f) => f.shape(),
            GenericArray::Float64(f) => f.shape(),
        }
    }
}

impl GenericArrayDataView<'_> {
    pub fn ndim(&self) -> usize {
        match self {
            GenericArray::Float32(f) => f.ndim(),
            GenericArray::Float64(f) => f.ndim(),
        }
    }

    pub fn item(&self, vm: &VirtualMachine) -> PyObjectRef {
        assert_eq!(self.ndim(), 0);
        let idx = vec![0_usize; self.ndim()];
        match self {
            GenericArray::Float32(f) => vm.new_pyobj(f.get(idx.as_slice()).copied()),
            GenericArray::Float64(f) => vm.new_pyobj(f.get(idx.as_slice()).copied()),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            GenericArray::Float32(f) => f.shape(),
            GenericArray::Float64(f) => f.shape(),
        }
    }
}

impl GenericArrayData {
    pub fn view(&self) -> GenericArrayDataView<'_> {
        match self {
            GenericArray::Float32(data) => GenericArray::Float32(data.view()),
            GenericArray::Float64(data) => GenericArray::Float64(data.view()),
        }
    }

    pub fn view_mut(&mut self) -> GenericArrayDataViewMut<'_> {
        match self {
            GenericArray::Float32(data) => GenericArray::Float32(data.view_mut()),
            GenericArray::Float64(data) => GenericArray::Float64(data.view_mut()),
        }
    }

    pub fn item(&self, vm: &VirtualMachine) -> PyObjectRef {
        assert_eq!(self.ndim(), 0);
        let idx = vec![0_usize; self.ndim()];
        match self {
            GenericArrayData::Float32(f) => vm.new_pyobj(f.get(idx.as_slice()).copied()),
            GenericArrayData::Float64(f) => vm.new_pyobj(f.get(idx.as_slice()).copied()),
        }
    }

    pub fn from_array(data: PyListRef, shape: PyListRef, vm: &VirtualMachine) -> PyResult<Self> {
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

    pub fn ndim(&self) -> usize {
        match self {
            GenericArrayData::Float32(f) => f.ndim(),
            GenericArrayData::Float64(f) => f.ndim(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            GenericArray::Float32(f) => f.shape(),
            GenericArray::Float64(f) => f.shape(),
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

pub fn view<'a>(data: &'a GenericArrayData, slices: &[Vec<SliceInfoElem>]) -> GenericArrayDataView<'a> {
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

pub fn view_mut<'a>(
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
