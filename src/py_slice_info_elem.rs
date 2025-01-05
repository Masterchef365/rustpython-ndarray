use ndarray::SliceInfoElem;
use num_traits::cast::ToPrimitive;
use rustpython_vm::{
    builtins::{PyInt, PyNone, PySlice},
    PyObjectRef, PyResult, TryFromObject, VirtualMachine,
};

#[derive(Clone, Copy)]
pub struct PySliceInfoElem(pub SliceInfoElem);

impl TryFromObject for PySliceInfoElem {
    fn try_from_object(vm: &VirtualMachine, obj: PyObjectRef) -> PyResult<Self> {
        Ok(Self(py_to_slice_info_elem(obj, vm)?))
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
