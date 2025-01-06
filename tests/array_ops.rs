use std::ops::Deref;

use rustpython_vm::{
    builtins::{PyBaseExceptionRef, PyNone},
    Interpreter, PyPayload, PyResult, TryFromBorrowedObject, VirtualMachine,
};

fn get_interpreter() -> Interpreter {
    rustpython::InterpreterConfig::new()
        .init_stdlib()
        .init_hook(Box::new(|vm| {
            vm.add_native_module(
                "ndarray".to_owned(),
                Box::new(rustpython_ndarray::make_module),
            );
        }))
        .interpreter()
}

#[track_caller]
fn run_code(source: &'static str) {
    let interp = get_interpreter();
    interp.enter(|vm| {
        let scope = vm.new_scope_with_builtins();
        vm.run_block_expr(scope, &source)
            .map_err(|e| write_exception(e, vm))
            .unwrap();
    })
}

#[test]
fn basic() {
    run_code("pass");
}

#[test]
#[should_panic]
fn basicer() {
    run_code("fail");
}


fn write_exception(excp: PyBaseExceptionRef, vm: &VirtualMachine) -> String {
    let mut s = String::new();
    vm.write_exception(&mut s, &excp).unwrap();
    s
}
