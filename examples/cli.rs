fn main() {
    let interpreter = rustpython::InterpreterConfig::new()
    .init_stdlib()
    .init_hook(Box::new(|vm| {
        vm.add_native_module("ndarray".to_owned(), Box::new(rustpython_ndarray::make_module));
    }))
    .interpreter();

    interpreter.enter(|vm| {
        let scope = vm.new_scope_with_builtins();
        let ndarray = vm.import("ndarray", 0).unwrap();
        scope.globals.set_item("ndarray", ndarray, vm).unwrap();
        rustpython::run_shell(vm, scope).unwrap();
    });
}
