"""Standard run/test targets for a kernel registered beside its assembly."""
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python_pytest//python_pytest:defs.bzl", "py_pytest_test")


def ipu_app(name, kernel_package, deps, data = []):
    """Declare one registered kernel with its adjacent test.py and .asm.

    kernel_package is a source path, for example src/ipu_apps/softmax/softmax_rows.
    The shared frontend selects the exact SPEC.name supplied as name.
    """
    kernel_data = data + [kernel_package + "/" + name + ".asm"]
    py_binary(
        name = name,
        srcs = ["src/ipu_apps/kernel_registry/runner.py"],
        main = "src/ipu_apps/kernel_registry/runner.py",
        args = ["--kernel", name],
        data = kernel_data,
        deps = deps,
        legacy_create_init = False,
    )
    py_pytest_test(
        name = "test_" + name,
        srcs = [kernel_package + "/test.py"],
        data = kernel_data,
        deps = deps,
        legacy_create_init = False,
    )
