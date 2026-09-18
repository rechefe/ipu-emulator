"""One executable test target per registered kernel."""
load("@rules_python//python:defs.bzl", "py_test")

_PYTEST_SHIM = "@rules_python_pytest//python_pytest:pytest_shim.py"


def ipu_app(name, kernel_package, deps, data = [], test_deps = [], asm = None):
    """Declare a kernel label usable by both bazel run and bazel test.

    bazel run selects the exact SPEC.name through the registry frontend;
    bazel test executes the adjacent test.py through the existing pytest shim.
    test_<name> remains a compatibility alias. asm is relative to kernel_package
    and defaults to name + ".asm". Pass pytest dependencies in test_deps.
    """
    kernel_data = data + [kernel_package + "/" + (asm if asm != None else name + ".asm")]
    test_file = kernel_package + "/test.py"
    py_test(
        name = name,
        srcs = [
            "src/ipu_apps/kernel_registry/bazel_entry.py",
            test_file,
            _PYTEST_SHIM,
        ],
        main = "src/ipu_apps/kernel_registry/bazel_entry.py",
        args = [name, "$(location :" + test_file + ")", "$(location " + _PYTEST_SHIM + ")"],
        data = kernel_data,
        deps = deps + test_deps,
        legacy_create_init = False,
    )
    native.alias(
        name = "test_" + name,
        actual = ":" + name,
    )
