"""One executable test target per registered kernel."""
load("@rules_python//python:defs.bzl", "py_binary", "py_test")
load("@rules_python_pytest//python_pytest:defs.bzl", "py_pytest_test")
load("//:asm_rules.bzl", "assemble_asm")

_PYTEST_SHIM = "@rules_python_pytest//python_pytest:pytest_shim.py"
_BENCHMARK_RUNNER = "src/ipu_apps/kernel_registry/benchmarking.py"


def ipu_app(name, kernel_package, deps, data = [], test_deps = []):
    """Declare a kernel label usable by both bazel run and bazel test.

    bazel run selects the exact SPEC.name through the registry frontend;
    bazel test runs the package's required test.py, which runs every case in
    cases.py (``test_case = case_tests(__package__)``) plus any kernel-specific
    tests. test_<name> remains a compatibility alias, and assemble_<name> builds
    the standalone assembled binary from kernel_package/<name>.asm. Pass pytest
    dependencies in test_deps.
    """
    asm_path = kernel_package + "/" + name + ".asm"
    kernel_data = data + [asm_path]
    test_file = kernel_package + "/test.py"
    py_test(
        name = name,
        srcs = ["src/ipu_apps/kernel_registry/bazel_entry.py", test_file, _PYTEST_SHIM],
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
    assemble_asm(
        name = "assemble_" + name,
        src = asm_path,
    )

def _kernel_name(asm_path):
    return asm_path.rpartition("/")[2][:-len(".asm")]

def _kernel_package(asm_path):
    return asm_path.rpartition("/")[0]

def ipu_apps_from_kernels(kernels_root, deps, test_deps = []):
    """Declare one ipu_app per kernel .asm file found under kernels_root.

    Discovers kernels by glob instead of a maintained list: dropping a new
    kernel package under kernels_root registers its run/test label without
    editing the calling BUILD file. Each kernel has its own package; any .bin
    fixtures alongside the kernel (e.g. fully_connected's test_data_format/)
    are pulled in as data automatically.
    """
    for asm in native.glob([kernels_root + "/**/*.asm"]):
        kernel_package = _kernel_package(asm)
        ipu_app(
            name = _kernel_name(asm),
            kernel_package = kernel_package,
            deps = deps,
            test_deps = test_deps,
            data = native.glob(
                [kernel_package + "/**/*.asm", kernel_package + "/**/*.bin"],
                exclude = [asm],
                allow_empty = True,
            ),
        )

def ipu_benchmarks_from_kernels(kernels_root, deps):
    """Declare `:benchmark_<package>` for every kernel package with a benchmark.py.

    benchmark.py only declares configs (CONFIGS); the shared runner in
    kernel_registry/benchmarking.py runs them through the package's cases and
    writes results.md beside it. E.g.
    `kernels/pooling/maxpool2d_stride2/benchmark.py` becomes
    `:benchmark_maxpool2d_stride2`, with every .asm in that directory as data.
    """
    for benchmark in native.glob([kernels_root + "/**/benchmark.py"]):
        kernel_package = _kernel_package(benchmark)
        py_binary(
            name = "benchmark_" + kernel_package.rpartition("/")[2],
            srcs = [_BENCHMARK_RUNNER, benchmark],
            main = _BENCHMARK_RUNNER,
            # The module's dotted name (under the "src" import root) and the
            # package's workspace path, where results.md is written.
            args = [
                kernel_package.partition("/")[2].replace("/", ".") + ".benchmark",
                native.package_name() + "/" + kernel_package,
            ],
            data = native.glob([kernel_package + "/*.asm"]),
            imports = ["src"],
            legacy_create_init = False,
            deps = deps,
        )

def ipu_families_from_kernels(kernels_root, deps, test_deps = [], data = []):
    """Declare `:<family>` for every directory that groups kernel packages.

    A family is any directory between kernels_root and a kernel package, e.g.
    `kernels/softmax` or `kernels/convolutions`.
    `bazel test :<family>` runs every kernel target beneath it plus the
    family's own test.py, if it has one, as `:<family>_test` -- tests that span
    the family's kernels. That test gets every .asm/.bin under the family, plus
    `data`, as data.
    """
    families = {}
    for asm in native.glob([kernels_root + "/**/*.asm"]):
        parts = _kernel_package(asm)[len(kernels_root) + 1:].split("/")
        for depth in range(1, len(parts)):
            family_dir = kernels_root + "/" + "/".join(parts[:depth])
            families.setdefault(family_dir, []).append(":" + _kernel_name(asm))
    for family_dir, tests in families.items():
        family = family_dir.rpartition("/")[2]
        test_file = native.glob([family_dir + "/test.py"], allow_empty = True)
        if test_file:
            py_pytest_test(
                name = family + "_test",
                srcs = test_file,
                data = native.glob(
                    [family_dir + "/**/*.asm", family_dir + "/**/*.bin"],
                    allow_empty = True,
                ) + data,
                imports = ["src"],
                legacy_create_init = False,
                deps = deps + test_deps,
            )
            tests = tests + [":" + family + "_test"]
        native.test_suite(name = family, tests = sorted(tests))
