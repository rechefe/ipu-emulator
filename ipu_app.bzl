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
    dependencies in test_deps. Kernel tests are ``large``: a kernel runs its
    cases in the emulator, which takes minutes for the bigger ones.
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
        size = "large",
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

def _kernel_asms(kernels_root):
    """Every kernel's main .asm: the one named after its folder.

    A multi-stage kernel keeps its other stages beside it (e.g.
    ``decimate_stage2.asm``); those are data of the kernel, not kernels.
    """
    return [
        asm
        for asm in native.glob([kernels_root + "/**/*.asm"])
        if _kernel_name(asm) == _kernel_package(asm).rpartition("/")[2]
    ]

def ipu_apps_from_kernels(kernels_root, deps, test_deps = []):
    """Declare one ipu_app per kernel .asm file found under kernels_root.

    Discovers kernels by glob instead of a maintained list: dropping a new
    kernel package under kernels_root registers its run/test label without
    editing the calling BUILD file. Each kernel has its own package; any .bin
    fixtures alongside the kernel (e.g. fully_connected's test_data_format/)
    are pulled in as data automatically, as are its family siblings' .asm
    files: a multi-stage kernel may run a sibling kernel's program as a stage.
    """
    for asm in _kernel_asms(kernels_root):
        kernel_package = _kernel_package(asm)
        family = _kernel_package(kernel_package)
        ipu_app(
            name = _kernel_name(asm),
            kernel_package = kernel_package,
            deps = deps,
            test_deps = test_deps,
            data = native.glob(
                [family + "/**/*.asm", kernel_package + "/**/*.bin"],
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
    `:benchmark_maxpool2d_stride2`, with every .asm in its family as data.
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
            # Family siblings too: a multi-stage kernel runs a sibling's program.
            data = native.glob([_kernel_package(kernel_package) + "/**/*.asm"]),
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
    the family's kernels. That test gets every kernel's .asm (a family test may
    run another family's kernel), the family's .bin fixtures, and `data`.
    """
    families = {}
    for asm in _kernel_asms(kernels_root):
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
                    [kernels_root + "/**/*.asm", family_dir + "/**/*.bin"],
                    allow_empty = True,
                ) + data,
                imports = ["src"],
                legacy_create_init = False,
                deps = deps + test_deps,
                size = "large",
            )
            tests = tests + [":" + family + "_test"]
        native.test_suite(name = family, tests = sorted(tests))

def ipu_multi_kernel_tests(suites_root, kernels_root, deps):
    """Declare one test per ``<suites_root>/<suite>/test_*.py``, named after the file.

    For tests that run several kernels together (one kernel's output feeding
    the next). Every kernel's ``.asm`` under kernels_root is data, as are the
    suite's other ``.py`` files (shared fixtures) and ``.asm`` programs. Files named
    ``test_full_*.py`` run whole model stacks and take hours, so they are
    ``manual``: run them by name, passing pytest options with ``--test_arg``
    and switches with ``--test_env``.
    """
    for test in native.glob([suites_root + "/*/test_*.py"]):
        suite, _, filename = test.rpartition("/")
        full = filename.startswith("test_full_")
        py_pytest_test(
            name = filename[:-len(".py")],
            size = "enormous" if full else "large",
            tags = ["manual"] if full else [],
            srcs = [test],
            data = native.glob(
                [suite + "/*.py", suite + "/*.asm"],
                exclude = [suite + "/test_*.py"],
            ) + native.glob([kernels_root + "/**/*.asm"]),
            imports = ["src"],
            legacy_create_init = False,
            deps = deps,
        )

def ipu_per_kernel_test(name, src, kernels_root, deps, data = [], size = "large"):
    """Run the test file ``src`` once per kernel, each as its own target.

    ``:<name>_<kernel>`` runs ``src`` with ``IPU_KERNEL=<kernel>``; the test
    selects that kernel's cases. Each target gets its own timeout and Bazel
    runs them concurrently. ``:<name>`` is the suite of all of them.
    """
    tests = []
    for asm in _kernel_asms(kernels_root):
        kernel = _kernel_name(asm)
        py_pytest_test(
            name = name + "_" + kernel,
            size = size,
            srcs = [src],
            env = {"IPU_KERNEL": kernel},
            data = data,
            legacy_create_init = False,
            deps = deps,
        )
        tests.append(":" + name + "_" + kernel)
    native.test_suite(name = name, tests = tests)
