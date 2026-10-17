"""ipu_kernel: the Bazel macro that declares one kernel (E06-13).

    load("//kernels:defs.bzl", "ipu_kernel")

    ipu_kernel()

in kernels/<family>/<name>/BUILD.bazel, a folder that holds <name>.asm,
kernel.toml and the files they name (reference.py, data/, test_*.py). It
declares:

* :<name>: `bazel test` runs every default-tier case in every format it runs in
  (`test_case[<case>-<format>]`) and the folder's test_*.py; `bazel run` runs
  one case through the CLI (`-- --case NAME --format FMT ...`).
* :<name>_nightly: the cases with `tier = "nightly"` (Q73): size enormous,
  so the default tier skips it and `--config=full` (the nightly run) runs it.
* :<name>_files: the folder's files, which //kernels:tree joins.
"""

load("@ipu_as_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_test")
load("//tools:lint.bzl", "ruff_test")

_ENTRY = "//src/tools/ipu-harness:src/ipu_harness/kernel_test.py"
_PYTEST_SHIM = "@rules_python_pytest//python_pytest:pytest_shim.py"

def ipu_kernel(name = None, extra_deps = [], size = "large", shard_count = None):
    """Declare kernel `name` from its folder.

    Args:
        name: the kernel's name; default the folder's (the last path component).
        extra_deps: libraries the folder's test_*.py import.
        size: the test size of :<name> (kernels run in the emulator).
        shard_count: shards of :<name> and :<name>_nightly; set it for a kernel
            with more than about 60 case x format runs, so a shard keeps inside
            the CI budget (Starlark cannot read kernel.toml to count them).
    """
    name = name or native.package_name().rpartition("/")[2]
    files = native.glob(
        ["**"],
        exclude = ["BUILD.bazel", "BUILD", "**/__pycache__/**", "**/*.pyc"],
    )
    native.filegroup(
        name = name + "_files",
        srcs = files,
        visibility = ["//visibility:public"],
    )
    deps = [
        "//src/tools/ipu-harness:ipu_harness_lib",
        "//src/tools/ipu-harness:kernel_test_lib",
        "//src/tools/ipu-kernels:ipu_kernels_lib",
        requirement("pytest"),
        requirement("torch"),
    ] + extra_deps
    # The folder's Python (reference.py, test_*.py), in //kernels:lint.
    ruff_test(name = "ruff", srcs = native.glob(["**/*.py"]))
    for target, tier, target_size, tags in [
        (name, "default", size, []),
        (name + "_nightly", "nightly", "enormous", ["nightly"]),
    ]:
        py_test(
            name = target,
            srcs = [_ENTRY, _PYTEST_SHIM],
            main = _ENTRY,
            args = ["$(rootpath kernel.toml)", "$(location " + _PYTEST_SHIM + ")", tier],
            data = files,
            deps = deps,
            legacy_create_init = False,
            size = target_size,
            shard_count = shard_count,
            tags = tags,
        )

def _label(kernel, target):
    return "//kernels/" + kernel + ":" + target

def ipu_kernel_tree(kernels):
    """Declare //kernels' :tree (every kernel's files) and :kernel_targets (its targets, JSON).

    Bazel cannot glob into subpackages, so `kernels` names each kernel folder
    ("<family>/<name>"); //src/tools/ipu-harness:test_kernel_tree checks it
    names every folder that holds a kernel.toml.

    Args:
        kernels: every kernel folder under kernels/, as "<family>/<name>".
    """
    native.filegroup(
        name = "tree",
        srcs = ["BUILD.bazel"] + [_label(k, k.rpartition("/")[2] + "_files") for k in kernels],
        visibility = ["//visibility:public"],
    )
    targets = {}
    families = {}
    for kernel in kernels:
        name = kernel.rpartition("/")[2]
        if name in targets:
            fail("ipu_kernel_tree: two kernels are named %s (%s and %s)" % (name, targets[name]["dir"], "kernels/" + kernel))
        label = _label(kernel, name)
        families.setdefault(kernel.rpartition("/")[0], []).append(label)
        targets[name] = {
            "dir": "kernels/" + kernel,
            "family": kernel.rpartition("/")[0],
            "run": label,
            "test": label,
            "nightly": label + "_nightly",
        }
    # :lint: every kernel's ruff_test, listed in //tools:lint.
    native.test_suite(name = "lint", tests = [_label(k, "ruff") for k in kernels], tags = ["lint"])
    # :<family>: every kernel of the family (the manifest's family suite).
    for family, labels in families.items():
        native.test_suite(name = family, tests = labels, visibility = ["//visibility:public"])
    content = json.encode_indent(
        {
            "kernels": targets,
            "families": {family: {"suite": "//kernels:" + family} for family in families},
            "query": "//kernels:query",
        },
        indent = "  ",
    )
    native.genrule(
        name = "kernel_targets",
        outs = ["kernel_targets.json"],
        visibility = ["//visibility:public"],
        cmd = "cat > $@ <<'IPU_KERNEL_TARGETS'\n" + content + "\nIPU_KERNEL_TARGETS\n",
    )
