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
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//tools:lint.bzl", "ruff_test")
load(":families.bzl", "FAMILY_DEPS", "KERNELS")

_ENTRY = "//src/tools/ipu-harness:src/ipu_harness/kernel_test.py"
_PYTEST_SHIM = "@rules_python_pytest//python_pytest:pytest_shim.py"

def ipu_kernel(name = None, extra_deps = [], size = "large", shard_count = None, uses = []):
    """Declare kernel `name` from its folder.

    Args:
        name: the kernel's name; default the folder's (the last path component).
        extra_deps: libraries the folder's test_*.py import.
        size: the test size of :<name> (kernels run in the emulator).
        shard_count: shards of :<name> and :<name>_nightly; set it for a kernel
            with more than about 60 case x format runs, so a shard keeps inside
            the CI budget (Starlark cannot read kernel.toml to count them).
        uses: kernels of the same family whose programs this one's stages
            hook runs (E06-14: `ctx.run_stage("../<kernel>/<any>.asm")`), by
            folder name; their files become this kernel's data.
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
    family = native.package_name().split("/")[1]
    # Its family's library and files only (E06-18): editing another family's
    # kernel reruns none of these tests. `uses` adds a same-family kernel's
    # files (E06-14 stages).
    data = files + ["//kernels/" + family + ":files"] + [
        "//kernels/" + family + "/" + used + ":" + used + "_files"
        for used in uses
    ]
    deps = [
        "//kernels/" + family + ":lib",
        "//src/tools/ipu-harness:kernel_test_lib",
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
            data = data,
            deps = deps,
            legacy_create_init = False,
            size = target_size,
            shard_count = shard_count,
            tags = tags,
        )

def ipu_kernel_family():
    """Declare one family, //kernels/<family> (E06-18).

    * :lib: the family's Python, ipu_kernels.<family> (packs, stages, layer
      adapters: //src/tools/ipu-kernels:ipu_kernels_<family>), on the harness,
      the shared ipu_kernels code and the :lib of each family FAMILY_DEPS lists.
    * :files: its shared files (*.asm, *.asm.j2 macros, registry.toml, *.npy,
      and any *.py the harness loads by path), with the :files of each family
      it uses.
    * :all: every kernel test of the family (the manifest's family suite).
    * :ruff: lint over its Python.
    """
    family = native.package_name().rpartition("/")[2]
    uses = FAMILY_DEPS.get(family, [])
    py_library(
        name = "lib",
        deps = [
            "//src/tools/ipu-harness:ipu_harness_lib",
            "//src/tools/ipu-kernels:ipu_kernels_" + family,
        ] + ["//kernels/" + other + ":lib" for other in uses],
        visibility = ["//visibility:public"],
    )
    py = native.glob(["*.py"], allow_empty = True)
    native.filegroup(
        name = "files",
        srcs = native.glob(["*.asm", "*.asm.j2", "registry.toml", "*.npy"], allow_empty = True) + py +
               ["//kernels/" + other + ":files" for other in uses],
        visibility = ["//visibility:public"],
    )
    native.test_suite(
        name = "all",
        tests = ["//kernels/%s/%s:%s" % (family, kernel, kernel) for kernel in KERNELS.get(family, [])],
        visibility = ["//visibility:public"],
    )
    ruff_test(name = "ruff", srcs = py)

def _label(kernel, target):
    return "//kernels/" + kernel + ":" + target

def ipu_kernel_tree():
    """Declare //kernels' :tree (every family's and kernel's files), :all_kernels
    (the tree, with every family library), :kernel_targets (JSON) and :lint,
    from KERNELS (kernels/families.bzl)."""
    kernels = [family + "/" + name for family in KERNELS for name in KERNELS[family]]
    native.filegroup(
        name = "tree",
        srcs = ["BUILD.bazel"] + [_label(k, k.rpartition("/")[2] + "_files") for k in kernels] +
               ["//kernels/" + family + ":files" for family in KERNELS],
        visibility = ["//visibility:public"],
    )
    # For the registry-wide users only (the query, the manifest, the tree tests).
    py_library(
        name = "all_kernels",
        data = [":tree"],
        deps = ["//kernels/" + family + ":lib" for family in KERNELS],
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
    # :lint: every kernel's and family's ruff_test and //kernels' own, listed in //tools:lint.
    native.test_suite(
        name = "lint",
        tests = [":ruff"] + [_label(k, "ruff") for k in kernels] +
                ["//kernels/" + family + ":ruff" for family in KERNELS],
        tags = ["lint"],
    )
    content = json.encode_indent(
        {
            "kernels": targets,
            "families": {family: {"suite": "//kernels/" + family + ":all"} for family in families},
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
