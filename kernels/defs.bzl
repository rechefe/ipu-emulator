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
* :assemble_<name>: <name>.bin, the default case's program in its first
  format (ipu_harness.kernel_bin), which E07 compares with the E02 asm digest.
* :<name>_files: the folder's files, which //kernels:tree joins.

A family's variants are entries of kernels/<family>/registry.toml over a
template folder, an ordinary ipu_kernel() folder (E06-17):

    load("//kernels:defs.bzl", "ipu_kernel_family")
    load(":registry.bzl", "ENTRIES")

    ipu_kernel_family(entries = ENTRIES)

declares :<entry>, :<entry>_nightly and :assemble_<entry> for each entry, as
ipu_kernel() does for a folder, and :registry_bzl_test, which fails when
registry.bzl (written by `bazel run //kernels:update_registry_bzl`) is stale.

A family folder's own test_*.py (cross-kernel conformance, datapath and
routing tests) run as :tests.
"""

load("@ipu_as_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python_pytest//python_pytest:defs.bzl", "py_pytest_test")
load("//tools:lint.bzl", "ruff_test")
load(":families.bzl", "FAMILY_DEPS", "KERNELS")

_ENTRY = "//src/ipu-harness:src/ipu_harness/kernel_test.py"
_PYTEST_SHIM = "@rules_python_pytest//python_pytest:pytest_shim.py"
_REGISTRY_BZL = "//src/ipu-harness:src/ipu_harness/registry_bzl.py"
_KERNEL_BIN = "//src/ipu-harness:kernel_bin"

def _kernel_targets(name, kernel, data, deps, size, shard_count):
    """:<name> (the default tier), :<name>_nightly and :assemble_<name>.

    Args:
        name: the kernel's name.
        kernel: the kernel, by its files' $(rootpath): kernel.toml, or
            <registry.toml>::<entry> (kernel_test.py's first argument).
        data: the files the kernel reads.
        deps: the tests' libraries.
        size: the test size of :<name>.
        shard_count: shards of :<name> and :<name>_nightly.
    """
    native.genrule(
        name = "assemble_" + name,
        srcs = data,
        outs = [name + ".bin"],
        tools = [_KERNEL_BIN],
        cmd = "$(execpath %s) %s $@" % (_KERNEL_BIN, kernel.replace("$(rootpath ", "$(execpath ")),
        visibility = ["//visibility:public"],
    )
    for target, tier, target_size, tags in [
        (name, "default", size, []),
        (name + "_nightly", "nightly", "enormous", ["nightly"]),
    ]:
        py_test(
            name = target,
            srcs = [_ENTRY, _PYTEST_SHIM],
            main = _ENTRY,
            args = [kernel, "$(location " + _PYTEST_SHIM + ")", tier],
            data = data,
            deps = deps,
            legacy_create_init = False,
            size = target_size,
            shard_count = shard_count,
            tags = tags,
        )

def _test_deps(family_package):
    return [
        "//" + family_package + ":lib",
        "//src/ipu-harness:kernel_test_lib",
        requirement("pytest"),
        requirement("torch"),
    ]

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
    # The family is the parent package. Its library and files only (E06-18):
    # editing another family's kernel reruns none of these tests. `uses` adds a
    # same-family kernel's files (E06-14 stages).
    family = native.package_name().rpartition("/")[0]
    data = files + ["//" + family + ":files"] + [
        "//" + family + "/" + used + ":" + used + "_files"
        for used in uses
    ]
    # The folder's Python (reference.py, test_*.py), in //kernels:lint; a
    # template whose reference is a shared op may have none (E07-02).
    ruff_test(name = "ruff", srcs = native.glob(["**/*.py"], allow_empty = True))
    _kernel_targets(name, "$(rootpath kernel.toml)", data, _test_deps(family) + extra_deps, size, shard_count)

def ipu_kernel_family(
        entries = [],
        kernels_lib = None,
        extra_deps = [],
        size = "large",
        shard_count = {},
        tests_tree = False,
        tests_size = "medium",
        tests_shard_count = None,
        tests_nightly_shard_count = None,
        tests_data = []):
    """Declare one family, //kernels/<family> (E06-18), and its registry entries (E06-17).

    * :lib: the family's Python, ipu_kernels.<family> (packs, stages, layer
      adapters: //src/ipu-kernels:ipu_kernels_<family>), on the harness,
      the shared ipu_kernels code and the :lib of each family FAMILY_DEPS lists.
    * :files: its shared files (*.asm, *.asm.j2 macros, registry.toml, *.npy,
      and any *.py the harness loads by path), with the :files of each family
      it uses.
    * :all: every kernel test of the family (the manifest's family suite),
      its entries' and :tests included.
    * :tests: the family folder's own test_*.py (conformance, datapath and
      routing tests across its kernels), when it has any: pytest, with the
      family's :lib and :files and every kernel's and template's files, so
      `Registry.load(kernel_tree(), families=[<family>])` sees the family.
      With tests_tree, also the whole kernel tree (//kernels:all_kernels), for
      a test that routes across families or calls ipu_harness.query.main;
      then an edit to any kernel reruns it. tests_data adds files it reads
      (a docs page whose tables it checks).
    * :tests_nightly: the family folder's test_full_*.py (whole model layers,
      minutes each), as :tests but size enormous and tagged nightly, so the
      default tier skips them and `--config=full` (the nightly run) runs them.
    * :ruff: lint over its Python.
    * :<entry>, :<entry>_nightly, :assemble_<entry> for each registry entry,
      and :registry_bzl_test when it has entries.

    Args:
        entries: ENTRIES of the family's registry.bzl ({"name", "template"} each).
        kernels_lib: the family's ipu_kernels library; default
            //src/ipu-kernels:ipu_kernels_<family>. A fixture family
            outside kernels/ names the shared one.
        extra_deps: libraries the template folders' Python imports, for the
            entries' tests (as ipu_kernel()'s), and the family's :tests.
        size: the test size of each :<entry>, as ipu_kernel()'s.
        shard_count: shards of :<entry> and :<entry>_nightly, by entry name,
            for an entry with more than about 60 case x format runs.
        tests_tree: give :tests the whole kernel tree, not only the family's.
        tests_size: the test size of :tests.
        tests_shard_count: shards of :tests (pytest_shard_plugin splits the collected tests).
        tests_nightly_shard_count: shards of :tests_nightly.
        tests_data: more data of :tests.
    """
    package = native.package_name()
    family = package.rpartition("/")[2]
    uses = FAMILY_DEPS.get(family, [])
    py_library(
        name = "lib",
        deps = [
            "//src/ipu-harness:ipu_harness_lib",
            kernels_lib or "//src/ipu-kernels:ipu_kernels_" + family,
        ] + ["//kernels/" + other + ":lib" for other in uses],
        visibility = ["//visibility:public"],
    )
    # The family's own tests are linted, not data: editing one reruns only :tests.
    # test_full_*.py run whole model layers and take minutes each: the nightly
    # tier's :tests_nightly.
    full_tests = native.glob(["test_full_*.py"], allow_empty = True)
    family_tests = native.glob(["test_*.py"], exclude = full_tests, allow_empty = True)
    py = native.glob(["*.py"], exclude = family_tests + full_tests, allow_empty = True)
    native.filegroup(
        name = "files",
        srcs = native.glob(["*.asm", "*.asm.j2", "registry.toml", "*.npy"], allow_empty = True) + py +
               ["//kernels/" + other + ":files" for other in uses],
        visibility = ["//visibility:public"],
    )
    # A template is also one of the family's KERNELS: list each folder once.
    folders = {k: None for k in KERNELS.get(family, []) + [e["template"] for e in entries]}
    for target, srcs, test_size, shards, tags in [
        ("tests", family_tests, tests_size, tests_shard_count, []),
        ("tests_nightly", full_tests, "enormous", tests_nightly_shard_count, ["nightly"]),
    ]:
        if not srcs:
            continue
        py_pytest_test(
            name = target,
            srcs = srcs,
            data = [":files"] + ["//%s/%s:%s_files" % (package, k, k) for k in sorted(folders.keys())] + tests_data,
            deps = _test_deps(package) + (["//kernels:all_kernels"] if tests_tree else []) + extra_deps +
                   (["//tools:pytest_shard_plugin"] if shards else []),
            args = ["-p", "pytest_shard_plugin"] if shards else [],
            size = test_size,
            shard_count = shards,
            tags = tags,
            legacy_create_init = False,
        )
    native.test_suite(
        name = "all",
        tests = ["//%s/%s:%s" % (package, kernel, kernel) for kernel in KERNELS.get(family, [])] +
                [":" + entry["name"] for entry in entries] + ([":tests"] if family_tests else []),
        visibility = ["//visibility:public"],
    )
    ruff_test(name = "ruff", srcs = py + family_tests + full_tests)
    if not entries:
        return
    templates = sorted({entry["template"]: None for entry in entries}.keys())
    py_test(
        name = "registry_bzl_test",
        srcs = [_REGISTRY_BZL],
        main = _REGISTRY_BZL,
        args = ["check", "$(rootpath registry.toml)", "$(rootpath registry.bzl)"],
        data = ["registry.toml", "registry.bzl"] + ["//%s/%s:%s_files" % (package, t, t) for t in templates],
        deps = ["//src/ipu-harness:ipu_harness_lib"],
        legacy_create_init = False,
        size = "small",
    )
    # Reading registry.toml checks every entry's template folder, so each
    # entry's data holds every template's files (E07-02), not only its own.
    template_files = ["//%s/%s:%s_files" % (package, t, t) for t in templates]
    for entry in entries:
        name = entry["name"]
        _kernel_targets(
            name,
            "$(rootpath registry.toml)::" + name,
            ["registry.toml", ":files"] + template_files,
            _test_deps(package) + extra_deps,
            size,
            shard_count.get(name),
        )

def _label(kernel, target):
    return "//kernels/" + kernel + ":" + target

def _add(targets, name, target):
    if name in targets:
        fail("ipu_kernel_tree: two kernels are named %s (%s and %s)" % (name, targets[name]["dir"], target["dir"]))
    targets[name] = target

def ipu_kernel_tree(registries = {}):
    """Declare //kernels' :tree (every family's and kernel's files), :all_kernels
    (the tree, with every family library), :kernel_targets (JSON) and :lint,
    from KERNELS (kernels/families.bzl) and the families' registry entries.

    Args:
        registries: ENTRIES of each family's registry.bzl, by family name
            (E06-17); //src/ipu-harness:test_kernel_tree checks it names
            every registry.toml and its entries.
    """
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
        label = _label(kernel, name)
        families.setdefault(kernel.rpartition("/")[0], []).append(label)
        _add(targets, name, {
            "dir": "kernels/" + kernel,
            "family": kernel.rpartition("/")[0],
            "run": label,
            "test": label,
            "nightly": label + "_nightly",
        })
    for family, entries in registries.items():
        for entry in entries:
            label = "//kernels/%s:%s" % (family, entry["name"])
            families.setdefault(family, []).append(label)
            _add(targets, entry["name"], {
                "dir": "kernels/%s/%s" % (family, entry["template"]),
                "family": family,
                "registry": "kernels/%s/registry.toml" % family,
                "template": entry["template"],
                "run": label,
                "test": label,
                "nightly": label + "_nightly",
            })
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
