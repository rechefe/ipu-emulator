"""mypy_test: type-check a py_library with the locked mypy (E01-11, Q65).

    load("//tools:typecheck.bzl", "mypy_test")

    mypy_test(name = "mypy", lib = ":ipu_isa_lib")                 # default mode
    mypy_test(name = "mypy", lib = ":ipu_isa_lib", strict = True)  # --strict

The test checks the Python sources `lib` declares in its own Bazel package.
Repository packages it imports resolve through MYPYPATH (the first-party import
roots of `lib` and its deps) and are checked only as far as they are used;
third-party packages resolve from the same runfiles as the library itself.
Shared settings live in the root mypy.ini; strictness is per target.
"""

load("@ipu_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "PyInfo", "py_test")

def _mypy_manifest_impl(ctx):
    lib = ctx.attr.lib
    info = lib[PyInfo]
    label = lib.label
    sources = [
        f
        for f in info.transitive_sources.to_list()
        if f.is_source and
           f.owner.workspace_name == label.workspace_name and
           f.owner.package == label.package and
           f.extension in ("py", "pyi")
    ]
    if not sources:
        fail("mypy_test: %s has no Python sources in its own package" % label)

    # PyInfo.imports holds runfiles-root-relative paths (`_main/src/...`). Keep
    # the first-party ones, made workspace-relative like the sources; the
    # third-party ones are already on the test's sys.path.
    prefix = ctx.workspace_name + "/"
    roots = sorted([
        imp[len(prefix):]
        for imp in info.imports.to_list()
        if imp.startswith(prefix)
    ])
    out = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(out, json.encode({
        "roots": roots,
        "sources": sorted([f.short_path for f in sources]),
    }))
    return [DefaultInfo(
        files = depset([out]),
        runfiles = ctx.runfiles(files = [out] + sources),
    )]

_mypy_manifest = rule(
    implementation = _mypy_manifest_impl,
    attrs = {
        "lib": attr.label(mandatory = True, providers = [PyInfo]),
    },
    doc = "JSON manifest of a py_library's own sources and first-party import roots.",
)

def mypy_test(name, lib, strict = False, tags = [], data = [], **kwargs):
    """mypy over `lib`'s sources with //:mypy.ini, from the locked wheel.

    Args:
        name: target name; `mypy` by convention.
        lib: the py_library to check (its sources in its own package).
        strict: pass `--strict` (Q65: required for ipu_isa and the reference
            model API; other packages start in default mode).
        tags: extra tags; `typecheck` is always added, so
            `bazel test --test_tag_filters=typecheck //...` selects them all.
        data: extra runfiles, e.g. a library `lib` imports but whose sources
            are not checked.
        **kwargs: passed to py_test (size, timeout, ...).
    """
    manifest = name + "_manifest"
    _mypy_manifest(
        name = manifest,
        lib = lib,
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )
    kwargs.setdefault("size", "small")
    py_test(
        name = name,
        srcs = [Label("//tools:mypy_runner.py")],
        main = Label("//tools:mypy_runner.py"),
        args = [
            "--config-file",
            "$(rootpath %s)" % Label("//:mypy.ini"),
            "--manifest",
            "$(rootpath :%s)" % manifest,
        ] + (["--strict"] if strict else []),
        # A caller's data joins ours: modules imported but not checked (a
        # strict library whose package is not yet strict-clean, E04-30).
        data = [
            ":" + manifest,
            Label("//:mypy.ini"),
        ] + data,
        deps = [
            lib,
            requirement("mypy"),
        ],
        tags = tags + ["typecheck"],
        **kwargs
    )
