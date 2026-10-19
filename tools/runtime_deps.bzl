"""Runtime third-party dependencies of a workspace package, written once.

Each package keeps its list as `RUNTIME_DEPS` in a `deps.bzl` beside its
`BUILD.bazel`. The package's `py_library` maps it through `requirement()`, and
`runtime_deps_manifest` writes it to `runtime_deps.json` so
`//tools:test_deps_consistency` can check it against `pyproject.toml` and the
lock. Test and docs packages are not runtime dependencies: the test and docs
targets that need them list them themselves.
"""

def runtime_deps_manifest(name, runtime_deps):
    """Writes `runtime_deps.json` for this package.

    Args:
      name: target name.
      runtime_deps: the package's `RUNTIME_DEPS` (distribution names).
    """
    content = json.encode({
        "package": native.package_name(),
        "runtime_deps": runtime_deps,
    })
    native.genrule(
        name = name,
        outs = ["runtime_deps.json"],
        cmd = "echo '{}' > $@".format(content),
        visibility = ["//tools:__pkg__"],
    )
