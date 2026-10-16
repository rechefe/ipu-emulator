"""ruff_test: ruff lint and format checks over a package's Python files (Q65).

Every BUILD.bazel file declares one, named `ruff`, over `glob(["**/*.py"])`; a
package glob stops at subpackages, so together they cover each Python file in
the repository exactly once. tools/check_lint_coverage.sh enforces that, and
//tools:lint is the suite of all of them.
"""

load("@ipu_as_deps//:requirements.bzl", "data_requirement", "requirement")
load("@rules_python//python:defs.bzl", "py_test")

def ruff_test(name, srcs, tags = [], **kwargs):
    """`ruff check` and `ruff format --check` over srcs with //:ruff.toml.

    Ruff runs from the locked wheel (requirements_lock.txt), never from PATH.

    Args:
        name: target name; `ruff` by convention (the coverage check looks for it).
        srcs: the Python files to check.
        tags: extra tags; `lint` is always added, so once no ruff_test is
            manual `bazel test --test_tag_filters=lint //...` selects them all
            (`//...` skips manual targets; //tools:lint names them instead).
        **kwargs: passed to py_test (size, timeout, ...).
    """
    files = name + "_files"
    native.filegroup(
        name = files,
        srcs = srcs,
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )
    kwargs.setdefault("size", "small")
    py_test(
        name = name,
        srcs = [Label("//tools:ruff_runner.py")],
        main = Label("//tools:ruff_runner.py"),
        # $(rootpaths) rejects an empty filegroup; a package with no Python
        # file (today only the root) passes no file and the test is a no-op.
        args = [
            "--test",
            "$(rootpath %s)" % Label("//:ruff.toml"),
        ] + (["$(rootpaths :%s)" % files] if srcs else []),
        data = [
            ":" + files,
            Label("//:ruff.toml"),
            data_requirement("ruff"),
        ],
        deps = [requirement("ruff")],
        tags = tags + ["lint"],
        **kwargs
    )
