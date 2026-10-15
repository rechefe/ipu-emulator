# Libraries depend on runtime requirements only, checked against the lock

Part of [Epic E01: Foundation](00-epic.md) · ID E01-03 · Size M · Depends on: E01-02 · Behaviour change: none

## Goal

Each `py_library` carries exactly the third-party packages its `pyproject.toml` lists under `[project].dependencies`. Test and docs packages reach only test and docs targets. A test fails when a declared dependency is missing from the lock or from the library, so "added a dependency, forgot to relock" cannot reach master.

## Background

Verified at 694a8a8:

- `tools/pyproject_deps.bzl:7-59` line-parses a `pyproject.toml`. It collects `dependencies = [` and also the `docs = [` and `dev = [` optional groups (pyproject_deps.bzl:20-23), and generates `get_requirements()`.
- `ipu_as_lib` (`src/tools/ipu-as-py/BUILD.bazel:14-27`) uses `get_requirements(requirement)`, so the assembler library depends on `pytest`, `pyslang`, `mkdocs`, `mkdocs-material` and `mkdocs-git-revision-date-localized-plugin`. `ipu_emu_lib` and `ipu_apps_lib` pull in `pytest` the same way. Every test of every package rebuilds its runfiles when a docs package changes.
- The parser only understands one requirement per line split on `>=`, `==` and `<` (pyproject_deps.bzl:41). `pkg[extra]`, `~=`, `!=`, environment markers and inline arrays break it silently.
- Test targets already add their own test-only packages by hand, for example `requirement("pyslang")` for `test_sv_package_compiles` and `requirement("pytest")` everywhere. `//docs:mkdocs` lists the three MkDocs packages by hand (`docs/BUILD.bazel`). That is the pattern to keep.
- Runtime dependencies are short: `ipu-as-py` has `click`, `lark`, `jinja2`; `ipu-emu-py` has `numpy`, `ml-dtypes`; `ipu-apps` has `numpy`; `ipu-common` has none.

## Decisions applied

- Q62: one responsibility per package, one-way dependencies enforced by Bazel deps. Third-party deps follow the same rule: a library declares what it imports.

## Implementation

1. Replace `get_requirements(requirement)` in the four `py_library` targets with explicit lists, for example:
   ```python
   py_library(
       name = "ipu_as_lib",
       ...
       deps = [
           "//src/tools/ipu-common:ipu_common_lib",
           requirement("click"),
           requirement("jinja2"),
           requirement("lark"),
       ],
   )
   ```
2. Give each test target that needed a dev package through the library its own `requirement(...)` (mostly `pytest`, already present on most).
3. Delete `tools/pyproject_deps.bzl` and the three `parse_pyproject_deps` repositories in `MODULE.bazel:21, 33-55`.
4. Add `//tools:deps_consistency_test` (`py_pytest_test`). It reads, with `tomllib` (`tomli` on Python 3.10, already in the lock):
   - every `pyproject.toml` and `tools/requirements-tools.in`;
   - `requirements_lock.txt`;
   - a small generated manifest of each library's third-party deps. Produce it with a `genrule` over `bazel query`-free data: each package `BUILD` exports its runtime list as a Starlark constant (`RUNTIME_DEPS = ["click", "jinja2", "lark"]`) that both the `py_library` and a `write_file` target use, so the list exists once.
   It asserts:
   - every requirement name in any `pyproject.toml` (runtime and extras) and in `requirements-tools.in` is pinned in the lock (normalised names, PEP 503);
   - each library's runtime list equals its package's `[project].dependencies` minus the workspace packages (`ipu-*`).
5. Fix `pyproject.toml` content the test exposes (for example `ml-dtypes` floor from E00-01).

## Tests

- [ ] `bazel query 'deps(//src/tools/ipu-as-py:ipu_as_lib)' | grep -E 'pytest|mkdocs|pyslang'` returns nothing; the same for `ipu_emu_lib`, `ipu_apps_lib`, `ipu_common_lib`.
- [ ] `deps_consistency_test` fails when a scratch commit adds `"requests>=2"` to a `pyproject.toml` without relocking, and when it adds a runtime dep without adding it to `RUNTIME_DEPS`.
- [ ] `bazel test //...` pass and fail counts are unchanged.

## Acceptance Criteria

- [ ] No library target depends on a test or docs package.
- [ ] `tools/pyproject_deps.bzl` is gone; each runtime dependency list is written once per package.
- [ ] `deps_consistency_test` runs in the default tier.

## Risks

- A library module that imports a package only its tests listed (for example a debug helper importing `pytest`) fails at import once the transitive dep disappears. The full test run catches it; fix by moving the import or declaring the dependency.
