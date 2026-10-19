# Per-family Bazel libraries, `FAMILY_DEPS` and the family import-graph test

Part of [Epic: Kernel framework](00-epic.md) · ID E06-18 · Size M · Depends on: E06-13 · Behaviour change: none

## Goal

Make the build graph match ownership, so that editing one kernel reruns only
its family's tests. A family is one Bazel library. Cross-family use is declared
once, in `FAMILY_DEPS`. An import-graph test fails on any undeclared edge (Q62).

## Background

- `src/tools/ipu-apps/BUILD.bazel:13-22` globs every `.py` under
  `src/ipu_apps/**` into one `ipu_apps_lib`, and every kernel test depends on
  it. Measured (cluster plan 06):
  `bazel query --noimplicit_deps --keep_going 'kind("test rule", rdeps(//src/tools/ipu-apps/..., //src/tools/ipu-apps:src/ipu_apps/kernels/reshape/identity/cases.py))' | wc -l`
  returns 206, which is every ipu-apps test.
- Every `.asm` in a family, or in the whole tree, is data of many targets:
  `BUILD.bazel:36,47,103,117`, `ipu_app.bzl:103-107,155-158,185-188`.
- `case_tests(__package__)` runs full-tree discovery
  (`kernel_registry/cases.py:69-76` `package_kernel` -> `kernels()`), so even a
  split graph would need every `app.py` at runtime.
- Today's cross-family source edges are `attention -> softmax`
  (`attention/test.py`) and the mobilevit suites, which use attention,
  elementwise, matmul, normalize, projections, reshape and softmax.

## Implementation

- `kernels/<family>/BUILD.bazel` calls `ipu_kernel_family()`, which declares:
  - `:lib`: a `py_library` over the family's shared Python
    (`ipu_kernels/<family>/**`, `pack.py`, `stages.py`, `gen.py`,
    `reference.py`, `layers.py`), with deps `ipu_harness_lib` plus `:lib` of
    each family in `FAMILY_DEPS[family]`;
  - `:files`: the family's `.asm`, `.asm.j2`, `kernel.toml`, `registry.toml`
    and `.npy` files, plus `:files` of the declared dependency families;
  - `:all`: a `test_suite` over the family's kernel targets.
- `kernels/BUILD.bazel` holds `FAMILY_DEPS = {"attention": ["softmax"]}` and
  `:all_kernels` (every `:files` and `:lib`) for the query, manifest and
  registry-wide tests.
- `ipu_kernel()` (E06-13) depends on its family's `:lib` and `:files` only.
  The runtime registry is loaded scoped to that family and its `FAMILY_DEPS`
  (E06-16 `families=`).
- **Import-graph test** (pure Python, `ast` over the sources): every import
  from or file reference into `ipu_kernels.<other>` or `kernels/<other>/` must
  be listed in `FAMILY_DEPS`. Stage `.asm` references (E06-14) and `pack` names
  that resolve to another family count as edges.

## Tests

- [ ] The import-graph test fails on a planted `from ipu_kernels.softmax import ...` in a family without that edge, and on a planted stage reference.
- [ ] After migration, the rdeps query for one kernel's `kernel.toml` lists that kernel's targets, its family suite, registry-wide tests and the dependent families only. Record before and after counts in the PR. The target from cluster plan 06 is 45 or fewer for `reshape/identity`.
- [ ] Editing one `kernel.toml` and rerunning `bazel test //...` re-executes only the dependents above. Report the uncached count.

## Acceptance Criteria

- [ ] No test under `//kernels/...` depends on a library that globs the whole kernel tree, except `:all_kernels` users.
- [ ] `FAMILY_DEPS` is the only place cross-family use is declared, and a test enforces it.
- [ ] `bazel test //...` is green.
