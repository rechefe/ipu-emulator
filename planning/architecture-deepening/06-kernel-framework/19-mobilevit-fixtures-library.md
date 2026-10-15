# MobileViT fixtures as a `py_library` (no `sys.path` hacks)

Part of [Epic: Kernel framework](00-epic.md) · ID E06-19 · Size S · Depends on: none · Behaviour change: none

## Goal

Import the MobileViT test fixtures through a declared Bazel library instead of
`sys.path.insert(Path(__file__).parent)`. This is independent of the rest of
E06, can land first, and gives the E07 chain port a clean starting point.

## Background

- 17 files in `src/tools/ipu-apps/test/mobilevit/` insert their own directory
  into `sys.path` to import `fixture_*.py`.
- The fixtures (`fixture_full_layer_chain.py`, `fixture_host_softmax_fold_stubs.py`,
  `fixture_kernel_support.py`, `fixture_packed_l4_measure.py`,
  `fixture_packed_l5_measure.py`; 853 lines) are declared only as **data**
  (`ipu_app.bzl:185-187`), not as sources. Bazel does not know they are Python
  dependencies.

## Implementation

- In `src/tools/ipu-apps/BUILD.bazel`, add
  `py_library(name = "mobilevit_fixtures", srcs = glob(["test/mobilevit/fixture_*.py"]), imports = ["test/mobilevit"], deps = [":ipu_apps_lib"])`.
- In `ipu_multi_kernel_tests` (`ipu_app.bzl:167-192`), depend on
  `:mobilevit_fixtures` and drop `suite/*.py` from `data`.
- Delete the 17 `sys.path.insert` lines and their `Path`/`sys` imports where
  they become unused.
- When E07 moves the chain tests next to the promoted kernels, the library moves
  with them (E07-23).

## Tests

- [ ] Every mobilevit test target builds, and the non-manual ones pass.
- [ ] Run the manual `test_full_*` targets by name once before and once after, and record the results in the PR (they take hours).
- [ ] A grep test: no `sys.path.insert` under `src/tools/ipu-apps/test/`.

## Acceptance Criteria

- [ ] The fixtures are a declared library, and there are no `sys.path` manipulations in mobilevit tests.
- [ ] `bazel test //...` is green.
