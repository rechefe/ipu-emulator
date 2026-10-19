# Test targets: split `test_debug_launch`, keep kernel test files out of `ipu_apps_lib`

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-09 · Size S · Depends on: none · Behaviour change: none (same tests, different targets)

## Goal

Each Bazel test target has the size its content needs, and editing one kernel's `test.py` reruns only the tests that use it. This is groundwork for the test tiers in E01-04, where size `enormous` means "heavy tier".

## Background

Verified at 694a8a8 (`src/tools/ipu-apps/`):

- `test_debug_launch` (`BUILD.bazel:99-105`) is one `py_pytest_test` with `size = "enormous"`. It collects 104 tests:
  - 15 tests of debugger launch behaviour on a 3-instruction program and on the `identity` kernel. Measured: 9.7 s together (`--test_arg=-k --test_arg="not every_kernel"`).
  - `test_every_kernel_enters_debugger_and_completes` (test_debug_launch.py:296-311), parametrised over every registered kernel (89). It runs each kernel's default case to completion under a stubbed debugger. Measured: the run passed 400 s at about item 80 and timed out under `--test_timeout=400`; in a full `bazel test //...` it was the last target still running after 470 s.
  So the `enormous` size is right for the sweep and wrong for the 15 fast tests, which today cannot run in a default tier that excludes `enormous` (Q73).
- `ipu_apps_lib` (`BUILD.bazel:14-23`) has `srcs = glob(["src/ipu_apps/**/*.py"])`. That glob includes every kernel's `test.py` and `test_*.py`: 97 test modules. Each kernel target (`ipu_app.bzl:24-32`) lists its `test.py` in `srcs` again and depends on `ipu_apps_lib`. So every kernel test module is a library source of every `ipu-apps` target, and editing one kernel's `test.py` changes the inputs of all of them: the whole `ipu-apps` suite reruns.
- No library module imports a kernel test module (`grep -rn "kernels\.[a-z_.]*\.test\b"` under `src/tools` finds nothing).

## Implementation

1. Split `test_debug_launch`:
   - Move `test_every_kernel_enters_debugger_and_completes` into `test/test_debug_launch_kernels.py`. Select the kernel from `IPU_KERNEL` the way `test_alias_profile_kernels.py:15-18` does.
   - Declare it with the existing macro, which gives one target per kernel (parallel, individually cached):
     ```python
     ipu_per_kernel_test(
         name = "test_debug_launch_kernels",
         src = "test/test_debug_launch_kernels.py",
         kernels_root = "src/ipu_apps/kernels",
         data = glob(["src/ipu_apps/**/*.asm", "src/ipu_apps/**/*.bin"]),
         deps = [":ipu_apps_lib", requirement("pytest")],
         size = "enormous",
     )
     ```
   - `test_debug_launch` keeps the 15 fast tests and becomes `size = "small"` (raise to `medium` if CI shows it near the 60 s small timeout).
2. Exclude test modules from the library:
   ```python
   srcs = glob(
       ["src/ipu_apps/**/*.py"],
       exclude = ["src/ipu_apps/**/test.py", "src/ipu_apps/**/test_*.py"],
   ),
   ```
   Family tests (`ipu_app.bzl:150-162`) already list their `test.py` in `srcs`. If a kernel `test.py` imports a sibling test helper module, add that module to the target's `srcs` explicitly.

## Tests

- [ ] `bazel test //src/tools/ipu-apps:test_debug_launch` runs 15 tests and passes in under 30 s.
- [ ] `bazel test --test_size_filters= //src/tools/ipu-apps:test_debug_launch_kernels` passes; the number of targets equals the number of registered kernels.
- [ ] `bazel query 'deps(//src/tools/ipu-apps:ipu_apps_lib)' | grep -E '/test(_[a-z0-9_]+)?\.py$'` returns nothing.
- [ ] Edit one kernel's `test.py` (whitespace) and run `bazel test //src/tools/ipu-apps/...`: only that kernel's target and its family suite rerun; the others report `(cached)`. Record the counts in the PR.
- [ ] `bazel test //...` pass and fail counts are unchanged (the split moves tests between targets; the total test-case count stays 104 for the debug-launch tests).

## Acceptance Criteria

- [ ] The fast debug-launch tests are in a `small` target; the per-kernel sweep is `enormous`, one target per kernel.
- [ ] No test module is a source of `ipu_apps_lib`.

## Risks

- E06 splits `ipu_apps_lib` into per-family libraries. This exclusion is a subset of that work and does not conflict with it.
