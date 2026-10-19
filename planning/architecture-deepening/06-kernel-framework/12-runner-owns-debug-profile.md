# The runner owns the debug TUI launch and alias profiling

Part of [Epic: Kernel framework](00-epic.md) · ID E06-12 · Size S · Depends on: E06-11 · Behaviour change: none

## Goal

Make debugging and alias profiling options of a run, owned by the runner. No
kernel code and no harness base class reads the environment or writes profiler
state. Profiling attaches to the one run a case already makes, so every case
runs once (cluster plan 06, Q3/Q4 recommendations adopted in L11-L25).

## Background

- `IpuApp.run` reads `IPU_DEBUG_TUI` (`kernel_registry/base.py:91`). It re-enters
  the registry to find the kernel name (`base.py:95-100`) and writes the alias
  profile into `state` and `state.stats` (`base.py:103-105`).
  `bazel_entry.py:20` reads the same variable to hand the terminal over.
- Multi-stage kernels override `run` and call `run_test` directly for stage 2
  (`convolutions/conv_universal_wide384/app.py:329`,
  `convolutions/depthwise_conv_stride2_{narrow,128,16}/app.py`), so stage 2
  gets no debug callback.
- Every case runs 3 times: once in `case_tests` and twice (baseline and
  profiled) in `test/test_alias_profile_kernels.py:24-28`, through 89
  `test_alias_profile_kernels_<kernel>` targets (`BUILD.bazel:113-119`).

## Implementation

1. **Debug.**
   - `DebugLaunch` is built only by the CLI entry, from `--debug` or from
     `IPU_DEBUG_TUI=1`, which stays as the Bazel-friendly spelling.
   - The CLI entry module is the only reader of `IPU_DEBUG_TUI` in the repo.
     The terminal hand-over now in `bazel_entry.py:18-43` moves there too.
   - `run_kernel` passes `DebugLaunch.callback(kernel_name)` and
     `break_on_entry` to the E04 run call. The callback comes from E08's
     `ipu_debug` adapters; `ipu_harness` depends only on the callback protocol.
   - Stages (E06-14) receive the same callback, so stage 2 is debuggable.
2. **Profiling.**
   - `RunOptions.alias_profile = True` attaches an `AliasProfile` before the
     run through the E04 API (`attach_alias_profile`). It does not set fields
     on state.
   - The generated kernel test (E06-13) runs every case once **with** profiling
     on, then asserts the cheap accounting invariants on that single run:
     `profile.cycles == cycles`, plus the alias-sum and totals checks from
     `test/test_alias_profile_kernels.py`.
   - A baseline-versus-profiled equivalence run happens only for the `default`
     case of each kernel, in each format.
   - `--profile-aliases` and `--alias-report PATH` keep their meaning in the
     CLI.

## Tests

- [ ] `IPU_DEBUG_TUI=1` and `--debug` both produce a `DebugLaunch`. With neither, no curses import happens.
- [ ] A grep test finds `IPU_DEBUG_TUI` only in the CLI entry module.
- [ ] A two-stage kernel under `--debug` stops at entry of both stages (with a fake callback).
- [ ] The profiled single run gives the same cycles and outputs as an unprofiled run for `scale/default` (the equivalence check).

## Acceptance Criteria

- [ ] No module in `ipu_harness` other than the CLI entry reads the environment for debug or profiling.
- [ ] A kernel test runs each case once. The baseline comparison runs only on `default`.
- [ ] `bazel test //...` is green.
