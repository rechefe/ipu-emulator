# Tests assert the STR_ACC_REG debug-only warning instead of ignoring it

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-11 · Size S · Depends on: none · Behaviour change: none (emulator output); the warning gets its own class, which is still a `UserWarning`

## Goal

Every test that executes `STR_ACC_REG` says so and checks the warning. Every other test fails if the warning appears. A kernel that starts using the emulator-only store by accident is caught.

## Background

Verified at 694a8a8:

- `Ipu.execute_str_acc_reg` (ipu.py:1033-1050) calls `warnings.warn("[DEBUG ONLY] STR_ACC_REG is not a hardware instruction ...", stacklevel=2)` on every execution, as a plain `UserWarning`.
- Kernels that use it: `linear/fully_connected`, `elementwise/residual_add`, `elementwise/residual_add_256x144`, `normalize/layernorm_128x16`, `normalize/layernorm_256x144`, `matmul/matmul_144x144_x128`, `reshape/unfold_16x16x192`, plus the mobilevit seam tests `test_seam_unfold_matmul_xmem_direct_l4.py` and `_l5.py`.
- Only two tests check the warning: `test_execute.py:2953` and `test_alias_profile.py:293` (`pytest.warns(UserWarning ...)`). In every kernel test it is emitted and ignored (the review counted about 20 occurrences in a normal run).
- `fully_connected/test.py:27` already expects the alias hit `A16_STR_ACC_REG`, so the instruction's use there is intended.

## Decisions applied

- Q17, Q39 context: `STR_ACC_REG` lives in the emulator-only `acc_store` slot (D6: `acc_store` is not real hardware). E03 tags it EMULATOR and Q45 renames it `ST.ACC`. This issue only makes its use visible in tests; it does not change the instruction.

## Implementation

1. `ipu_emu/errors.py`: add `class DebugOnlyInstructionWarning(UserWarning)` and export it from `ipu_emu`. `execute_str_acc_reg` warns with that category. The message stays the same.
2. `kernel_registry/testing.py` `case_tests`: decide once per kernel whether its program uses the instruction, by assembling it (already done at testing.py:44-45) and checking whether any instruction word has a non-NOP `acc_store` slot. Then:
   - if it does, wrap `run_case` in `pytest.warns(DebugOnlyInstructionWarning)`;
   - if it does not, run it inside `warnings.catch_warnings()` with `simplefilter("error", DebugOnlyInstructionWarning)`.
   Checking the assembled words, not the source text, keeps working after the Q45 rename.
3. Kernel-specific tests that call `run_case` or `app.run` directly for the kernels above (for example `fully_connected/test.py:21-27`) and the two mobilevit seam tests use `pytest.warns(DebugOnlyInstructionWarning)`.
4. `ipu-emu-py` tests that execute the instruction (`test_alias_tracking.py`, `test_debug_tui.py`, `test_stats.py`, plus the two that already check it) use `pytest.warns(DebugOnlyInstructionWarning)` or `@pytest.mark.filterwarnings("error::ipu_emu.errors.DebugOnlyInstructionWarning")` with an explicit `pytest.warns` around the run.

## Tests

- [ ] A test of `case_tests` with a toy kernel: one that uses `STR_ACC_REG` passes only if the warning fires; one that does not use it fails if the warning fires (inject it with a monkeypatched handler).
- [ ] `bazel test //...` passes, and running the emulator and app tests with `--test_arg=-W --test_arg=error::UserWarning` shows no `DebugOnlyInstructionWarning` escaping any test.

## Acceptance Criteria

- [ ] The warning is never emitted outside a `pytest.warns` block in the test suite.
- [ ] A new kernel that uses `STR_ACC_REG` without saying so fails its generated test.

## Risks

- E06 replaces `case_tests` with generated per-kernel tests. The rule "emulator-only instructions are declared and asserted" should carry over; E05's `--target hardware` gives the same signal from the ISA tags.
