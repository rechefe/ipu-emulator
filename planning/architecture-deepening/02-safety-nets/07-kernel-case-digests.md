# Per-case kernel digests: binary, CR/LR after setup, output, cycles

Part of [Epic E02: Safety nets before the refactors](00-epic.md) · ID E02-07 · Size M · Depends on: E02-01 · Behaviour change: none

## Goal

Make every kernel test also prove "bit-identical to the base commit", not only "within tolerance of the reference". For each case, record what ran and what it produced: the program, the CR and LR values the harness programmed, the XMEM image after setup, the output bytes and the cycle count. E04 (emulator core), E06 (harness with automatic placement) and E07 (kernel migration) then prove their refactors by leaving these files unchanged, or show exactly which case changed in a behaviour PR.

## Background

- Kernel tests run every case through `run_case` (`ipu_apps/kernel_registry/cases.py:101-159`), called from `case_tests` (`kernel_registry/testing.py:21-48`). `run_case` builds the harness and calls `app.run`, which calls `ipu_emu.emulator.run_test`: load program, `setup(state)`, run, `teardown(state)` (`ipu_emu/emulator.py:309-322`).
- Checks compare with tolerances, for example `check_close` with `rtol=1e-4, atol=1e-5` (`kernel_registry/case_support.py:46-47`). A refactor that moves a result by one ulp passes.
- Most cases export output through an `output_path` binding (`case_support.prepared_image`, `case_support.py:33-44`); `run_case` can copy it (`cases.py:146-151`). Some families check state directly.
- Render-per-run kernels swap `self.inst_path` inside `run` (`kernels/convolutions/conv_universal_wide384/app.py:330-338`), so the file `run_case` assembled is not always the program that ran. `state.inst_mem` after loading is.
- Harnesses program CRs by hand: about 500 `set_cr` calls in 56 files (plan 05). E06 replaces them with an ABI and automatic placement; Q81 keeps placement deterministic so migrated kernels can reproduce today's layout.
- Q70: kernel tests stay in kernel folders.

## Implementation

1. Add `ipu_apps/kernel_registry/baseline.py` with:
   ```python
   @dataclass(frozen=True)
   class CaseRecord:
       program_sha256: str          # canonical encoding of state.inst_mem after load
       cr: tuple[int, ...]          # CR0..CR15 after setup, as integers (readable in review)
       lr: tuple[int, ...]          # LR0..LR15 after setup
       xmem_setup_sha256: str       # digest of non-zero 64 KiB XMEM pages after setup, with page indices
       output_sha256: str | None    # bytes of the output_path file after teardown, if the case has one
       state_sha256: str            # all registers + non-zero XMEM pages after the run
       cycles: int

   def record_case(...) -> CaseRecord
   ```
   Hash only non-zero pages (numpy `reshape(-1, 65536).any(axis=1)`), so the cost per case is one pass over XMEM, not a hash of 512 MiB.
2. In `run_case`, wrap `app.setup` on the instance to capture `cr`, `lr`, `xmem_setup_sha256` and `program_sha256` right after the harness's own `setup`, and capture the rest after the run. No harness changes.
3. Store records in `<kernel folder>/baseline.json`, keyed by the test id `case_tests` already builds (`<case>` or `<options label>`). One file per kernel, next to `test.py` (Q70).
4. `case_tests` compares each run with its record after the case's own check passes. A mismatch fails with the case id and the list of fields that differ (for `cr`/`lr`, the register names and both values).
5. Add `bazel run //src/tools/ipu-apps:update_kernel_baselines -- [<kernel> ...]`. It runs the named kernels' cases (all kernels if none named) in-process and rewrites their `baseline.json`. A missing record fails the test with a message naming this command.
6. Only the default element format and mode each case already runs is recorded. Cases that run wide-vector debug mode record that run.

## Tests

- [ ] All kernel tests pass with their new `baseline.json` on the base commit.
- [ ] Planted change: altering one `set_cr` value in one harness (scratch branch) fails that kernel's test, naming the CR. Altering one input value fails `xmem_setup_sha256`.
- [ ] The digest overhead per case is under 0.2 s; total kernel-test time grows by less than 5 % (measured with E01 benchmarks).

## Acceptance Criteria

- [ ] Every registered kernel folder has a `baseline.json` covering every case and sweep entry its `test.py` runs.
- [ ] A refactor that changes any CR, LR, program word, setup image, output byte or cycle count of any case fails `bazel test //...`.
- [ ] Records are human-readable for CR and LR values.

## Risks

- E06 deletes `MemoryApp` and rewrites harnesses. Records keyed by kernel and case name survive that if E06 keeps the names; if a case is renamed, the E06 PR maps old to new ids explicitly, not by regenerating.
- `xmem_setup_sha256` is placement-sensitive by design. If E06 moves a region on purpose, that is a behaviour PR for the affected kernels, and `output_sha256` still proves the result is unchanged.
- Kernels whose case check depends on random data without a fixed seed would make records flaky. Fixing the seed is in scope here; list any such case in the PR.
