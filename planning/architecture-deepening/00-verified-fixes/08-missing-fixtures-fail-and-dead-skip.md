# Kernel tests: a missing fixture fails, and the dead `needs_softmax` skip goes

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-08 · Size S · Depends on: none · Behaviour change: none (emulator); tests now fail instead of skipping when a checked-in fixture is missing

## Goal

A kernel test either runs or fails. No test passes the suite by skipping because a file that is part of the repository is missing, and no skip marker guards a condition that cannot happen.

## Background

Verified at 694a8a8 (`src/tools/ipu-apps/src/ipu_apps/kernels/`):

- `linear/fully_connected/test.py:12-18`:
  ```python
  def _run(name):
      try:
          state, cycles = run_case("fully_connected", CASES[name])
      except MissingInputFixture as exc:
          pytest.skip(str(exc))
  ```
  `MissingInputFixture` (`fully_connected/cases.py:9, 22`) fires when a checked-in `.bin` under `test_data_format/` is not in the runfiles. A packaging mistake turns every `test_mac_accounting_and_alias_hits` case into a skip, and `bazel test` still reports PASSED. `test_missing_input_fixture_is_skipped` (test.py:29-33) pins this behaviour.
- `attention/test.py:721-725`:
  ```python
  _SOFTMAX_ASM = [files(f"ipu_apps.kernels.softmax.{k}").joinpath(f"{k}.asm")
                  for k in ("softmax_rows_partial", "softmax_columns_packed")]
  needs_softmax = pytest.mark.skipif(not all(p.is_file() for p in _SOFTMAX_ASM), ...)
  ```
  used at test.py:733 and 758. `attention/test.py` runs as `:attention_test`, a family test, and `ipu_families_from_kernels` gives every family test every kernel's `.asm` as data (`ipu_app.bzl:150-158`: `native.glob([kernels_root + "/**/*.asm", ...])`). So the condition is always false and the marker is dead code. If the data wiring ever broke, the two tests would skip silently rather than fail.

## Decisions applied

- Q78: `fully_connected` goldens will come from a reference per format and the `.bin` files will be deleted (E07). Until then, the fixtures that exist must be required.

## Implementation

1. `fully_connected/test.py`: remove the `try/except` in `_run`, so `MissingInputFixture` (a `FileNotFoundError`) fails the test with its path in the message.
2. Rename `test_missing_input_fixture_is_skipped` to `test_missing_input_fixture_fails` and assert `pytest.raises(MissingInputFixture, match="missing fully connected fixture")`.
3. `attention/test.py`: delete `needs_softmax` and its two uses. Keep `_SOFTMAX_ASM` only if a test still reads it; otherwise delete it too.
4. Grep `src/tools/ipu-apps` for other `pytest.skip` / `skipif` whose condition is "a repository file is missing" and treat them the same way. At HEAD the remaining skips are environment opt-ins (`IPU_RUN_SLOW_TESTS` in `projections/test.py:110-113`) and optional-dependency checks (PyTorch), which stay.

## Tests

- [ ] `bazel test //src/tools/ipu-apps:fully_connected //src/tools/ipu-apps:attention_test` passes with the same number of executed tests and fewer or equal skips.
- [ ] `test_missing_input_fixture_fails` passes.
- [ ] Manually remove one `.bin` from the `fully_connected` target's data in a scratch commit: the target fails (record in the PR).

## Acceptance Criteria

- [ ] `grep -n "pytest.skip\|skipif" src/tools/ipu-apps/src/ipu_apps/kernels/linear/fully_connected/test.py src/tools/ipu-apps/src/ipu_apps/kernels/attention/test.py` shows no fixture- or runfiles-based skip.
- [ ] No production code changes.
