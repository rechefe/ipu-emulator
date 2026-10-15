# Emulator test support: `conftest.py` and one helper module

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-01 · Size S · Depends on: — · Behaviour change: none

## Goal

Give the emulator tests one place that assembles, loads and runs a program, so that later E04 refactors change one helper instead of six copies. Tests stop depending on run order and on private members.

## Background

- There is no `conftest.py` under `src/tools/ipu-emu-py/test/`.
- The assemble-load-run helper is written six times: `_make_state`/`_run` (test_execute.py:41, 65), `_run` (test_alias_tracking.py:29, test_stats.py:18), `_make_state_with_program` (test_debug_cli.py:59), `_run_both` (test_prototyping_equivalence.py:119) and `_run_wide` (test_wide_vector_debug.py:24).
- `reset_labels()` (from `ipu_as.label`) is called only in test_emulator_e2e.py:31-34, so the assembler's label table leaks between tests in other files.
- Tests reach private members: `ipu._take_snapshot()` (test_execute.py:3138; test_wide_vector_debug.py:643, 679, 696), `xmem._data` (test_prototyping_equivalence.py:115), `regfile._desc` (test_debug_tui.py:2943).
- The `[DEBUG ONLY]` `UserWarning` from `ST.ACC` (ipu.py:1043) fires in many tests without being asserted.
- `pytest.ini` holds only `testpaths = test`.

## Implementation

1. Add `test/emu_support.py` (a `py_library` named `emu_test_support` in `BUILD.bazel`) with:
   ```python
   def make_state(asm: str, *, cr: Mapping[int, int] | None = None,
                  fmt: ElementFormat | None = None, **state_kw) -> IpuState
   def run(asm: str, **kw) -> IpuState            # make_state + run to halt
   def run_both(asm: str, **kw) -> tuple[IpuState, IpuState]   # faithful, prototyping
   def decode_program(words: Iterable[int]) -> list[dict[str, int]]
   ```
   Until E04-17 lands, `fmt` maps to today's `dtype`/`wide_vector_*` keywords inside the helper. That mapping is the only place tests know about the old flags.
2. Add `test/conftest.py` with an autouse fixture that calls `reset_labels()` before and after each test. E05 removes the global label table; the fixture is deleted then.
3. Add `filterwarnings` to `pytest.ini`: `error::DeprecationWarning:ipu_emu`, and an explicit `ignore` entry for the `[DEBUG ONLY]` message. Convert tests that expect the warning to `pytest.warns`.
4. Add `conftest.py` and `emu_test_support` to every `py_pytest_test` in `src/tools/ipu-emu-py/BUILD.bazel` (check that `rules_python_pytest` picks up `conftest.py` from `srcs`).
5. Move each test file to the helpers in its own commit. Replace private access with public seams that exist today: `Ipu.execute_vliw_cycle()` instead of `_take_snapshot()` + handler call where possible. Leave the rest with a `# TODO(E04-04/E04-05)` marker; those issues add the public replacement (`RegFile.view`, `XMem.__eq__`).

## Tests

- [ ] Test count per target is unchanged: record `pytest --collect-only -q | tail -1` before and after in the PR.
- [ ] Run the emulator tests once in reverse file order (`-p no:randomly` plus `pytest-reverse`, or a manual order) and show they pass, which proves the label leak is gone.
- [ ] `bazel test //src/tools/ipu-emu-py/...` and `bazel test --define=ipu_proto=1 //src/tools/ipu-emu-py/...` are green.

## Acceptance Criteria

- [ ] One helper module; the six local copies are deleted.
- [ ] `reset_labels` runs around every emulator test.
- [ ] No unasserted `UserWarning` or `DeprecationWarning` from `ipu_emu` in test output.
- [ ] Every remaining private access in emulator tests carries a TODO naming the issue that removes it.

## Risks

- Bazel may not see `conftest.py` unless it is listed in `srcs`. Verify with a deliberately failing fixture once.
