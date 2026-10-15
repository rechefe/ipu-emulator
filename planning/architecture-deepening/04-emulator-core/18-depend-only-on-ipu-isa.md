# `ipu_emu` depends only on `ipu_isa`

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-18 · Size S · Depends on: E03 (binary codec moved into `ipu_isa`), E08 (debuggers move to `ipu_debug`) · Behaviour change: none

## Goal

One-way package dependencies (Q62, Q69): `ipu_emu` imports `ipu_isa`, numpy and the standard library, nothing else. The assembler and the debuggers are not reachable from the emulator.

## Background

- `src/tools/ipu-emu-py/BUILD.bazel` gives `ipu_emu_lib` a dependency on `//src/tools/ipu-as-py:ipu_as_lib`.
- `execute.py:30` imports `ipu_as.compound_inst.CompoundInst` to decode binary words; `debug_pipeline.py:4` and `debug_cli.py:363` import it too.
- The debuggers (`debug_cli.py`, `debug_tui.py`, `debug_control.py`, `debug_launch.py`, `debug_pipeline.py`) live in `ipu_emu` today. E08 moves them to `ipu_debug`.
- `ipu_emu/__init__.py` imports `debug_cli` eagerly.

## Implementation

1. Replace `CompoundInst` decoding in `execute.py` with the binary codec E03 moves into `ipu_isa` (`decode_word`/`decode_program`).
2. After E08 has moved the debuggers, remove `ipu_as_lib` from `ipu_emu_lib` deps in `BUILD.bazel`. If E08 has not landed, do step 1 now and file the BUILD change as the last step of E08.
3. Curate `ipu_emu/__init__.py`'s exports: `IpuState`, `Ipu`, `RunStats`, `load_program`, `run_until_complete`, the error classes. No debugger import.
4. Add `ipu_emu` to the import-graph test (owned by E01/E03) with the allowed set `{ipu_isa}`.

## Tests

- [ ] Import-graph test: `ipu_emu` imports only `ipu_isa`, numpy and stdlib.
- [ ] `bazel build //src/tools/ipu-emu-py:ipu_emu_lib` succeeds without `ipu_as_lib`.
- [ ] Output-identical proof: E02 kernel output digests; `test_emulator_e2e.py`.

## Acceptance Criteria

- [ ] `grep -rn "ipu_as\b\|from ipu_as" src/tools/ipu-emu-py/src/` finds nothing.
- [ ] `import ipu_emu` does not import any debugger module.
