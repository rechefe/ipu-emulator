# `ipu_isa` imports nothing from the assembler or the emulator

Part of [Epic E03: ISA package](00-epic.md) · ID E03-02 · Size S · Depends on: E03-01, E01 (import-graph test) · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

`ipu_isa` must not import `ipu_as` or `ipu_emu` (Q31, Q62). Today it does, through two factories that build assembler classes and work only because every caller happens to have `ipu_as` on the path. Move those factories to the assembler, delete dead factories, and enforce the rule.

## Background

- `create_assembler_opcodes()` (`instruction_spec.py:1313-1367`) does `import ipu_as.ipu_token` inside the function (`:1326`) and builds `EnumToken` subclasses. Its only caller is `ipu_as/opcodes.py:13-16`.
- `create_assembler_reg_classes()` (`registers.py:304-357`) imports `ipu_as` the same way (`:318-333`). Its only caller is `ipu_as/reg.py:12,24`.
- `//src/tools/ipu-common:ipu_common_lib` has no Bazel dep on `ipu_as` (`ipu-common/BUILD.bazel:11-17`). These are cycle-dodging lazy imports, which Q62 forbids.
- `create_emulator_constants()` (`instruction_spec.py:1370-1411`) is called once (`ipu_emu/ipu.py:67`) and the result `_emu_constants` is never read.
- `ipu_as/reg.py:19-21` restates register counts (`IPU_LR_REG_NUM = 16` and so on); `:63` derives `IPU_LRD_REG_NUM` from it.

## Implementation

1. Move `create_assembler_opcodes` into `ipu_as/opcodes.py` and `create_assembler_reg_classes` into `ipu_as/reg.py`, unchanged except for the import. `ipu_isa` keeps the pure `create_assembler_reg_enums()` (names only, no classes).
2. Delete `create_emulator_constants` and `ipu.py:40,67`.
3. Replace the literals in `ipu_as/reg.py:19-21` with counts read from `REGISTER_DEFINITIONS`.
4. Register `ipu_isa` in E01's import-graph test: `ipu_isa` may import only the standard library and its declared third-party deps. Add `test_purity.py` in `ipu-isa/test/` that imports every `ipu_isa` module in a subprocess and asserts that no `ipu_as`, `ipu_emu`, `ipu_gen` or `ipu_apps` module is in `sys.modules`. Its Bazel deps are `ipu_isa_lib` only, so Bazel enforces the same rule.
5. Remove the `"create_assembler_opcodes"`, `"create_emulator_constants"` entries from `instruction_spec.__all__` and update the module docstring ("NO CODE GENERATION ... factory functions").

## Tests

- [ ] `test_purity.py` passes; it fails if a planted `import ipu_as` is added to any `ipu_isa` module.
- [ ] E02 decode snapshots (enum descriptors) and SV golden unchanged.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `grep -rn "ipu_as\|ipu_emu" src/tools/ipu-isa/src` finds nothing.
- [ ] No function-level import exists in `ipu_isa` to avoid an import cycle.
