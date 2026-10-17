# Assembler readiness for the new mnemonic standard

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-16 · Size S · Depends on: E05-02, E03 (mnemonic rename in `instruction_spec.py`) · Behaviour change: none (binaries unchanged)

## Goal

When E03 renames the mnemonics to the new standard (Q45), the assembler accepts them with no edit beyond `instruction_spec.py`. The rename covers `LDR_MULT_REG` → `LD.MULT`, `LDR_CYCLIC_MULT_REG` → `LD.MULT.CYCLIC`, `LDR_MULT_MASK_REG` → `LD.MASK`, `STR_POST_AAQ_REG` → `ST.AAQ`, `STR_ACC_REG` → `ST.ACC`, `INCR_MOD_POW2` → `INC.MODP2`, `BLT` / `BGE` / `BGT` / `BLE` → `.S`, and `BKPT` → `HALT`. This issue removes the assembler's hard-coded mnemonic knowledge and proves dotted multi-part mnemonics work end to end. The rename itself, the `.asm` rewrite script and the SV rename table are E03.

## Background

- Mnemonics are data-driven from `INSTRUCTION_SPEC` and lower-cased for lookup (`inst.py:182-187`). Dotted mnemonics already exist (`MULT.RC.VE`, `ACC.ADD.FIRST`), and the `TOKEN` regex allows dots (`asm_grammar.lark:23`).
- Hard-coded mnemonic text remains in `ipu_as`:
  - `immediate.py:61` names `INCR_MOD_POW2` in an error message;
  - `gen_docs.py:39, 83, 222` names `INCR_MOD_POW2`, `ADD`, `SUB`, `INC` and `DEC` in prose;
  - `gen_vscode.py:487` builds a `BKPT;;` snippet.

  The generators move to `ipu_gen` in E08. Fix them there or here, whichever lands first.
- `nop` is special-cased by name (`compound_inst.py:36`).

## Implementation

1. Replace each hard-coded mnemonic in `ipu_as` messages with a lookup from the ISA view (for example the opcode that uses the operand type), so the message follows a rename.
2. Keep `nop` as the one reserved mnemonic. Assert at import that no ISA instruction is named `NOP` in any slot except as the slot NOP.
3. Add a test that parametrizes over every mnemonic in the ISA view, assembles a minimal valid instance of each (operands from the operand-type metadata), and checks encode → decode → re-assemble. It runs before and after E03's rename with no edit.
4. Add a test that a 3-part dotted mnemonic (`LD.MULT.CYCLIC`) and a signed-branch suffix (`BLT.S`) lex and parse as one `TOKEN`. Use a synthetic ISA entry if the rename has not landed.

## Tests

- [ ] All-mnemonics round trip passes on the base ISA and on the renamed ISA.
- [ ] `grep -rn "INCR_MOD_POW2\|LDR_\|STR_\|BKPT" src/tools/ipu-as*/src` finds no hard-coded mnemonic text.
- [ ] E02 digests unchanged. `bazel test //...` green.

## Acceptance Criteria

- [ ] After the E03 rename, no file under `ipu_as` needs editing for the assembler to accept the new mnemonics.

## Open question

- Q45 says the `.asm` sources and docs are rewritten by script. It does not say whether the old mnemonics stay as deprecated aliases for a transition period, which matters to downstream users with their own `.asm` files. If aliases are wanted, they belong in `PSEUDO_INSTRUCTION_SPEC` (one-to-one expansions, ASSEMBLER tag) so no assembler code changes. Not decided.
