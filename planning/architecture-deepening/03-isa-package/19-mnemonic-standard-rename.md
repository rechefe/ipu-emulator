# Mnemonic standard: rename script, schema rule and SV rename table

Part of [Epic E03: ISA package](00-epic.md) · ID E03-19 · Size M · Depends on: E03-04, E02-01, E02-02, E02-03, E02-04 · Behaviour change: yes, names only: old mnemonics stop assembling; every binary is byte-identical · `*_WIDTH` localparams: **unchanged**; SV enum member names and union member names change (names-only breaking change for the downstream RTL repo, Q46)

## Goal

Apply the mnemonic standard (Q33, Q45): Control-stage instructions use plain RISC-style names with variants as dot suffixes; datapath instructions use `UNIT.OP[.VARIANT]`; no mnemonic contains an underscore; branch signedness is in the name. Rename with a script over every source, enforce the standard in the schema, and publish the SV rename table for the downstream RTL repo.

## Background

Mapping decided in Q45 (opcodes unchanged, so binaries are unchanged):

| Old | New | Slot |
|---|---|---|
| `LDR_MULT_REG` | `LD.MULT` | LOAD |
| `LDR_CYCLIC_MULT_REG` | `LD.MULT.CYCLIC` | LOAD |
| `LDR_MULT_MASK_REG` | `LD.MASK` | LOAD |
| `STR_POST_AAQ_REG` | `ST.AAQ` | STORE |
| `STR_ACC_REG` | `ST.ACC` | ACC_STORE |
| `INCR_MOD_POW2` | `INC.MODP2` | LR |
| `BLT`, `BGE` | `BLT.S`, `BGE.S` | COND |
| `BGT`, `BLE` (pseudo) | `BGT.S`, `BLE.S` | pseudo |
| `BKPT` | `HALT` | COND (hardware semantics and RTL docs: E03-20) |

`.U` variants (`BLT.U`, `BGE.U`, `BGT.U`, `BLE.U`) are reserved (Q33). All other mnemonics are unchanged.

Occurrences at HEAD 694a8a8, whole-word: 101 `.asm` files (kernels use some lower-case forms, for example the convolution family), 52 `.py` files (spec, emulator string compares such as `ipu_emu/alias_observer.py:69,330-346`, `alias_detectors.py:10`, `isa_alias_spec.py:187`, tests, inline test programs), 17 `.md` files (docs, SKILL.md, requirements such as `faults.md:32`), 3 `.js` files (`vscode-ipu-asm/test/`), 1 `.tex` file. Handlers follow the old names (`execute_ldr_mult_reg`, `ipu_emu/ipu.py:1052`; `execute_bkpt`, `:1875`).

The SV package derives enum members as `<SLOT>_INST_OPCODE_<MNEMONIC with . -> _>` and union members as the lower-case form (`gen_codegen.py:112-133`).

## Implementation

1. `tools/rename_mnemonics.py` (kept in the repo for downstream users and open branches): whole-word replacement over `.asm`, `.py`, `.md`, `.j2`, `.js`, `.tex`, excluding generated files and `planning/`.
   - In `.asm` and inline assembly strings, match case-insensitively and keep the original case style (`ldr_mult_reg` becomes `ld.mult`); upper-casing kernels is E07's job.
   - In prose and code, match upper case only.
   - Do not match a name already followed by `.` (so `BLT.S` is never renamed twice; the script is idempotent).
   - Rename handlers to follow the mnemonic (`execute_ld_mult`, `execute_halt`, ...) and update `execute_fn`.
   - Print a per-file count; the PR includes the output.
2. Spec: rename the dict keys and docs; the `BLT.S`/`BGE.S` operation text says "signed 32-bit" explicitly.
3. Schema rules (E03-03 list): mnemonics match `[A-Z][A-Z0-9]*(\.[A-Z0-9]+)*`; no underscore; every non-NOP instruction in a datapath slot (LOAD, STORE, ACC_STORE, MULT, ACC, AAQ) contains a dot; reserved names (`BLT.U`, `BGE.U`, `BGT.U`, `BLE.U`) are listed in `ipu_isa.RESERVED_MNEMONICS` and may not be defined. The assembler's unknown-mnemonic error checks that list and says "reserved, not implemented" (a few lines in `ipu_as`).
4. SV rename table, published in the PR description, the release log and a short note in `docs/content/specs/` next to the instruction format page:

   | Old SV name | New SV name |
   |---|---|
   | `LOAD_INST_OPCODE_LDR_MULT_REG` / member `ldr_mult_reg` | `LOAD_INST_OPCODE_LD_MULT` / `ld_mult` |
   | `LOAD_INST_OPCODE_LDR_CYCLIC_MULT_REG` / `ldr_cyclic_mult_reg` | `LOAD_INST_OPCODE_LD_MULT_CYCLIC` / `ld_mult_cyclic` |
   | `LOAD_INST_OPCODE_LDR_MULT_MASK_REG` / `ldr_mult_mask_reg` | `LOAD_INST_OPCODE_LD_MASK` / `ld_mask` |
   | `STORE_INST_OPCODE_STR_POST_AAQ_REG` / `str_post_aaq_reg` | `STORE_INST_OPCODE_ST_AAQ` / `st_aaq` |
   | `ACC_STORE_INST_OPCODE_STR_ACC_REG` / `str_acc_reg` | `ACC_STORE_INST_OPCODE_ST_ACC` / `st_acc` (removed from the SV package by E03-22) |
   | `LR_INST_OPCODE_INCR_MOD_POW2` / `incr_mod_pow2` | `LR_INST_OPCODE_INC_MODP2` / `inc_modp2` |
   | `COND_INST_OPCODE_BLT` / `blt` | `COND_INST_OPCODE_BLT_S` / `blt_s` |
   | `COND_INST_OPCODE_BGE` / `bge` | `COND_INST_OPCODE_BGE_S` / `bge_s` |
   | `COND_INST_OPCODE_BKPT` / `bkpt` | `COND_INST_OPCODE_HALT` / `halt` |

   Enum values, typedef names, operand field names and every `*_WIDTH` are unchanged. A test diffs the old and new SV golden and asserts that the only changed identifiers are the ones in this table.

## Tests

- [ ] E02-01 asm digests unchanged for every program (the main proof).
- [ ] E02-02 field names unchanged; disassembly snapshots regenerated, and a test asserts the only token differences are the mapped mnemonics.
- [ ] E02-03 SV golden regenerated; the localparam dict is not edited; the identifier-diff test above passes.
- [ ] E02-04 generated artifacts regenerated (docs, `isa.json`, grammar).
- [ ] Each schema rule fails on a planted violation (`LD_MULT`, a datapath mnemonic without a dot, defining `BLT.U`).
- [ ] `rename_mnemonics.py` run twice changes nothing the second time.

## Acceptance Criteria

- [ ] No old mnemonic remains in sources, tests or docs outside `planning/` and the release log.
- [ ] The schema enforces the standard.
- [ ] The SV rename table is published for the downstream RTL repo.

## Risks

- Hard-coded mnemonic text in `ipu_as` messages and generators (`immediate.py:61`, `gen_docs.py`, `gen_vscode.py`) is rewritten by the script here; E05-16 later replaces it with lookups from the ISA view so the next rename needs no assembler edit.
- Open branches and downstream users with old mnemonics. The script stays in `tools/` and the release log links it.
- E09-01's docs asm-block test fails if the script misses a doc. That is the intended guard.

## Open question

- Whether old mnemonics stay accepted as deprecated aliases for a transition period is not decided (same question as E05-16). If wanted, they are one-to-one entries in `PSEUDO_INSTRUCTION_SPEC` (ASSEMBLER tag), so no assembler code changes. Until decided, this issue does not add them.
