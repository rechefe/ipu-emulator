# Strict schema check for the spec dicts

Part of [Epic E03: ISA package](00-epic.md) · ID E03-03 · Size S · Depends on: E03-01 · Behaviour change: none (the current spec passes) · `*_WIDTH` localparams: unchanged

## Goal

Keep authoring `INSTRUCTION_SPEC` and `PSEUDO_INSTRUCTION_SPEC` as nested dicts (Q30), but make the import-time check strict: unknown keys are rejected at every level, and the cross-field rules that today are only conventions become errors with a message that names the slot, instruction, operand and rule. A typo such as `"raed": "snapshot"` must fail at import instead of silently dropping the read flag.

## Background

`validate_instruction_spec()` (`instruction_spec.py:1619-1699`) checks that `operands`, `doc` and `execute_fn` exist, that operand types are known and that `read` is `snapshot` or `live`. It does not check:

- **Unknown keys.** Any extra key on an instruction, operand, `expands_to` or `SLOT_METADATA` entry is ignored.
- **Duplicate operand names** within one instruction. Operand names become SV field names (`<name>_<column>`, CLAUDE.md) and handler keyword arguments.
- **`read` legality per operand type.** `read` is meaningful only on register operands. Today `read` is set on `LrdIdx` (ACC.RESHAPE `source`/`dest`, `instruction_spec.py` acc slot) and `_resolve_operand` (`ipu_emu/ipu.py:812-833`) has no `LrdIdx` branch, so the flag is silently ignored and the handler reads the register pair itself (`ipu.py:1589-1590`).
- **Doc operand list vs operands.** `InstructionDoc.operands` entries start with the operand name; `INCR_MOD_POW2` documents `dst` while the operand is `dest`.
- **Canonical name uniqueness.** Mnemonics are matched case-insensitively (`canonical_instruction_name`, `instruction_spec.py:1210`), but nothing rejects two names that differ only in case.
- **Pseudo literal args.** `B`, `BZ`, `BNZ` expand with the literal `"CR0"` (`PSEUDO_INSTRUCTION_SPEC`). `validate_pseudo_instruction_spec` (`:1702-1786`) only checks that args are strings.

## Implementation

1. Add `ipu_isa/schema.py` holding the allowed keys as data, one frozen set per level:

   | Level | Required | Optional |
   |---|---|---|
   | instruction | `operands`, `doc`, `execute_fn` | (later issues add keys here: `tag` E03-07, `implicit_reads`/`implicit_writes` E03-14) |
   | operand | `name`, `type` | `read` |
   | `SLOT_METADATA` entry | `description` | `hardware` (replaced by `tag` in E03-07) |
   | pseudo | `operands`, `expands_to`, `doc` | — |
   | `expands_to` | `slot`, `instruction`, `args` | — |

   Adding a key to the spec means adding it here first; that is the point.
2. Rewrite `validate_instruction_spec()` and `validate_pseudo_instruction_spec()` as a list of small rule functions over the dicts, each raising `SpecError(IpuError, ValueError)` with a path such as `acc.ACC.RESHAPE.operands[source].read`. Rules:
   - unknown or missing keys at every level above;
   - operand names unique per instruction and match `[a-z][a-z0-9_]*`;
   - `read` allowed only when the operand type is readable: `LrIdx`, `CrIdx`, `LcrIdx`, `MultStageReg`, `LrdIdx` (an explicit set here; E03-05 moves it onto the operand-type registry);
   - `doc.operands` leading names (backticks stripped) equal the operand names, in order;
   - mnemonics unique per slot after `upper()`; exactly one `NOP` per slot;
   - pseudo literal args are valid register names for the target operand's type (`CR0` is valid for `LcrIdx`), checked against `create_assembler_reg_enums()`.
3. Fix the one data error the rules find: `INCR_MOD_POW2` doc operand `dst` to `dest`. (The syntax-line mismatches are removed by E03-04, which drops the `syntax` field.)
4. Keep the existing `__all__` names `validate_instruction_spec` and `validate_pseudo_instruction_spec`.

## Tests

- [ ] `ipu-isa/test/test_schema.py`: one negative test per rule. Each test deep-copies the spec, plants one violation (unknown key `raed`, duplicate operand name, `read` on `ActivationFn`, doc name mismatch, `Nop` and `NOP` in one slot, pseudo literal `CR16`), runs the validators on the copy and asserts the error path. The validators take the dicts as arguments so a copy can be checked.
- [ ] The real spec passes.
- [ ] E02 baselines unchanged except `instructions.md`/`isa.json` text for the `dst` fix (regenerated in this PR, one-word diff).

## Acceptance Criteria

- [ ] Every key at every level is either known or rejected at import.
- [ ] Each rule has a negative test.
- [ ] Validators are pure functions of their input dicts.

## Risks

- `read` on `LrdIdx` stays legal although the emulator ignores it today. E03-14 lists it as a known deviation from the declared timing contract; E04-25 makes the emulator honour it.
