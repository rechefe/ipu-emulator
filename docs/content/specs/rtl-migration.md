# RTL migration: `ipu_instr_pkg.sv` since master 694a8a89

For the downstream RTL repository that vendors `ipu_instr_pkg.sv`. It lists
every difference between the package generated on master at 694a8a89 and the
package this documentation describes, and how to update. Each item was checked against the two
generated files (`bazel build //docs:generate_instruction_format_artifacts`
in each tree) and `tools/sv_localparam_diff.sh origin/master`.

## What changes

Four things change: some names, the emulator-only slots leaving the package,
the header line, and the operational AaQ stage of ADR 0007, which removes the
STORE slot and changes widths and opcode values (section 4: a breaking
change). The comment above `ipu_compound_inst_t` changes with the second.

### 1. Eight instructions' enum and union members are renamed (E03-19)

The mnemonic standard renamed nine instructions. Their opcode values are the
same; only the SV names change. The full table, with the old and new
mnemonics, is in [Mnemonic rename](mnemonic-rename.md). The members still in
the package (`STR_POST_AAQ_REG` → `ST.AAQ` was renamed too, and section 4
removes it):

| Old enum member | New enum member | Old union member | New union member | Value |
|---|---|---|---|---|
| `LOAD_INST_OPCODE_LDR_MULT_REG` | `LOAD_INST_OPCODE_LD_MULT` | `ldr_mult_reg` | `ld_mult` | `2'd0` |
| `LOAD_INST_OPCODE_LDR_CYCLIC_MULT_REG` | `LOAD_INST_OPCODE_LD_MULT_CYCLIC` | `ldr_cyclic_mult_reg` | `ld_mult_cyclic` | `2'd1` |
| `LOAD_INST_OPCODE_LDR_MULT_MASK_REG` | `LOAD_INST_OPCODE_LD_MASK` | `ldr_mult_mask_reg` | `ld_mask` | `2'd2` |
| `LR_INST_OPCODE_INCR_MOD_POW2` | `LR_INST_OPCODE_INC_MODP2` | `incr_mod_pow2` | `inc_modp2` | `4'd3` |
| `COND_INST_OPCODE_BLT` | `COND_INST_OPCODE_BLT_S` | `blt` | `blt_s` | `3'd2` |
| `COND_INST_OPCODE_BGE` | `COND_INST_OPCODE_BGE_S` | `bge` | `bge_s` | `3'd3` |
| `COND_INST_OPCODE_BKPT` | `COND_INST_OPCODE_HALT` | `bkpt` | `halt` | `3'd5` |

`STR_ACC_REG` → `ST.ACC` was in the emulator-only ACC_STORE slot, which the
package no longer has (next section) and ADR 0007 removed.

No other enum member, union member, typedef or operand field name changes.

### 2. The package holds the hardware word only (E03-22)

The ACC_STORE and BREAK slots existed only in the emulator, so the package no
longer has them (the emulator-only AAQ_EXACT slot of ADR 0007 was never in
it). Removed:

- the typedefs `acc_store_inst_opcode_t`, `acc_store_slot_u`,
  `acc_store_slot_t`, `break_inst_opcode_t`, `break_slot_u` and
  `break_slot_t`;
- the localparams `ACC_STORE_SLOT_U_WIDTH` (8), `ACC_STORE_SLOT_T_WIDTH` (9),
  `BREAK_SLOT_U_WIDTH` (20) and `BREAK_SLOT_T_WIDTH` (22);
- the members `acc_store_slot` and `break_slot` of `ipu_compound_inst_t`,
  which were its two lowest slots.

So `IPU_COMPOUND_INST_WIDTH` goes from 186 to 169: 31 bits leave with these
two slots, and section 4 removes 9 more and adds 23. The hardware word is the
emulator word without its 40 lowest bits, the 18 of AAQ_EXACT and the 22 of
BREAK: emulator word `>> 40` is the hardware word.

### 3. The header names the content, not the commit (E03-21)

The second line was `// Source: rechefe/ipu-emulator @ <commit>`. It is now
`// Content: sha256:<hash of the rest of the file>`, so the package is the same
file whatever commit it is generated at. Check a vendored copy with:

```bash
bazel run //docs:verify_sv_package -- "$PWD/ipu_instr_pkg.sv"
```

It prints `unmodified` and exits 0 when the file is exactly what its header
names, and exits 1 when it was edited, line endings included. A package
generated on master has no hash, so it fails this check.

### 4. The AaQ stage stores its own row: a breaking wire change (ADR 0007)

[ADR 0007](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0007-operational-aaq-and-nx8.md)
makes the AaQ stage operational: `ACTIVATE.QUANTIZE` activates, quantizes to
one NX8 row and stores it with its metadata word, so the STORE slot and its
`ST.AAQ` are gone. The DS and QF operands name 16-bit CR halves. This changes
widths and opcode values, so the hardware word is not compatible with the one
before.

Removed:

- the typedefs `store_inst_opcode_t`, `store_slot_u` and `store_slot_t`, the
  localparams `STORE_SLOT_U_WIDTH` (8) and `STORE_SLOT_T_WIDTH` (9), and the
  member `store_slot` of `ipu_compound_inst_t`, which was its lowest slot;
- `dstructure_cr_reg_field_t`, the 4-bit DS operand type, and
  `activation_fn_field_t`, the 4-bit activation of the old placeholder.

Added:

- `half_cr_reg_field_t` (5 bits, `CR0_L` = 0 to `CR15_H` = 31, code =
  2 × index + H): the type of every DS and QF operand;
- `lcr_half_reg_field_t` (6 bits: `LR0`–`LR15`, `CR0`–`CR15`, `CR0_L`–`CR15_L`,
  `CR0_H`–`CR15_H`): the type of the LR and COND source operands that read a
  CR, which may now read a sign-extended half;
- `aaq_activation_fn_field_t` (3 bits, the architecture spec's encoding:
  `IDENTITY` 0 to `GENERIC` 6).

Changed:

| Name | Was | Is |
|---|---|---|
| `aaq_inst_opcode_t` | 1 bit: `NOP` 0, `ACTIVATE_QUANTIZE` 1 | 2 bits: `NOP` 0, `LD_LUT` 1, `ACTIVATE_QUANTIZE` 2 |
| `AAQ_SLOT_U_WIDTH` / `AAQ_SLOT_T_WIDTH` | 8 / 9 | 21 / 23 |
| `MULT_SLOT_U_WIDTH` / `MULT_SLOT_T_WIDTH` | 20 / 23 | 26 / 29 |
| `ACC_SLOT_U_WIDTH` / `ACC_SLOT_T_WIDTH` | 13 / 17 | 14 / 18 |
| `COND_SLOT_U_WIDTH` / `COND_SLOT_T_WIDTH` | 20 / 23 | 22 / 25 |
| `IPU_COMPOUND_INST_WIDTH` | 155 | 169 |

The five MULT compute members gain a `qf_<n>` field; their DS field, `cr_idx_<n>`
or `dstructure_cr_idx_<n>`, is now `ds_<n>`, and the `AGG` members' `cr_idx_0`
is `ds_0` (operand renames rename SV fields). `activate_quantize` now holds
`activation_fn_0`, `table_id_1`, `ds_2`, `qf_3` and `table_offset_4`; `ld_lut` is
new. `LOAD_SLOT_*` and `LR_SLOT_*` keep their widths.

`ipu_compound_inst_t` now holds eight slots, from `cond_slot` at the top down
to `aaq_slot`, which ends at bit 0.

## Programs for the RTL

The assembler writes the emulator's word by default: 209 bits, which is 28
bytes per word in a `.bin` file (little-endian) and `0x` plus 54 hex digits per
line in a `.mem` file. Its 40 lowest bits are the AAQ_EXACT and BREAK slots,
which hold their NOP encodings when a word does not use them, not zeros. For
the hardware word, pass `--target hardware`. The words are then 169 bits,
written as 24 bytes per word in `.bin` and `0x` plus 44 hex digits per line in
`.mem`, and an emulator-only instruction is an error. Neither is master's
word: section 4 changed the hardware slots.

```bash
bazel run //src/ipu-as:ipu-as -- assemble --target hardware --input "$PWD/prog.asm" --output "$PWD/prog.mem"
```

`--format` is now optional: by default an `--output` ending in `.bin` gives
bin, and any other gives mem. Master asked for the format when `--format` was
not given.

## In the IPU emulator repository

The packages moved from `src/tools/<package>` to `src/<package>`, and the
assembler and emulator also lost their `-py` suffix (`src/tools/ipu-as-py` is
now `src/ipu-as`, `src/tools/ipu-emu-py` is now `src/ipu-emu`), so every label
under them changed. The Python import names `ipu_as` and `ipu_emu` did not.
The test that elaborates the package with slang moved to `ipu-gen` (E08-13),
with the generator; the package's golden, the tests that read it and its
update target are in `ipu-gen` too. A CI job in the RTL repository that calls
one of master's labels needs the new one; the rows without a master label are
new since master:

| Master label or path | Label or path now |
|---|---|
| `//src/tools/ipu-as-py:ipu-as` | `//src/ipu-as:ipu-as` |
| `//src/tools/ipu-as-py:test_sv_package_compiles` | `//src/ipu-gen:test_sv_package_compiles` |
| `//src/tools/ipu-as-py:<target>` | `//src/ipu-as:<target>` |
| `//src/tools/ipu-emu-py:<target>` | `//src/ipu-emu:<target>` |
| none | `src/ipu-gen/test/baselines/ipu_instr_pkg.sv`: a checked-in golden of the package |
| none | `//src/ipu-gen:test_sv_package_golden`: the package against the golden, and every `*_WIDTH` pinned by name |
| none | `//src/ipu-gen:update_sv_golden`: regenerates the golden |
| none | `//src/ipu-gen:test_sv_rename_table`: the [Mnemonic rename](mnemonic-rename.md) table checked against the golden |
| none | `tools/sv_localparam_diff.sh <base-ref>`: the `localparam` lines of the package against those of another commit |

## How to update

1. Generate the package:
   `bazel build //docs:generate_instruction_format_artifacts`, then copy
   `bazel-bin/docs/ipu_instr_pkg.sv` into the RTL repository.
2. Check the copy, by absolute path since it is in another repository:
   `bazel run //docs:verify_sv_package -- /abs/path/to/ipu_instr_pkg.sv`.
3. Replace the seven old enum members and the seven old union members in
   section 1.
4. Remove every use of the ACC_STORE, BREAK and STORE typedefs, localparams
   and `ipu_compound_inst_t` members (sections 2 and 4), and of
   `dstructure_cr_reg_field_t` and `activation_fn_field_t`.
5. Size the instruction word, memories and buses by `IPU_COMPOUND_INST_WIDTH`
   (169). A constant 186 or 155 is now wrong.
6. Decode the AAQ slot with its new 2-bit opcode, the MULT, ACC and COND slots
   with their new fields, and the DS and QF operands as CR halves (section 4).
7. Find every literal bit range of the instruction word: every slot moved.
   `cond_slot` was `[185:163]` and is `[168:144]`. Read the slots through the
   members of `ipu_compound_inst_t` instead.
8. Rename the old mnemonics in the RTL repository's test sources, or they no
   longer assemble (`BKPT` is now `HALT`, for example):
   `bazel run //tools:rename_mnemonics -- --check /abs/path/to/rtl-repo` lists
   what would change, and the same command without `--check` rewrites it. It
   reads only files git tracks, so commit new test sources first.
9. Migrate the test programs to the ADR 0007 syntax (`ST.AAQ` is gone; the DS
   and QF operands are halves such as `CR15.L`) and assemble them with
   `--target hardware`. To compare with a word assembled for the emulator,
   take its bits `[208:40]`: the emulator word shifted right by 40 is the
   hardware word.
