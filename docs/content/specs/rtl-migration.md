# RTL migration: `ipu_instr_pkg.sv` from master to this branch

For the downstream RTL repository that vendors `ipu_instr_pkg.sv`. It lists
every difference between the package generated on master (694a8a89) and on
this branch, and how to update. Each item was checked against the two
generated files (`bazel build //docs:generate_instruction_format_artifacts`
in each tree) and `tools/sv_localparam_diff.sh origin/master`.

## What changes

No opcode value changes, and the hardware slots keep their order, widths and
operand fields. Three things change: some names, the emulator-only slots
leaving the package, and the header line.

### 1. Nine enum and union members are renamed (E03-19)

The mnemonic standard renamed nine instructions. Their opcode values are the
same; only the SV names change. The full table, with the old and new
mnemonics, is in [Mnemonic rename](mnemonic-rename.md). The members still in
the package:

| Old enum member | New enum member | Old union member | New union member | Value |
|---|---|---|---|---|
| `LOAD_INST_OPCODE_LDR_MULT_REG` | `LOAD_INST_OPCODE_LD_MULT` | `ldr_mult_reg` | `ld_mult` | `2'd0` |
| `LOAD_INST_OPCODE_LDR_CYCLIC_MULT_REG` | `LOAD_INST_OPCODE_LD_MULT_CYCLIC` | `ldr_cyclic_mult_reg` | `ld_mult_cyclic` | `2'd1` |
| `LOAD_INST_OPCODE_LDR_MULT_MASK_REG` | `LOAD_INST_OPCODE_LD_MASK` | `ldr_mult_mask_reg` | `ld_mask` | `2'd2` |
| `STORE_INST_OPCODE_STR_POST_AAQ_REG` | `STORE_INST_OPCODE_ST_AAQ` | `str_post_aaq_reg` | `st_aaq` | `1'd0` |
| `LR_INST_OPCODE_INCR_MOD_POW2` | `LR_INST_OPCODE_INC_MODP2` | `incr_mod_pow2` | `inc_modp2` | `4'd3` |
| `COND_INST_OPCODE_BLT` | `COND_INST_OPCODE_BLT_S` | `blt` | `blt_s` | `3'd2` |
| `COND_INST_OPCODE_BGE` | `COND_INST_OPCODE_BGE_S` | `bge` | `bge_s` | `3'd3` |
| `COND_INST_OPCODE_BKPT` | `COND_INST_OPCODE_HALT` | `bkpt` | `halt` | `3'd5` |

The ninth, `STR_ACC_REG` → `ST.ACC`, is in the emulator-only ACC_STORE slot,
which the package no longer has (next section).

No other enum member, union member, typedef or operand field name changes.

### 2. The package holds the hardware word only (E03-22)

The ACC_STORE and BREAK slots exist only in the emulator, so the package no
longer has them. Removed:

- the typedefs `acc_store_inst_opcode_t`, `acc_store_slot_u`,
  `acc_store_slot_t`, `break_inst_opcode_t`, `break_slot_u` and
  `break_slot_t`;
- the localparams `ACC_STORE_SLOT_U_WIDTH` (8), `ACC_STORE_SLOT_T_WIDTH` (9),
  `BREAK_SLOT_U_WIDTH` (20) and `BREAK_SLOT_T_WIDTH` (22);
- the members `acc_store_slot` and `break_slot` of `ipu_compound_inst_t`,
  which were its two lowest slots.

So `IPU_COMPOUND_INST_WIDTH` goes from 186 to 155. Every other `*_WIDTH` is
unchanged. `ipu_compound_inst_t` keeps the same nine slots in the same order,
from `cond_slot` at the top down to `store_slot`, which now ends at bit 0.

In bits: the new 155-bit word is the old word without its 31 lowest bits,
the 9 bits of ACC_STORE and the 22 of BREAK. Old word `>> 31` is the new word,
and bits `[185:31]` of the old word are bits `[154:0]` of the new one.

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

## Programs for the RTL

The assembler writes the emulator's word by default: 186 bits, which is 24
bytes per word in a `.bin` file and 48 hex digits per line in a `.mem` file.
That is the same output as master's assembler. For the hardware word, pass
`--target hardware`. The words are then 155 bits, written as 20 bytes per word
in `.bin` and 40 hex digits per line in `.mem`, and an emulator-only
instruction is an error.

```bash
bazel run //src/tools/ipu-as-py:ipu-as -- assemble --target hardware --input "$PWD/prog.asm" --output "$PWD/prog.mem"
```

The output format comes from the `--output` suffix (`.bin` or `.mem`). Master
asked for it instead (`--format`).

## In this repository

The test that elaborates the package with slang moved from
`//src/tools/ipu-as-py:test_sv_package_compiles` to
`//src/tools/ipu-gen:test_sv_package_compiles` (E08-13), with the generator.
New since master:

- `//src/tools/ipu-as-py:test_sv_package_golden`: a checked-in golden of the
  package, and every `*_WIDTH` pinned by name;
- `//src/tools/ipu-as-py:test_sv_rename_table`: the [Mnemonic rename](mnemonic-rename.md)
  table checked against that golden;
- `tools/sv_localparam_diff.sh <base-ref>`: the `localparam` lines of the
  package against those of another commit.

## How to update

1. Generate the package on this branch:
   `bazel build //docs:generate_instruction_format_artifacts`, then copy
   `bazel-bin/docs/ipu_instr_pkg.sv` into the RTL repository.
2. Check the copy: `bazel run //docs:verify_sv_package -- <path>`.
3. Replace the eight old enum members and union members in section 1.
4. Remove every use of the ACC_STORE and BREAK typedefs, localparams and
   `ipu_compound_inst_t` members (section 2).
5. Size the instruction word, memories and buses by `IPU_COMPOUND_INST_WIDTH`
   (155). A constant 186 is now wrong.
6. Assemble test programs with `--target hardware`. A program assembled for the
   emulator, or by master's assembler, is the hardware word shifted left by 31
   bits: shift it right by 31 to compare.
