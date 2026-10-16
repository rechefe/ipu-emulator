# Mnemonic rename (E03-19): SystemVerilog names

The mnemonic standard (Q33, Q45) renamed nine instructions. Opcodes did not
change, so every program assembles to the same bits, and every `*_WIDTH`
localparam, typedef name and operand field name in `ipu_instr_pkg.sv` is
unchanged. Only these enum members and union members of the package are
renamed. A downstream RTL repository that vendors the package replaces each
old name with the new one.

`tools/rename_mnemonics.py` applies the same rename to assembly sources, Python,
JavaScript and docs on another branch or checkout:

```bash
bazel run //tools:rename_mnemonics -- --check   # list what would change
bazel run //tools:rename_mnemonics              # rewrite the workspace in place
```

| Old mnemonic | New mnemonic | Old enum member | New enum member | Old union member | New union member | Value |
|---|---|---|---|---|---|---|
| `LDR_MULT_REG` | `LD.MULT` | `LOAD_INST_OPCODE_LDR_MULT_REG` | `LOAD_INST_OPCODE_LD_MULT` | `ldr_mult_reg` | `ld_mult` | `2'd0` |
| `LDR_CYCLIC_MULT_REG` | `LD.MULT.CYCLIC` | `LOAD_INST_OPCODE_LDR_CYCLIC_MULT_REG` | `LOAD_INST_OPCODE_LD_MULT_CYCLIC` | `ldr_cyclic_mult_reg` | `ld_mult_cyclic` | `2'd1` |
| `LDR_MULT_MASK_REG` | `LD.MASK` | `LOAD_INST_OPCODE_LDR_MULT_MASK_REG` | `LOAD_INST_OPCODE_LD_MASK` | `ldr_mult_mask_reg` | `ld_mask` | `2'd2` |
| `STR_POST_AAQ_REG` | `ST.AAQ` | `STORE_INST_OPCODE_STR_POST_AAQ_REG` | `STORE_INST_OPCODE_ST_AAQ` | `str_post_aaq_reg` | `st_aaq` | `1'd0` |
| `STR_ACC_REG` | `ST.ACC` | `ACC_STORE_INST_OPCODE_STR_ACC_REG` | `ACC_STORE_INST_OPCODE_ST_ACC` | `str_acc_reg` | `st_acc` | `1'd0` |
| `INCR_MOD_POW2` | `INC.MODP2` | `LR_INST_OPCODE_INCR_MOD_POW2` | `LR_INST_OPCODE_INC_MODP2` | `incr_mod_pow2` | `inc_modp2` | `4'd3` |
| `BLT` | `BLT.S` | `COND_INST_OPCODE_BLT` | `COND_INST_OPCODE_BLT_S` | `blt` | `blt_s` | `3'd2` |
| `BGE` | `BGE.S` | `COND_INST_OPCODE_BGE` | `COND_INST_OPCODE_BGE_S` | `bge` | `bge_s` | `3'd3` |
| `BKPT` | `HALT` | `COND_INST_OPCODE_BKPT` | `COND_INST_OPCODE_HALT` | `bkpt` | `halt` | `3'd5` |

The pseudo-instructions `BGT` and `BLE` became `BGT.S` and `BLE.S`; they have
no SV name. `BLT.U`, `BGE.U`, `BGT.U` and `BLE.U` are reserved for unsigned
branches (`ipu_isa.RESERVED_MNEMONICS`): no spec may define them, and the
assembler rejects them as "reserved, not implemented".

The `ACC_STORE` slot is emulator-only; E03-22 removes it from the SV package.

`//src/tools/ipu-as-py:test_sv_rename_table` checks this table against the SV
golden.
