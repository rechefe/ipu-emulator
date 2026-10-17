# Registers

This page is **generated** from `REGISTER_DEFINITIONS` in `ipu_isa.registers`. A register with assembly names is an instruction operand; any other register is named only in the instruction reference, or is emulator storage for debugging.

| Register | Kind | Count | Bytes each | Vector | Assembly names |
|----------|------|------:|-----------:|--------|----------------|
| `R` | MULT | 2 | 128 | yes | `R0`, `R1` |
| `R_WIDE_DEBUG` | MULT | 2 | 512 | yes | — |
| `R_CYCLIC` | MULT | 1 | 512 | yes | — |
| `R_CYCLIC_WIDE_DEBUG` | MULT | 1 | 2048 | yes | — |
| `R_MASK` | MULT | 1 | 128 | yes | — |
| `R_ACC` | ACC | 1 | 512 | yes | — |
| `POST_AAQ_REG` | AAQ | 1 | 512 | yes | — |
| `LR` | LR | 16 | 4 | no | `LR0`–`LR15` |
| `CR` | CR | 16 | 4 | no | `CR0`–`CR15` |
| `MULT_RES` | MISC | 1 | 512 | yes | — |
| `MEM_BYPASS` | MISC | 1 | 128 | yes | — |

`LR` and `CR` hold 32-bit scalars. `CR0` and `CR1` are read-only constants; see [Configuration registers](configuration-registers.md).
