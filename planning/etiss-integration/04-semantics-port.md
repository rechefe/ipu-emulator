# Port instruction semantics and numerics to `IPUFuncs.c`

Part of the [ETISS integration epic](00-epic.md). Depends on #3.

## Goal

Implement every `ipu_<slot>_<name>()` handler declared by the generated
`IPUFuncs_gen.h` as a faithful C port of the Python emulator, bit-exact with
`ipu.py`, `ipu_math.py`, `activations.py`, and `ipu_config.py`.

## Implementation

Structure `src/tools/ipu-etiss/arch/`:

- `ipu_math.c/.h` — FP8 e(x)m(8-x) codec (x = 1…7) and INT8 helpers:
  256-entry decode LUT per dtype built at init; encoder ported from
  `_float32_to_fp8_scalar` (sign/zero/NaN/inf/overflow-clamp/subnormal/carry
  rules, Python `round` semantics); `ipu_mult`, `ipu_add`, `ipu_sub`
  (int32 wrap vs double). `dtype_one_byte`.
- `ipu_activations.c` — the 12 activation functions, double precision,
  `elu_alpha` from `struct IPU`.
- `ipu_regfile.c` — `ipu_snapshot()`, `LRD` pair access, CR read-only rules
  (`CR0`/`CR1`), dstructure decode (`valid_elements`, `partition`, `pad_mode`),
  `R_CYCLIC` slot validation, XMEM row addressing (`base + offset`, 128-byte
  rows, bounds → error return code).
- `IPUFuncs.c` — the handlers, grouped by slot in the same order as `ipu.py`:
  - **lr**: `SET`, `ADD`, `SUB`, `INCR_MOD_POW2`, `INC`, `DEC`, `ADDB`, `ADDBI`, `NOP`.
  - **load/store/acc_store**: `LDR_MULT_REG`, `LDR_CYCLIC_MULT_REG`,
    `LDR_MULT_MASK_REG`, `STR_POST_AAQ_REG`, `STR_ACC_REG` (simulation-only
    slot, kept for parity).
  - **mult**: `MULT.RC.VV/VE/VS`, `MULT.VE`, `MULT.EE`, including
    `_mult_mask_and_shift`, partition vectors, and `pad_mode` fill.
  - **acc**: `ACC.ADD/MAX/SUB` (+`.FIRST`), `ACC.STRIDE`, `AGG.SUM/MAX`
    (+`.FIRST`), `ACC.RESHAPE`.
  - **aaq**: `ACTIVATE.QUANTIZE` (activation + quantize into `POST_AAQ_REG`).
  - **cond**: `BEQ/BNE/BLT/BGE/BR/BKPT` (signed compare rules from
    `_to_signed_reg`; `nextPc = label × 28`).
  - **break**: `BREAK`, `BREAK.IFEQ`, honouring `break_mode`.
- Error model: handlers return `0` or a negative `IPU_ERR_*` code registered
  as ETISS return codes; the generated code propagates it (`if (rc) return rc;`)
  and the runner prints the same message text as `EmulatorError`.
- Stats: increment `mult_active_cycles`, `acc_active_cycles`, `xmem_reads`,
  `xmem_writes` where `Ipu.dispatch_instruction` does (`_Plan.stat`).

Work in the order kernels need it so parity can be measured early:
`fully_connected` (LR ops, `LDR_MULT_REG`, `MULT.RC.VV`, `ACC.ADD*`,
`ACTIVATE.QUANTIZE`, `STR_POST_AAQ_REG`, branches) → `identity` → softmax
kernels (`AGG.*`, `ACC.MAX`, `ACC.RESHAPE`, `ACC.STRIDE`, masks, cyclic loads).

## Tests

- [ ] `test_ipu_math_parity`: exhaustive 256 × 7 FP8 decode and a dense
      float32 grid (plus NaN/inf/±0/subnormal edge cases) for encode, C vs
      Python, driven from pytest through a tiny C test binary.
- [ ] `test_activation_parity`: dense grid for all 12 functions, bit-exact
      float32 after narrowing.
- [ ] Per-handler unit tests as `ipu_etiss_test` programs (one instruction
      each, assembled from inline asm), asserting register/XMEM state from
      `regs.json` / `xmem.bin` against the Python `_run()` result.
- [ ] `fully_connected` for `int8`, `fp8_e4`, `fp8_e5` bit-exact vs golden.
- [ ] All softmax kernels bit-exact vs golden.

## Acceptance Criteria

- [ ] Every instruction in `INSTRUCTION_SPEC` has a C handler; the plugin
      links with no stubs.
- [ ] Numerics tests pass bit-exact.
- [ ] All application kernels produce identical output and cycle counts on
      ETISS and Python.
