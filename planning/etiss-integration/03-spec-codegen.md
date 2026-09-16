# `gen_etiss.py`: generate struct, bit layout, and decode callback from `instruction_spec.py`

Part of the [ETISS integration epic](00-epic.md). Depends on #2 (build) and can
start in parallel with it once #1 passes.

## Goal

Produce every table the C/C++ side needs from the Python single source of
truth, so the ETISS plugin never contains a hand-typed opcode, bit offset, or
register size. Mirrors `gen_codegen.py` (SystemVerilog) but targets ETISS.

## Implementation

Add `ipu_as/gen_etiss.py` (Jinja2 templates under `ipu_as/templates/etiss/`)
plus a `py_binary` and `genrule`s in `src/tools/ipu-etiss/BUILD.bazel`.

Generated files:

- `IPU_gen.h` — `struct IPU` (first member `ETISS_CPU cpu`), one field per
  `REGISTER_DEFINITIONS` entry (`LR[16]`, `CR[16]`, `R[2][128]`, `R_CYCLIC[512]`,
  `R_MASK[128]`, `R_ACC[512]`, `MULT_RES[512]`, `POST_AAQ_REG[512]`), the
  shadow copies used for `"read": "snapshot"` operands, config fields
  (`dtype`, `elu_alpha`, `break_mode`), stat counters, and the
  `IMEM_*`/`XMEM_*` address constants. Opcode enums from
  `create_emulator_constants()`.
- `IPU_layout_gen.h` — for each slot instance (`cond`, `lr0..2`, `load`, …):
  opcode `[msb:lsb]`, and for each union field its `[msb:lsb]`, walked exactly
  like `CompoundInst.get_fields()` (LSB-first). Also `IPU_WORD_BITS = 186`,
  `IPU_WORD_BYTES = 28`.
- `IPUFuncs_gen.h` — a prototype per instruction:
  `void ipu_<slot>_<name>(ETISS_CPU*, ETISS_System*, void* const*, <operands…>)`
  with operand names from the spec, in spec order. `execute_fn` names map
  1:1 (`execute_ldr_mult_reg` → `ipu_ldr_mult_reg`). Break-slot handlers
  return `int` (`IPU_BREAK`/`IPU_CONTINUE`).
- `IPUDecode_gen.cpp` — the `IPU_VLIW` `InstructionDefinition` callback and
  ASM printer (see spec §7). Operand extraction uses `SLOT_UNIONS[*].opcode_bindings`
  so a field shared by several instructions is read once per opcode case.
- `ipu_emu/etiss/layout.py` — the same address constants and word geometry
  for the Python runner (imported, not re-typed).

Rules: no opcode literals; slot order from `COMPOUND_LAYOUT_SLOT_ORDER`;
execution order from the same list `Ipu.execute_vliw_cycle` uses (make that
order a named constant in `instruction_spec.py` and have both consumers read
it).

## Tests

- [ ] `test_gen_etiss.py`: for 10k random words, the offsets in
      `IPU_layout_gen.h` extract the same values as `decode_instruction_word`.
      (Implemented by generating a tiny C test that prints fields, or by
      evaluating the generated offsets in Python — prefer the C route so the
      header itself is exercised.)
- [ ] The generated prototype header compiles against a stub `IPUFuncs.c`
      that has every handler; removing one handler breaks the link.
- [ ] `gen_etiss.py` is deterministic (golden-file test of generated output
      for the current spec, regenerated with a single command).

## Acceptance Criteria

- [ ] All four C/C++ artefacts and the Python layout module are generated
      under Bazel from `instruction_spec.py` / `registers.py`.
- [ ] Field extraction is proven equal to the Python decoder.
- [ ] Adding an instruction to the spec adds its prototype and decode case
      with no manual C++ edits.
