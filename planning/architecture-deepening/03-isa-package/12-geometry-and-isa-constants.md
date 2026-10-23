# Geometry and ISA constants live in `ipu_isa`

Part of [Epic E03: ISA package](00-epic.md) · ID E03-12 · Size S · Depends on: E03-05 · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

Every machine fact that is not an instruction but that the assembler, emulator, harness and kernels all rely on gets one home in `ipu_isa` (Q31, L33): the lane count, instruction-memory depth, the constant registers CR0/CR1, and the dstructure CR field layout. Consumers import them instead of restating them. Element-format-dependent sizes (element and row bytes) come in E03-13.

## Background

- Lane count: `ipu_isa.registers.get_lane_count()` exists (`registers.py:197-205`), yet `LANES = get_lane_count()` is restated in `ipu_emu/ipu.py:85`, `stats.py:24`, `alias_observer.py:14`, and `LANES`/`ROW_BYTES`/`ELEM_BYTES` are defined at module level 171 times under `src/tools/ipu-apps/`.
- Instruction-memory depth is written twice: `INST_MEM_SIZE = 1024` (`ipu_emu/ipu_state.py:30`) and `MAX_PROGRAM_SIZE = 1024` (`ipu_as/ipu_token.py:6`; E03-05 moves this one to `ipu_isa.geometry`). Requirement OQ-4 (`docs/content/requirements/open-questions.md:17`) proposes 1024 words "as a parameter".
- ISA facts in the emulator: `CR_ZERO_REG_INDEX/VALUE`, `CR_ONE_REG_INDEX/VALUE`, `CR_READ_ONLY_INITIAL_VALUES`, the dstructure bit layout (`DSTRUCTURE_*`), `Partition`, `PadMode`, `DStructureConfig`, `encode_dstructure`/`decode_dstructure` (`ipu_emu/ipu_config.py:10-110`). 21 files import `ipu_config`.
- The pseudo-instructions `B`, `BZ`, `BNZ` hard-code the literal `"CR0"` and rely on CR0 being zero; nothing checks it against `CR_ZERO_REG_INDEX`.
- XMEM size is `XMEM_SIZE_BYTES = 1 << 29` (`ipu_emu/xmem.py:14`). E06-04 needs XMEM capacity from `ipu_isa` geometry.
- `CR_DSTRUCTURE_REG_INDEX = 15` (`ipu_config.py:12`) encodes a convention, not an ISA rule (CLAUDE.md: any CR0–CR15 is a valid operand).

## Implementation

1. `ipu_isa/geometry.py`:
   ```python
   LANES: Final[int]                 # from REGISTER_DEFINITIONS (r_acc word view), 128
   INST_MEM_WORDS: Final[int]        # 1024; MAX_PROGRAM_SIZE and INST_MEM_SIZE alias it in their packages
   R_CYCLIC_ELEMENTS: Final[int]     # 512, from REGISTER_DEFINITIONS
   XMEM_BYTES: Final[int]            # 1 << 29; tagged EMULATOR (the emulator's address space, not a hardware size)
   ```
2. `ipu_isa/constants.py`: move the CR constants and the dstructure layout, `Partition`, `PadMode`, `DStructureConfig`, `encode_dstructure`, `decode_dstructure` from `ipu_config.py`. Keep `CR_DSTRUCTURE_REG_INDEX` out of the ISA: rename it `DEFAULT_DSTRUCTURE_CR` and leave it in `ipu_emu` with a comment that it is a convention. `ipu_config.py` keeps only emulator settings; its importers switch to `ipu_isa.constants` in this PR (no re-export shim, D5).
3. `ipu_emu` uses `geometry.LANES` and `geometry.INST_MEM_WORDS` instead of local copies (`ipu.py:85`, `stats.py:24`, `alias_observer.py:14`, `ipu_state.py:30`, `xmem.py:14`).
4. Schema rule (E03-03 list): a pseudo-instruction literal that names a CR must name the register at `CR_ZERO_REG_INDEX` or `CR_ONE_REG_INDEX`, and the pseudo's docs must state which.
5. Out of scope: the ~171 module-level constants in `ipu-apps`. E06's harness and E07's kernel migration delete them; E07 adds a lint against new ones.

## Tests

- [ ] `ipu-isa/test/test_constants.py`: dstructure encode/decode round trip over every `Partition` and `PadMode`; `LANES == 128`, `INST_MEM_WORDS == 1024`.
- [ ] Pseudo validation negative test: changing the `B` literal to `CR2` fails at load.
- [ ] E02 baselines unchanged; kernel case records unchanged (CR values after setup prove the dstructure encoding moved without change).

## Acceptance Criteria

- [ ] Lane count, instruction-memory depth, CR0/CR1 values and the dstructure layout are each defined once, in `ipu_isa`.
- [ ] `ipu_emu` contains no copy of them.

## Open question

- OQ-4 leaves IMEM depth open. This issue keeps 1024 as one constant; making it a run-time parameter is a separate decision.
