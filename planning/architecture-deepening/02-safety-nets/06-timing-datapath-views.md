# Timing characterization: datapath registers and end-of-word control

Part of [Epic E02: Safety nets before the refactors](00-epic.md) · ID E02-06 · Size S · Depends on: E01 · Behaviour change: none

## Goal

Pin which value of each implicitly read datapath register (R0, R1, R_CYCLIC, R_MASK, MULT_RES, R_ACC, POST_AAQ_REG) a consumer sees when the same instruction word writes it, plus the end-of-word control rules (BKPT, break and resume, idle words, branch signedness). Same purpose as E02-05: E03 declares the contract (Q32), E04 rebuilds the cycle and handlers, and these tests prove the refactors are output-identical. The two datapath rows that Q32 changes are separate tests, flipped by E04-26 and E04-27.

## Background

- MULT reads R0/R1 through the `ra` operand (`read: "snapshot"`) and R_CYCLIC from the cycle-start copy (#157); kernels rely on this in about 204 bundles in 69 files (plan 01).
- MULT reads R_MASK from the live register file (`ipu_emu/ipu.py:945` `get_r_mask()`, and `ipu.py:909` for the all-ones shortcut), so a same-word `LDR_MULT_MASK_REG` is visible. No kernel co-issues the two.
- ACC.ADD/SUB/MAX/STRIDE and AGG.* read this cycle's MULT_RES from the live file (`ipu.py:1256-1500`, `1653-1698`). ACC.RESHAPE reads the cycle-start MULT_RES (`ipu.py:1603`). No kernel co-issues MULT and ACC.RESHAPE.
- ACC/AGG read the old R_ACC from the cycle-start copy (`ipu.py:1421`, `1448`, `1476`, `1668`, `1701`). ACTIVATE.QUANTIZE reads R_ACC from the live file (`ipu.py:1765`); pinned by `test_agg_and_activate_quantize_same_cycle_sees_live_r_acc` (`test_execute.py:1559`), although the spec summary says "snapshot".
- STR_POST_AAQ_REG stores the live POST_AAQ_REG (`ipu.py:1838`); STR_ACC_REG stores the live R_ACC.
- `execute_bkpt` sets `pc = INST_MEM_SIZE` (`ipu.py:1875-1877`) after every other slot of the word has run; COND is dispatched last.
- `execute_blt`/`execute_bge` compare signed 32-bit values (`_to_signed_reg`, `ipu.py:1853-1861`).
- Q32 targets: MULT reads R0/R1/R_CYCLIC/R_MASK at cycle start; MULT_RES is forwarded to ACC/AGG/RESHAPE; R_ACC is forwarded to AAQ.

## Implementation

Add `src/tools/ipu-emu-py/test/test_timing_datapath_views.py`, one minimal program per row:

| Test | Producer → consumer in one word | Reads today | Q32 target |
|---|---|---|---|
| `test_mult_reads_r0_r1_at_cycle_start` | `LDR_MULT_REG` → MULT `ra` | cycle start | same |
| `test_mult_reads_r_cyclic_at_cycle_start` | `LDR_CYCLIC_MULT_REG` → MULT.RC.* | cycle start | same |
| `test_mult_reads_r_mask_forwarded` | `LDR_MULT_MASK_REG` → MULT.* mask | forwarded | **cycle start (E04-27)** |
| `test_acc_reads_this_cycle_mult_res` | MULT.* → ACC.ADD/SUB/MAX(.FIRST)/STRIDE | forwarded | same |
| `test_agg_reads_this_cycle_mult_res` | MULT.* → AGG.* | forwarded | same |
| `test_reshape_reads_previous_mult_res` | MULT.* → ACC.RESHAPE | cycle start | **forwarded (E04-26)** |
| `test_acc_reads_old_r_acc_at_cycle_start` | ACC.* accumulate onto R_ACC | cycle start | same |
| `test_activate_reads_this_cycle_r_acc` | ACC/AGG → ACTIVATE.QUANTIZE (keep the existing `test_execute.py:1559` test; this one covers ACC.ADD as well) | forwarded | same |
| `test_store_reads_this_cycle_post_aaq` | ACTIVATE.QUANTIZE → STR_POST_AAQ_REG | forwarded | same |
| `test_acc_store_reads_this_cycle_r_acc` | ACC.* → STR_ACC_REG | forwarded | same |
| `test_bkpt_halts_after_whole_word_retires` | word with BKPT plus a store and an LR write | all slots retire, then halt | same (BKPT becomes HALT, E03) |
| `test_branches_compare_signed` | `BLT`/`BGE` with a negative LR | signed | same (BLT.S/BGE.S, E03) |
| `test_break_has_no_side_effects_and_resume_rereads` | `BREAK` word with an LR write; edit a register while stopped, resume | no effect before resume; resumed word sees the edit | same |
| `test_idle_word_advances_pc_once` | empty instruction-memory entry | pc + 1, `mult_idle_cycles` + 1 | same |

Each test docstring states the rule in one sentence and, for the two flips, names the E04 issue that changes it.

## Tests

- [ ] All tests pass on the base commit.
- [ ] Together with E02-05, every row of plan 01's forwarding table has exactly one test.

## Acceptance Criteria

- [ ] The two Q32 datapath flips are isolated tests.
- [ ] Runtime under 5 s.

## Risks

- `test_reshape_mult_res_reads_from_snapshot` (`test_execute.py:1886`) already pins the RESHAPE row. Keep it; E04-26 flips both, or deletes the older one in favour of this file. Note this in the test docstring.
- Wide-vector debug mode has its own read paths for R0/R1 (`ipu.py:1246`). Run the four MULT rows in both the default element format and wide FP32 mode, parametrized.
