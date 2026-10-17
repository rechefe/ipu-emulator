# Timing characterization: which LR value each consumer reads

Part of [Epic E02: Safety nets before the refactors](00-epic.md) · ID E02-05 · Size S · Depends on: E01 · Behaviour change: none

## Goal

Pin, with one minimal program per row, which LR value every consumer reads when the same instruction word also writes that LR: the value at cycle start, or the forwarded value written earlier in the cycle. Today this is decided by the order of calls in `ipu.py` and by each handler. E03 declares the contract as data (Q32) and E04 rebuilds the cycle (E04-06, E04-07, E04-12); these tests prove those refactors keep today's behaviour. The two rows that decision Q32 changes are separate tests, so the E04 behaviour PR (E04-25) flips exactly those tests and nothing else.

## Background

- Slot dispatch order is hard-coded: `execute_vliw_cycle` (`ipu_emu/ipu.py:2297-2304`) and `execute_vliw_cycle_skip_break` (`ipu.py:2336-2343`) call `lr, load, mult, acc, aaq, store, acc_store, cond`. BREAK is evaluated first (`ipu.py:2290-2292`).
- Operand reads with a `read` flag are resolved in `_resolve_operand` (`ipu.py:795-833`) from either the cycle-start copy (`_snapshot_buffer`) or the live register file. Several handlers read LRs themselves: ACC.RESHAPE `source`/`dest` from the cycle-start copy (`ipu.py:1589-1590`), its `reshape_mask` LR from the live file (`ipu.py:1594`), MULT.RC.VE `src` from the live file (`ipu.py:1234`).
- Existing pins: `test_inc_dec_same_cycle` (`test_execute.py:288`), LR conflict tests (`test_execute.py:255`, `:398`). Most rows below have no test.
- Q32 target behaviour: the Control stage (LR, COND) reads LRs at cycle start; every other stage reads post-LR-write values; LR sub-slots read cycle-start values.

## Implementation

Add `src/tools/ipu-emu-py/test/test_timing_lr_views.py`. Each test assembles a two- or three-word program, runs it, and asserts one observable value. Programs write LR*n* in an LR sub-slot and read it in the named consumer in the same word; the expected value shows which version was read. Use the shared `_run()` helper pattern from `test_execute.py:65` (E04-01 later moves it into `conftest.py`).

| Test | Consumer and operand | Reads today | Q32 target |
|---|---|---|---|
| `test_lr_alu_sources_read_cycle_start` | `SET`/`ADD`/`SUB`/`INCR_MOD_POW2`/`ADDB` sources | cycle start | same |
| `test_lr_subslots_read_cycle_start` | `INC LR0 1 ; ADD LR1 LR0 CR1` in one word | cycle start | same |
| `test_inc_dec_read_own_dest_at_cycle_start` | `INC`/`DEC` on the LR another sub-slot reads | cycle start | same |
| `test_cond_operands_read_cycle_start` | `BEQ`/`BNE`/`BLT`/`BGE`/`BR` | cycle start | same |
| `test_break_ifeq_reads_cycle_start` | `BREAK.IFEQ reg` | cycle start | same |
| `test_load_offset_and_index_read_forwarded` | `LDR_MULT_REG`/`LDR_CYCLIC_MULT_REG`/`LDR_MULT_MASK_REG` `offset`, `index` | forwarded | same |
| `test_store_offsets_read_forwarded` | `STR_POST_AAQ_REG`, `STR_ACC_REG` `offset` | forwarded | same |
| `test_mult_lr_operands_read_forwarded` | MULT `rc_idx`, `ra_idx`, `mask_shift` | forwarded | same |
| `test_mult_rc_ve_src_reads_forwarded` | MULT.RC.VE `src` (no `read` flag) | forwarded | same |
| `test_acc_stride_offset_reads_forwarded` | ACC.STRIDE `offset` | forwarded | same |
| `test_reshape_mask_lr_reads_forwarded` | ACC.RESHAPE `reshape_mask` given as an LR | forwarded | same |
| `test_agg_dest_slot_reads_cycle_start` | AGG.SUM/AGG.MAX(.FIRST) `dest_slot` | cycle start | **forwarded (E04-25)** |
| `test_reshape_source_dest_read_cycle_start` | ACC.RESHAPE `source`/`dest` (LRD pairs) | cycle start | **forwarded (E04-25)** |
| `test_lr_write_conflict_raises` | two sub-slots write one LR; `ADDB` LRD overlapping another LR write | raises | same (becomes fault F-3 in E04-19) |
| `test_lr_subslot_exception_leaves_no_partial_write` | a word whose second LR sub-slot raises | no LR written | same |

Each test docstring states the rule in one sentence, using "cycle start" and "forwarded" (GLOSSARY.md), and the test that will change names its E04 issue.

The last row documents a known divergence: the prototyping path executes LR sub-slots one at a time (`ipu.py:2245`), the faithful path resolves all three first (`ipu.py:1186-1206`). The test runs faithful only here; E02-08 runs the same program in prototyping mode and records the outcome as an expected failure until E04-07 closes it.

## Tests

- [ ] All tests above pass on the base commit in a default build.
- [ ] Every row of plan 01's forwarding table that involves an LR has exactly one test.

## Acceptance Criteria

- [ ] The two Q32 flips are each one test, named so the E04-25 PR changes only their expected values.
- [ ] Test runtime under 5 s.

## Risks

- A program that touches several rules at once makes a flip change two tests. Keep each program minimal: one producer, one consumer, one asserted value.
- Mnemonics in these programs are renamed by E03's script (Q45); the script must cover `src/tools/**/test/*.py`.
