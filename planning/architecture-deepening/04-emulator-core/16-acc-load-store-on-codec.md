# ACC, AGG, load and store handlers written once against the codec

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-16 · Size M · Depends on: E04-13, E04-14 · Behaviour change: none

## Goal

Finish removing format branches from the datapath handlers (Q43, Q48): ACC.ADD/SUB/MAX (and `.FIRST` variants), ACC.STRIDE, ACC.RESHAPE, AGG.*, the load slot (LD.MULT, LD.MULT.CYCLIC, LD.MASK), the store slot (ST.AAQ) and `acc_store` (ST.ACC). The AAQ handler gets codec-based lane I/O only; its activation and quantization logic is untouched (AaQ is out of scope, Q35).

## Background

- ACC handlers split early on wide mode at ipu.py:1423, 1450, 1478 and 1500 and loop per lane with `ipu_add`/`ipu_sub` in narrow mode.
- `IpuState` load/store helpers branch on `wide_vector_debug` (ipu_state.py:181, 200, 213, 230) and choose `r_wide_debug`/`r_cyclic_wide_debug` register names.
- `ACTIVATE.QUANTIZE` has separate narrow, wide and prototyping loops (ipu.py:1705-1829).

## Implementation

1. ACC/AGG: `acc = codec.load_acc(view.read("r_acc"))`, `res = codec.load_acc(view.read("mult_res"))`, `codec.add/sub/max`, `codec.store_acc(...)`. AGG folds use the same functions; keep today's fold order and accumulation precision exactly (pinned by the exhaustive and kernel tests).
2. Load/store: one path that moves `codec.element_bytes * LANES` bytes. Register names come from the format's register layout in `ipu_isa` (E03 derives the registers from the format), not from `"_wide_debug"` string suffixes.
3. `IpuState.write_constant_ones`, `load_r_reg_from_xmem`, `store_r_reg_to_xmem`, `load_r_cyclic_from_xmem`, `store_acc_to_xmem`, `load_r_mask_from_xmem` use the codec.
4. AAQ: replace only the lane read/write code with codec calls. Leave the activation math, quantization and the "requires INT8 mode" check (ipu.py:1815) as they are. Record every AaQ inconsistency found on the way as an open question for E09.

## Tests

- [ ] Output-identical proof: `test_execute.py` (ACC, AGG, RESHAPE, load/store), `test_wide_vector_debug.py`, `test_prototyping_equivalence.py` under `--define=ipu_proto=1`, E02 kernel output digests for every kernel and format in use today.
- [ ] `TestNarrowStoreOverflow` (test_execute.py:3116-3144) and the wide overflow tests (test_wide_vector_debug.py:625-655) still pass: lanes before the overflow written, the lane zeroed, `OverflowError` raised.
- [ ] E04-02 benchmark before/after.

## Acceptance Criteria

- [ ] `grep -n "_wide_vector_active\|WideVectorArithmetic\|wide_vector_debug" src/ipu_emu/slots/` finds only the AAQ module, and only inside the activation/quantization code that is out of scope.
- [ ] `ipu_state.py` load/store helpers have no format branch.
- [ ] Matmul benchmark number recorded against the Q71 target.
