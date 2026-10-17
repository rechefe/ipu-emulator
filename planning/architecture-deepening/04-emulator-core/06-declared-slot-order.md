# Emulator consumes the declared slot execution order

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-06 · Size S · Depends on: E03 (declared timing contract in `ipu_isa`), E02 (timing characterization tests) · Behaviour change: none

## Goal

Delete every hard-coded slot order in the emulator. The emulator iterates `SLOT_EXECUTION_ORDER` from `ipu_isa`, which E03 declares together with the per-instruction register views (Q32). One edit in the ISA package then changes the order everywhere (D2).

## Background

- `Ipu._SLOT_ORDER = ("load", "mult", "acc", "aaq", "store", "acc_store", "cond")` (ipu.py:2134) is used only by the prototyping `_cycle_plan` (ipu.py:2186).
- `execute_vliw_cycle` (ipu.py:2297-2304) and `execute_vliw_cycle_skip_break` (ipu.py:2336-2343) each hard-code `lr, load, mult, acc, aaq, store, acc_store, cond` as eight calls.
- `COMPOUND_LAYOUT_SLOT_ORDER` (instruction_spec.py) is the bit-layout order (cond first, break last). It is a different thing and is not touched.
- E02 adds timing characterization tests that pin every row of the plan-01 forwarding table (one minimal two-bundle program per producer slot, register and consumer). They are the proof that this change is output-identical.

## Implementation

1. Import `SLOT_EXECUTION_ORDER` (expected: `("break", "lr", "load", "mult", "acc", "aaq", "store", "acc_store", "cond")`) from `ipu_isa`.
2. Replace both eight-call blocks with one loop over `SLOT_EXECUTION_ORDER`, skipping `break` (evaluated first, separately) and dispatching `lr` through `_dispatch_lr_slots`.
3. Delete `_SLOT_ORDER`; `_cycle_plan` iterates the same tuple.
4. Add an import-time assertion (as an exception, not `assert`) that the set of slots in `SLOT_EXECUTION_ORDER` equals the set of slots with plans, so a new slot cannot be silently skipped.

## Tests

- [ ] E02 timing characterization tests pass unchanged.
- [ ] `test_execute.py` (including `test_concurrent_load_and_store_same_cycle`, `test_inc_dec_same_cycle`, the LR conflict tests) and `test_prototyping_equivalence.py` under `--define=ipu_proto=1` pass.
- [ ] New: a test that monkeypatches a reordered `SLOT_EXECUTION_ORDER` into the module and shows the dispatch order follows it (spy on dispatched slot names).
- [ ] E02 kernel output digests unchanged.

## Acceptance Criteria

- [ ] `grep -n '"load", "mult"' src/tools/ipu-emu-py/src/ipu_emu/` finds nothing.
- [ ] The slot order exists only in `ipu_isa`.
