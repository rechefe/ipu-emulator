# Observers consume fact records, not `Ipu` privates

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-11 · Size M · Depends on: E04-05, E04-09 · Behaviour change: none

## Goal

The alias observer, the profiler and test instrumentation see the emulator only through public records: `IssueFacts` per op, the `CycleObserver` protocol per cycle, and ranged register/XMEM events. No observer imports `ipu.py` internals. This is also the event stream that E04-29's commit record is built from.

## Background

- `alias_observer.py:73` imports `_SLOT_PLANS, _SLOT_OPCODE_FIELDS, _LR_PLANS, _LR_OPCODE_FIELDS` from `ipu_emu.ipu`.
- It calls `Ipu` privates: `_lrd_lr_indices` (:137), `_effective_mult_mask` (:141), `_mult_identity` (:145), `_wide_cr_scalar_byte_as_int32` (:225, :230), `_snapshot_buffer` (:256), `_row_size_bytes` (:259, :277), `_element_width_bytes` (:279).
- The profiler bridge is called with `start(self, "lr", plan, kwargs, fields)` from `_dispatch_lr_slots` (ipu.py:1209) and `dispatch_instruction` (ipu.py:1952).
- The mobilevit measuring fixture (ipu-apps/test/mobilevit/fixture_packed_l4_measure.py:51-74) monkeypatches `Ipu.dispatch_instruction`, so it counts nothing in prototyping mode, where dispatch is bypassed.

## Implementation

1. Extend `CycleObserver`:
   ```python
   class CycleObserver(Protocol):
       def begin(self, pc: int) -> None: ...
       def issued(self, facts: IssueFacts) -> None: ...
       def retired(self, facts: IssueFacts, result: Any) -> None: ...
       def end(self) -> None: ...
   ```
   `Cycle` supports several observers (a tuple, checked once per cycle for emptiness).
2. Move the helpers the observer needs to public functions in a small `ipu_emu/geometry.py` module: `lrd_lr_indices`, `effective_mult_mask`, `row_size_bytes(fmt)`, `element_width_bytes(fmt)`, `cr_scalar(...)`. `Ipu` calls them too.
3. Add `IssueFacts.lr_writes` (from the schedule) and `IssueFacts.mask` (for MULT ops) so the observer no longer recomputes them.
4. `alias_observer` takes `IssueFacts` and `decode_word()` (E04-07), and reads the snapshot through `Cycle.snapshot` (read-only `RegFileReader`).
5. Move the mobilevit fixture to a `CycleObserver` that counts `issued` events, so it also counts in prototyping mode. Delete the `Ipu.dispatch_instruction` delegate if no caller remains.

## Tests

- [ ] New grep gate test: no `ipu._` attribute access and no `from ipu_emu.ipu import _` outside `ipu.py` and its slot modules.
- [ ] New: the mobilevit fixture gives identical counts in faithful and prototyping modes.
- [ ] Output-identical proof: `test_alias_profile.py` (`test_same_cycle_load_not_visible_to_mult`, `test_address_update_overwritten_before_live_read_is_not_a_dependency`), `test_alias_tracking.py` (`test_same_cycle_overwrite_uses_snapshot_then_revokes`), `test_stats.py`; alias profile JSON for three kernels byte-identical before/after.

## Acceptance Criteria

- [ ] `alias_observer.py` imports nothing private from `ipu_emu`.
- [ ] Observers are attached only through `add_observer` methods.
- [ ] Nothing monkeypatches `Ipu` methods in `src/tools/ipu-apps/test/`.
