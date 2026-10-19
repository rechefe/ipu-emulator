# Paged sparse XMEM with ranged events and registered observers

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-05 · Size M · Depends on: E04-01 · Behaviour change: none

## Goal

Keep the 512 MiB XMEM address space (L2) but allocate only pages that are written. Report observer events for the bytes actually touched (L6). Allow more than one observer. Target (Q71): `IpuState()` under 5 ms and 5 MB.

## Background

- `XMem.__init__` allocates `bytearray(XMEM_SIZE_BYTES)` (xmem.py:37), 512 MiB per state.
- `clear()` (xmem.py:111-116) reports `("write", 0, self._data)` to the observer and then builds `b"\x00" * 512 MiB`.
- `__setitem__` (xmem.py:121-126) reports the whole buffer for any write, and `ExecutionObserver.memory` then walks every 128-byte block.
- Observers are attached by poking a private attribute from outside: `state.xmem._profile_observer = self.memory` and `state.regfile._profile_observer = ...` (alias_observer.py:44-45). Only one observer can be attached.
- `XMEM_SIZE_BYTES` is imported by `ipu.py` and by apps (`pooling/app.py`, `kernel_registry/memory.py`, `sampling/test.py`); it must stay.
- Only a test reads `xmem._data` (test_prototyping_equivalence.py:115).

## Implementation

Each step is one commit; the PR may be split at step 3.

1. **Ranged events.** Change the observer signature to `(kind, address, size, data)` with `kind in {"read", "write", "clear"}`. `clear()` sends one `("clear", 0, XMEM_SIZE_BYTES, None)`; `ExecutionObserver.memory` handles `clear` by dropping its store/version tables instead of looping over blocks. `__setitem__` accepts an int or a contiguous slice only (strided slices raise `TypeError`) and reports its real range.
2. **Registered observers.** `XMem.add_observer(fn)` and `RegFile.add_observer(fn)` return a remover. `ExecutionObserver` uses them. `_profile_observer` is removed (no shim needed; D5 keeps only the reference API stable).
3. **Paged storage.** `PAGE_BYTES = 1 << 16` (a multiple of the 512-byte wide row, so a row access never crosses a page). Storage is `dict[int, bytearray]`; a missing page reads as zeros. Reads and writes that cross a page boundary are split internally. Add `XMem.__eq__` (page-wise; absent page equals zero page), `touched_pages()` and `snapshot_bytes(address, size)` (no observer event; for debuggers). Full-range slicing raises and points to `touched_pages()`.
4. Change test_prototyping_equivalence.py:115 to `state_a.xmem == state_b.xmem`.

```python
class XMem:
    def read_address(self, address: int, size: int) -> bytearray
    def write_address(self, address: int, data: bytes | bytearray) -> None
    def fill(self, address: int, size: int, value: int = 0) -> None
    def clear(self) -> None
    def add_observer(self, fn: MemoryObserver) -> Callable[[], None]
    def touched_pages(self) -> Iterator[tuple[int, memoryview]]
    def snapshot_bytes(self, address: int, size: int) -> bytes
```

## Tests

- [ ] Existing `test_xmem.py` passes unchanged (bounds errors keep their `ValueError` and message until E04-19).
- [ ] New: an untouched address reads as zeros; a write across a page boundary reads back; `clear()` drops every page; `__eq__` treats absent and zero pages as equal; a strided `__setitem__` raises; two observers both receive events; the remover detaches; a 1-byte write produces exactly one event of size 1.
- [ ] New in `test_alias_profile.py`: `clear()` under a profiler is O(1) (the observer's version table stays small).
- [ ] Output-identical proof: E02 kernel output digests unchanged; `test_prototyping_equivalence.py` under `--define=ipu_proto=1`; alias profile outputs unchanged.
- [ ] E04-02 benchmark: before/after `IpuState()` time and RSS, and the emulator suite wall time, pasted in the PR.

## Acceptance Criteria

- [ ] `IpuState()` < 5 ms and < 5 MB RSS delta (tracked; record the number).
- [ ] No observer event ever covers more bytes than the operation touched.
- [ ] No code outside `xmem.py` and `regfile.py` sets an observer attribute.

## Risks

- A caller that materialises the full buffer (`bytes(state.xmem[:])`) would allocate 512 MiB. A grep of `src/` shows none; full-range slicing now raises.
- The MMIO plan (`riscv-host-integration/02-emulator-mmio-model.md`) wants reset to clear XMEM; this makes that cheap.
