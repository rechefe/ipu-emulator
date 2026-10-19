# `RegFile` read view and write view

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-04 · Size S · Depends on: E04-01 · Behaviour change: none

## Goal

Split `RegFile.raw()` into a read view and a write view so that the interface states intent. Reads have no side effects; writes report exactly the range they touch. This is the seam that E04-10 (packed resolver), E04-12 (cycle view) and E04-29 (commit record register writes) build on.

## Background

- `RegFile.raw(name)` (regfile.py:101-109) fires the profile observer for the whole register, pops `_constant_one_bytes[name]`, and adds the name to `_mutable_exports` for the life of the object. It does this on every call, including pure reads.
- `raw_readonly` (regfile.py:111) copies through `bytes(...)`.
- `ipu.py` has 36 `.raw(` calls. Many are reads:
  - `self.state.regfile.raw("r_mask")[slot:slot + 16]` in the prototyping mask shortcut (ipu.py:909);
  - `self._snapshot_buffer.raw("r_acc")` (ipu.py:1421, 1448, 1476, 1668, 1701) and `raw("mult_res")` (ipu.py:1603).
- Each of those reads drops constant-ones provenance. Fixing that changes alias/identity statistics, so it is a separate behaviour issue (E04-24). This issue keeps today's provenance results exactly.

## Implementation

1. Add to `RegFile`:
   ```python
   def view(self, name: str) -> memoryview            # read-only; no observer, no provenance change
   def mutable(self, name: str, start: int = 0, size: int | None = None) -> memoryview
       # fires the observer for [start, start+size) and forgets provenance for that range;
       # does not add to _mutable_exports
   def forget_provenance(self, name: str) -> None      # explicit; used only to keep E04-04 output-identical
   ```
   `raw()` stays, reimplemented on these, and emits `DeprecationWarning`.
2. Convert `ipu.py` call sites mechanically, one commit per handler group:
   - pure reads → `view(name)`;
   - destinations (`mult_res` in MULT, `r_acc` in ACC/AGG, `post_aaq_reg`) → `mutable(name, start, size)`;
   - at the read sites that drop provenance today (ipu.py:909, 1421, 1448, 1476, 1603, 1668, 1701) add `forget_provenance(name)` with a `# E04-24` comment. This keeps alias/identity counts identical.
3. `get_<name>_words` returns `np.frombuffer(view)` (read-only). Add `set_<name>_words(arr)` for writers.
4. Add `RegFile.descriptor(name)` as the public replacement for `_desc` (test_debug_tui.py:2943).

## Tests

- [ ] New: `view()` does not call the observer and does not clear `has_constant_ones`.
- [ ] New: writing through `view()` raises `TypeError`.
- [ ] New: `mutable(name, s, n)` reports `(name, s, n)` to the observer and forgets provenance only for `[s, s+n)`.
- [ ] Output-identical proof: `test_alias_profile.py`, `test_alias_tracking.py` and `test_stats.py` pass with unchanged counts; `test_prototyping_equivalence.py` passes under `--define=ipu_proto=1`; E02 kernel output digests unchanged.

## Acceptance Criteria

- [ ] No `.raw(` call remains in `ipu.py`.
- [ ] Every remaining `forget_provenance` call carries `# E04-24`.
- [ ] Alias and identity statistics are unchanged on the full suite.
