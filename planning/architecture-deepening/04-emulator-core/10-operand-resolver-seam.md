# One operand resolver interface, two adapters

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-10 · Size S · Depends on: E04-04, E04-08, E04-09 · Behaviour change: none

## Goal

Turn the two hand-written versions of operand resolution into one interface with two adapters: a faithful resolver that uses `RegFile`'s public accessors, and a packed resolver (prototyping) that unpacks LR/CR banks in one call. With two adapters the seam is real (D7: prototyping lives only at adapters).

## Background

- `_resolve_reads` (ipu.py:2044-2066) resolves each read operand through `_resolve_operand` (ipu.py:795-833).
- `_resolve_reads_prototype` (ipu.py:2068-2123) is a second version. It unpacks the LR, CR and LCR banks three times, reaches into `RegFile._storage`, and carries its own copy of the mult-stage operand check (ipu.py:2113-2116 duplicates ipu.py:823-826).

## Implementation

```python
class OperandResolver(Protocol):
    def resolve(self, plan: Plan, encoded: Mapping[str, int], snapshot: RegFileReader) -> dict[str, Any]

class FaithfulResolver: ...   # RegFile public accessors, one operand at a time
class PackedResolver: ...     # struct.unpack_from over RegFile.packed_bank(name) views
```

1. Add `RegFile.packed_bank(name) -> memoryview` (read-only, built on E04-04's `view`) so nothing outside `regfile.py` touches `_storage`.
2. Move the mult-stage operand check into one function both adapters call. (It becomes an F-1 check in E04-19.)
3. `Issuer` takes a resolver at construction; `Cycle` picks `PackedResolver` when prototyping is on.
4. Delete `_resolve_reads_prototype`.

## Tests

- [ ] New property test: for every plan and 200 random encodings per plan (including the wide `MultStageReg` case), both resolvers return equal operand dicts against the same register file.
- [ ] Output-identical proof: `test_prototyping_equivalence.py` under `--define=ipu_proto=1`; E02 kernel output digests in both modes.

## Acceptance Criteria

- [ ] `grep -n "_storage" src/ipu_emu/` finds matches only in `regfile.py`.
- [ ] One mult-stage operand check.
- [ ] Resolution exists in exactly two adapters behind one protocol.
