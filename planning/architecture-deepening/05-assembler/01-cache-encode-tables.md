# Cache per-slot encode tables and enum reverse maps

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-01 · Size S · Depends on: E02 (all-`.asm` binary digests), E01 (benchmark tracking) · Behaviour change: none

## Goal

Stop rebuilding ISA lookup tables for every instruction the assembler encodes. This is a performance-only change with no interface change. It is the first half of the at-least-3x encode-speed target (Q71). E05-02 delivers the rest.

## Background

- `_build_struct_table(slot)` (`ipu_as/inst.py:54-70`) walks `INSTRUCTION_SPEC[slot]` and builds a new dict each time. Every subclass's `struct_by_opcode_table()` (`inst.py:375, 413, 451, 489, 526, 561, 596, 631, 665`) calls it. Callers:
  - `Inst.__init__` (`inst.py:85-89`), once per instruction;
  - `find_inst_type_by_opcode` (`inst.py:182-187`), which loops over all nine subclasses and builds each table per instruction;
  - `_struct_entry` and `decode`.
- `EnumToken.encode` (`ipu_token.py:123-124`) calls `_reverse_map()` (`ipu_token.py:126`), which rebuilds the name-to-value dict on every encode.
- `Opcode.find_opcode_class` (`ipu_common/instruction_spec.py:1331-1338`) is another linear scan over subclasses per opcode token.
- `CompoundInst.get_fields()` (`compound_inst.py:145`) recomputes the field list on every call. The emulator's decode path calls it too.

## Implementation

1. Wrap `_build_struct_table` in `functools.cache`. Return a read-only mapping (`types.MappingProxyType`), so a caller cannot mutate the shared table.
2. Store the reverse map on each `EnumToken` subclass on first use: a class-level dict keyed by `cls`, filled lazily. `encode` becomes a single dict lookup.
3. Replace the subclass scan in `find_inst_type_by_opcode` with a module-level `opcode -> Inst subclass` dict built once. Do the same for `Opcode.find_opcode_class` (or have it delegate to the new dict).
4. Cache `CompoundInst.get_fields()` and `CompoundInst.bits()` with `functools.cache` on the classmethods (they depend only on static ISA data).
5. Add a `bazel run //src/tools/ipu-as-py:bench_assemble` target. It assembles every `.asm` in the E02 digest manifest and prints wall time split into render, parse and encode. Record the numbers in E01's benchmark log (tracked, not gated: Q64).

## Tests

- [ ] E02 `.asm` digest test unchanged (byte-identical `.bin` and `.mem` for every source).
- [ ] New unit test: `struct_by_opcode_table()` returns the same object on two calls, and mutating it raises `TypeError`.
- [ ] New unit test: `EnumToken` encode of every enum member equals the base-commit value (table-driven over `get_all_enum_descriptors()`).
- [ ] `bazel test //...` and `bazel test --define=ipu_proto=1 //...` green.

## Acceptance Criteria

- [ ] No ISA table is rebuilt per instruction on the encode path (verified with a call-count test on `_build_struct_table`).
- [ ] `bench_assemble` exists. The PR description has before and after numbers for the encode stage.
- [ ] Digests unchanged.

## Risks

- A cached mutable table shared across callers would hide accidental mutation. Read-only mappings guard against that.
