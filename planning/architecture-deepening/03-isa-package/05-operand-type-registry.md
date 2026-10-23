# Operand-type registry replaces six parallel tables

Part of [Epic E03: ISA package](00-epic.md) · ID E03-05 · Size M · Depends on: E03-02, E03-03 · Behaviour change: none · `*_WIDTH` localparams: unchanged (typedef names and widths reproduced from the same inputs; E02-03 golden proves it)

## Goal

Make one record per operand type the only place that knows its width, its value domain, its readability, its wire names and its documentation. Adding an operand type then means one registry entry in `ipu_isa` plus one token class in `ipu_as` (and an emulator resolver if it is readable). A missing piece fails at import with a message naming it.

## Background

Six tables must agree by hand today:

| Table | Location | Knows |
|---|---|---|
| `VALID_OPERAND_TYPES` | `instruction_spec.py:1415-1435` | the set of names |
| `get_operand_type_bits()` | `union_layout.py:61-92` | widths |
| `OPERAND_TYPE_MAP` | `ipu_as/inst.py:33-51` | token class |
| `_TYPE_FIELD_SUFFIX` | `ipu_emu/ipu.py:213-230` | decoded-field suffix (= snake case of the token class name) |
| `_OPERAND_TYPE_TO_SV_TYPEDEF` | `ipu_as/gen_codegen.py:31-42` | SV typedef |
| `OPERAND_TYPE_DETAILS` | `ipu_as/gen_docs.py:19+` | long-form docs, also used by `gen_vscode.py:474` |

Drift already present: `_TYPE_FIELD_SUFFIX` lacks `LrOrReshapeMaskImmediate`; `BreakImmediate` is 16 bits in both `union_layout.py:86` and `ipu_as/immediate.py:75`; `Label` is the literal 10 (`union_layout.py:87`) restating `MAX_PROGRAM_SIZE = 1024` (`ipu_as/ipu_token.py:6`); the readable set lives only in E03-03's rule.

## Implementation

1. Author the registry as a dict in `ipu_isa/operand_types.py` (Q30), checked by E03-03's schema code with its own key set, and loaded into frozen records:

   ```python
   class OperandKind(Enum): REGISTER; ENUM; IMMEDIATE; LABEL

   @dataclass(frozen=True, slots=True)
   class OperandType:
       name: str                 # "LrIdx" — spec key, appears as an SV comment
       kind: OperandKind
       bits: int | None          # None: width decided by union packing (LrIncDecImmediate, LrOrReshapeMaskImmediate)
       domain: Domain            # register bank + count, enum member names, or (min, max) range
       readable: bool            # True only for LrIdx, CrIdx, LcrIdx, MultStageReg, LrdIdx
       token_class: str          # "LrRegField" — names the ipu_as token class
       field_suffix: str         # "lr_reg_field" — decoded-word field suffix (wire format, written out, not computed)
       sv_typedef: str | None    # "lr_reg_field_t" for REGISTER/ENUM, else None
       summary: str              # one line
       doc: str                  # long form, moved from gen_docs.OPERAND_TYPE_DETAILS

   OPERAND_TYPES: Mapping[str, OperandType]   # MappingProxyType, insertion order kept
   def operand_type(name: str) -> OperandType  # KeyError lists valid names
   ```
   Register widths are computed from `REGISTER_DEFINITIONS`; enum widths from the member count; `Label` from `(MAX_PROGRAM_SIZE - 1).bit_length()` with `MAX_PROGRAM_SIZE` moved to `ipu_isa/geometry.py` (E03-12 adds the rest of the geometry there). `field_suffix` and `sv_typedef` are written out as strings, so renaming a Python class can never change a wire name.
2. Derive the old tables from the registry and delete the copies:
   - `VALID_OPERAND_TYPES = frozenset(OPERAND_TYPES)`;
   - `get_operand_type_bits()` reads `OperandType.bits` (DERIVED stays 0 until E03-08 removes the placeholder);
   - `ipu.py` `_TYPE_FIELD_SUFFIX` becomes `{t.name: t.field_suffix for t in OPERAND_TYPES.values()}`;
   - `gen_codegen._OPERAND_TYPE_TO_SV_TYPEDEF` and `_check_sv_typedef_map` go; `_sv_logic_type` reads `sv_typedef`;
   - `gen_docs.OPERAND_TYPE_DETAILS` goes; `gen_docs` and `gen_vscode` read `doc`;
   - `BreakImmediateType.bits()`, `AddbiImmediate.bits()` and `LabelToken.bits()` return `operand_type(...).bits`; `ipu_token.MAX_PROGRAM_SIZE` imports from `ipu_isa.geometry`;
   - E03-03's readable set becomes `{t.name for t in OPERAND_TYPES.values() if t.readable}`.
3. `OPERAND_TYPE_MAP` stays in `ipu_as/inst.py` as the adapter from ISA names to token classes, with an import-time check: its keys equal `OPERAND_TYPES`, and each class's `__name__` equals `token_class`. Each enum token's member list equals the registry `domain`.

## Tests

- [ ] `ipu-isa/test/test_operand_types.py`: every type has `bits` or is derived; `sv_typedef` is set exactly for REGISTER and ENUM kinds; `field_suffix` values are unique; `readable` set equals today's E03-03 set.
- [ ] `ipu-as-py` totality test: removing one entry from `OPERAND_TYPE_MAP` in a copy raises with the type name; a class whose `__name__` differs from `token_class` raises.
- [ ] E02 baselines unchanged: SV golden, decode snapshots (field names, emulator field maps), generated artifacts (`operand-types.md`, `isa.json`), asm digests.

## Acceptance Criteria

- [ ] Each operand-type fact (width, domain, readability, wire names, docs) is written exactly once.
- [ ] Adding an operand type needs one registry entry and one token class; a missing piece fails at import with a message naming it.

## Risks

- `token_class`, `field_suffix` and `sv_typedef` are wire format (emulator dict keys, SV typedef names). The registry makes that explicit; document in the module that changing one is a breaking change that must be called out like a `*_WIDTH` change.
- Coordinate with E05-03 (explicit field-name table): it reads `field_suffix` from this registry instead of a second table.
