# Read-only typed ISA view from one loader

Part of [Epic E03: ISA package](00-epic.md) · ID E03-09 · Size M · Depends on: E03-03, E03-05, E03-06, E03-07, E03-08 · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

Authors keep writing nested dicts (Q30). Consumers stop walking them by string keys. One loader validates the dicts (E03-03 rules) and builds a frozen, typed view; every consumer reads that view. This is the interface of the ISA module: small, typed, immutable, with all the knowledge (opcodes, layout, tags, derived syntax) behind it.

## Background

- Consumers index the dicts directly with string keys: `INSTRUCTION_SPEC[slot][name]["operands"]`, `op.get("read")`, `SLOT_UNIONS[slot].opcode_bindings`. Occurrences outside the ISA package: `ipu_emu/ipu.py` (11), `ipu_as/inst.py` (8), `lark_tree.py` (7), `gen_vscode.py` (6), `gen_codegen.py` (5), `gen_docs.py` (3), `compound_inst.py` (2), `opcodes.py` (2), `debug_pipeline.py` (2), the two MobileViT measurement fixtures (3 each), plus tests.
- Lookup helpers exist but return dicts: `get_instruction`, `get_instruction_by_opcode`, `get_operand_names_and_types` (`instruction_spec.py:1262-1307`), `find_pseudo_instruction` (`:1600-1612`).
- `InstructionDoc` is a mutable dataclass (`instruction_spec.py:111-120`).

## Implementation

1. Add `ipu_isa/view.py`:
   ```python
   class Read(StrEnum): CYCLE_START = "snapshot"; FORWARDED = "live"   # spellings renamed in E03-14

   @dataclass(frozen=True, slots=True)
   class Operand:
       name: str                    # becomes SV field <name>_<column>: wire format
       type: OperandType
       read: Read | None

   @dataclass(frozen=True, slots=True)
   class Instruction:
       slot: Slot
       mnemonic: str
       opcode: int                  # position in the slot; never written by hand
       operands: tuple[Operand, ...]
       tag: Tag
       doc: InstructionDoc          # frozen
       execute_fn: str
       @property
       def syntax(self) -> str      # E03-04's derivation
       def operand(self, name: str) -> Operand

   @dataclass(frozen=True, slots=True)
   class PseudoInstruction:
       mnemonic: str; operands: tuple[Operand, ...]; expands_to: Instruction
       args: tuple[str, ...]; doc: InstructionDoc; tag: Tag   # always ASSEMBLER

   @dataclass(frozen=True)
   class Isa:
       slots: Mapping[Slot, SlotInfo]
       operand_types: Mapping[str, OperandType]
       layout: IsaLayout
       def instructions(self, slot: Slot) -> tuple[Instruction, ...]
       def instruction(self, slot: Slot, mnemonic: str) -> Instruction   # case-insensitive
       def by_opcode(self, slot: Slot, opcode: int) -> Instruction
       def find_slot(self, mnemonic: str) -> Slot
       def pseudo(self, mnemonic: str, arity: int) -> PseudoInstruction | None

   def load_isa(instruction_spec, pseudo_spec, slots, operand_types) -> Isa   # validates, then builds
   @functools.cache
   def isa() -> Isa                                                           # the repository's ISA
   ```
   Later issues add fields to these records as they add keys to the schema: implicit reads and writes (E03-14), formats and faults (E03-13, E03-15).
2. `load_isa` is the only code that reads the authoring dicts. The old helpers (`get_instruction`, `get_instruction_by_opcode`, `get_operand_names_and_types`, `find_pseudo_instruction`) become thin functions over `isa()` with the same return shape, so callers migrate one at a time (L34). E03-10 and E03-11 migrate the callers; E03-11 deletes the helpers.
3. `test/fixtures/spec_snapshot.json`: a JSON dump of every `Instruction` record (slot, mnemonic, opcode, operand names/types/read, tag, execute_fn), captured from the dicts once in this PR. The test asserts `isa()` reproduces it, which proves the loader loses nothing.

## Tests

- [ ] Record round trip against `spec_snapshot.json`.
- [ ] `load_isa` on a dict copy with a planted schema violation raises `SpecError`; on a valid copy with one extra instruction returns a view that contains it (used by E03-24).
- [ ] Records are immutable (`dataclasses.FrozenInstanceError`; mappings are `MappingProxyType`).
- [ ] E02 baselines unchanged.

## Acceptance Criteria

- [ ] One loader builds the view; nothing else in `ipu_isa` reads the authoring dicts after load.
- [ ] Every ISA fact a consumer needs is reachable from `isa()` without string-key dict access.
- [ ] `load_isa` takes the dicts as arguments, so tests can build a modified ISA in-process.
