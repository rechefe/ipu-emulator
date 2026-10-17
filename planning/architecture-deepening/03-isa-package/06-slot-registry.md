# Slot registry replaces five parallel slot tables

Part of [Epic E03: ISA package](00-epic.md) · ID E03-06 · Size M · Depends on: E03-02, E03-03 · Behaviour change: none · `*_WIDTH` localparams: unchanged

## Goal

One record per slot holds everything known about it: how many instances a word has, its layout position, its wire names and its documentation. Adding a slot then means one registry entry, its instructions, the emulator handler module and nothing else in the ISA package.

## Background

Slot knowledge is spread over these tables (verified at HEAD 694a8a8):

| Table | Location | Knows |
|---|---|---|
| `SLOT_COUNT`, `COMPOUND_LAYOUT_SLOT_ORDER`, `SLOT_METADATA`, `is_hardware_slot` | `instruction_spec.py:145-189` | count, layout order, description, hardware flag |
| `_SLOT_FIELD_PREFIX` | `ipu_emu/ipu.py:233-244` | decoded-field prefix (`lr_inst_<i>` special-cased) |
| `_SLOT_META` | `ipu_as/gen_codegen.py:99-109` | SV opcode enum name and struct base name |
| `slot_to_class_name` | `create_assembler_opcodes` (moved to `ipu_as/opcodes.py` by E03-02) | assembler opcode class name |
| `_slot_to_inst`, legend colours and titles | `ipu_as/compound_inst.py:94-104`, `:208-217` | `Inst` subclass, SVG legend |
| `SLOT_LABELS`, `PIPELINE_STAGES` | `ipu_emu/debug_pipeline.py:6-11`, `:58-59` | `LR0`/`LR1`/`LR2` labels, stage grouping |

There is no slot type; slots are strings, compared by value in many places.

## Implementation

1. Author `SLOTS` as a dict in `ipu_isa/slots.py` (Q30), validated by the schema code and loaded into frozen records:

   ```python
   class Slot(StrEnum):          # Slot.LR == "lr", so existing string compares keep working (L34)
       COND = "cond"; LR = "lr"; LOAD = "load"; MULT = "mult"; ACC = "acc"
       AAQ = "aaq"; STORE = "store"; ACC_STORE = "acc_store"; BREAK = "break"

   @dataclass(frozen=True, slots=True)
   class SlotInfo:
       slot: Slot
       count: int                  # 3 for LR
       title: str                  # "Loop register"
       description: str            # was SLOT_METADATA[...]["description"]
       doc_intro: str              # was the per-Inst-subclass heading/intro in ipu_as/inst.py (E05-02 needs it)
       field_prefix: str           # "lr_inst"; instances become f"{prefix}_{i}" when count > 1  (wire format)
       opcode_enum: str            # "lr_inst_opcode"   (SV enum base and decoded opcode suffix; wire format)
       sv_struct_base: str         # "lr_slot"          (wire format)
       asm_opcode_class: str       # "LrInstOpcode"
       legend_colour: str          # "#FF6B6B"
       def instance_labels(self) -> tuple[str, ...]: ...   # ("LR0", "LR1", "LR2"), ("MULT",)

   SLOTS: Mapping[Slot, SlotInfo]
   LAYOUT_ORDER: tuple[Slot, ...]   # MSB -> LSB, was COMPOUND_LAYOUT_SLOT_ORDER
   ```
   Wire names are written out, not computed from Python class names. Execution order is not here; E03-14 adds it next to the timing contract. Hardware/emulator tags are added by E03-07.
2. `SLOT_COUNT`, `COMPOUND_LAYOUT_SLOT_ORDER` and `SLOT_METADATA` become derived views over `SLOTS` (same values), so callers migrate one at a time (L34).
3. Replace the copies: `ipu.py` `_SLOT_FIELD_PREFIX`; `gen_codegen._SLOT_META`; `slot_to_class_name` in `ipu_as/opcodes.py`; the legend in `compound_inst.py`; `SLOT_LABELS` and the LR labels in `debug_pipeline.py`. `compound_inst._slot_to_inst` stays (E05-02 deletes it) but is keyed by `Slot` and checked total against `SLOTS` at import.
4. Schema rule: `INSTRUCTION_SPEC` keys equal `SLOTS` keys; `LAYOUT_ORDER` lists every slot exactly once.

## Tests

- [ ] `ipu-isa/test/test_slots.py`: layout order covers every slot once; `instance_labels(Slot.LR) == ("LR0", "LR1", "LR2")`; `field_prefix`, `opcode_enum` and `sv_struct_base` values are unique; derived `SLOT_COUNT`/`COMPOUND_LAYOUT_SLOT_ORDER` equal today's literals.
- [ ] Totality: a `_slot_to_inst` copy missing one slot raises at import.
- [ ] E02 baselines unchanged: SV golden, decode snapshots (field names, emulator field maps), generated artifacts, disassembly.

## Acceptance Criteria

- [ ] Every slot fact is written once, in `SLOTS`.
- [ ] No module outside `ipu_isa` holds a slot-name table.

## Risks

- `StrEnum` equality with plain strings keeps today's comparisons working; new code uses `Slot` members. Low risk.
