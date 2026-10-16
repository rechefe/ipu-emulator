"""The slot registry: every fact about each VLIW slot, once (E03-06).

An instruction word holds one instance of each slot (three of LR).
Each slot has one entry here: how many instances a word holds, its title and
documentation, its wire names and its colour in the layout diagrams. The
old per-module slot tables (``SLOT_COUNT``, ``COMPOUND_LAYOUT_SLOT_ORDER``,
``SLOT_METADATA``, the emulator's field prefixes, the SV enum and struct
names, the assembler's opcode class names, the diagram legend and the debugger
labels) are derived from it.

``field_prefix``, ``opcode_enum`` and ``sv_struct_base`` are wire format: the
emulator keys decoded words by them and the downstream RTL repo names its
types after them. They are written out, not computed from Python class names;
changing one is a breaking change a PR must call out.

Each slot is tagged HARDWARE or EMULATOR (E03-07). In ``LAYOUT_ORDER`` the
EMULATOR slots come after every HARDWARE slot, so the hardware word is the
most significant bits of the emulator word (Q41).

Execution order is not here; E03-14 declares it with the timing contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ipu_isa import schema
from ipu_isa.tags import Tag


class Slot(str, Enum):
    """A slot. ``Slot.LR == "lr"``, so string comparisons keep working.

    A ``str`` mixin rather than ``StrEnum`` (Python 3.11); ``__str__`` returns
    the value, as ``StrEnum`` does.
    """

    BREAK = "break"
    LOAD = "load"
    STORE = "store"
    ACC_STORE = "acc_store"
    MULT = "mult"
    ACC = "acc"
    AAQ = "aaq"
    LR = "lr"
    COND = "cond"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SlotInfo:
    slot: Slot
    count: int  # instances per word: 3 for LR
    pad_to_bits: int | None  # pad the packed slot to this width (wire format), or None
    tag: Tag  # HARDWARE or EMULATOR
    # A program that uses the slot is a debugging build: the assembler warns,
    # the emulator's kernel tests expect DebugOnlyInstructionWarning, and the
    # instruction reference and the editor mark the slot simulation-only.
    # ACC_STORE only. BREAK is EMULATOR too, but a breakpoint is the
    # debugger's own tool and is not flagged (E03-23 and E05-15 revisit this).
    debug_only: bool
    title: str  # "Loop register"
    description: str  # one paragraph, for the editor and the docs
    doc_heading: str  # heading of the slot's section in the instruction reference
    doc_intro: str  # intro of that section
    field_prefix: str  # "lr_inst"; instance i is f"{prefix}_{i}" when count > 1 (wire format)
    opcode_enum: str  # "lr_inst_opcode": SV enum base and decoded opcode suffix (wire format)
    sv_struct_base: str  # "lr_slot" (wire format)
    asm_opcode_class: str  # "LrInstOpcode": the assembler's opcode token class
    legend_colour: str  # "#FFA07A"
    legend_label: str  # the slot's entry in the struct-layout legend

    def instance_labels(self) -> tuple[str, ...]:
        """``("LR0", "LR1", "LR2")`` for LR, ``("MULT",)`` for MULT."""
        name = self.slot.value.upper()
        if self.count == 1:
            return (name,)
        return tuple(f"{name}{index}" for index in range(self.count))

    def field_prefixes(self) -> tuple[str, ...]:
        """Decoded-field prefix of each instance: ``("lr_inst_0", ...)`` or ``("mult_inst",)``."""
        if self.count == 1:
            return (self.field_prefix,)
        return tuple(f"{self.field_prefix}_{index}" for index in range(self.count))


#: The keys each authored entry holds (all required).
ENTRY_KEYS = schema.Keys(
    frozenset(
        {
            "count",
            "pad_to_bits",
            "tag",
            "debug_only",
            "title",
            "description",
            "doc_heading",
            "doc_intro",
            "field_prefix",
            "opcode_enum",
            "sv_struct_base",
            "asm_opcode_class",
            "legend_colour",
            "legend_label",
        }
    )
)

_SPEC: dict[Slot, dict] = {
    Slot.BREAK: {
        "count": 1,
        "pad_to_bits": None,
        "tag": Tag.EMULATOR,
        "debug_only": False,
        "title": "Break",
        "description": "Debug break: halts execution or enters debug mode.",
        "doc_heading": "Break Instructions",
        "doc_intro": "Debug break instructions for halting execution and entering debug mode.",
        "field_prefix": "break_inst",
        "opcode_enum": "break_inst_opcode",
        "sv_struct_base": "break_slot",
        "asm_opcode_class": "BreakInstOpcode",
        "legend_colour": "#FFD93D",
        "legend_label": "BreakInst (Break / Debug)",
    },
    Slot.LOAD: {
        "count": 1,
        "pad_to_bits": None,
        "tag": Tag.HARDWARE,
        "debug_only": False,
        "title": "Load",
        "description": (
            "First-stage memory loads that feed the multiply unit (mult-stage, cyclic, "
            "and mask registers)."
        ),
        "doc_heading": "LOAD Instructions",
        "doc_intro": (
            "First-stage memory loads that feed the multiply unit "
            "(mult-stage registers, cyclic register, and mask register)."
        ),
        "field_prefix": "load_inst",
        "opcode_enum": "load_inst_opcode",
        "sv_struct_base": "load_slot",
        "asm_opcode_class": "LoadInstOpcode",
        "legend_colour": "#FF6B6B",
        "legend_label": "LoadInst (Memory Load)",
    },
    Slot.STORE: {
        "count": 1,
        "pad_to_bits": None,
        "tag": Tag.HARDWARE,
        "debug_only": False,
        "title": "Store",
        "description": (
            "Last-stage memory stores that drain POST_AAQ_REG to external memory after "
            "activation and quantization."
        ),
        "doc_heading": "STORE Instructions",
        "doc_intro": (
            "Last-stage memory stores that drain **POST_AAQ_REG** to external "
            "memory after activation and quantization."
        ),
        "field_prefix": "store_inst",
        "opcode_enum": "store_inst_opcode",
        "sv_struct_base": "store_slot",
        "asm_opcode_class": "StoreInstOpcode",
        "legend_colour": "#E74C3C",
        "legend_label": "StoreInst (Memory Store)",
    },
    Slot.ACC_STORE: {
        "count": 1,
        "pad_to_bits": None,
        "tag": Tag.EMULATOR,
        "debug_only": True,
        "title": "Accumulator store",
        "description": (
            "Simulation-only stores of R_ACC to external memory (STR_ACC_REG). Not "
            "implemented in real IPU hardware."
        ),
        "doc_heading": "ACC_STORE Instructions",
        "doc_intro": (
            "Simulation-only slot for storing **R_ACC** to external memory "
            "(`STR_ACC_REG`). Not implemented in real IPU hardware."
        ),
        "field_prefix": "acc_store_inst",
        "opcode_enum": "acc_store_inst_opcode",
        "sv_struct_base": "acc_store_slot",
        "asm_opcode_class": "AccStoreInstOpcode",
        "legend_colour": "#C0392B",
        "legend_label": "AccStoreInst (STR_ACC_REG, simulation-only)",
    },
    Slot.MULT: {
        "count": 1,
        "pad_to_bits": None,
        "tag": Tag.HARDWARE,
        "debug_only": False,
        "title": "Multiply",
        "description": "Multiply-stage instructions: element-wise and vector multiply operations.",
        "doc_heading": "MULT Instructions",
        "doc_intro": (
            "Multiplication instructions for element-wise and element-vector operations.\n"
            "The multiplication result (`mult_result`) is forwarded to the ACC stage in the "
            "CPU and not stored in any register in the way.\n"
        ),
        "field_prefix": "mult_inst",
        "opcode_enum": "mult_inst_opcode",
        "sv_struct_base": "mult_slot",
        "asm_opcode_class": "MultInstOpcode",
        "legend_colour": "#4ECDC4",
        "legend_label": "MultInst (Multiply)",
    },
    Slot.ACC: {
        "count": 1,
        "pad_to_bits": None,
        "tag": Tag.HARDWARE,
        "debug_only": False,
        "title": "Accumulator",
        "description": "Accumulation instructions for combining multiply results into R_ACC.",
        "doc_heading": "ACC Instructions",
        "doc_intro": (
            "Accumulation instructions for combining values with optional masking and shifting."
        ),
        "field_prefix": "acc_inst",
        "opcode_enum": "acc_inst_opcode",
        "sv_struct_base": "acc_slot",
        "asm_opcode_class": "AccInstOpcode",
        "legend_colour": "#45B7D1",
        "legend_label": "AccInst (Accumulator)",
    },
    Slot.AAQ: {
        "count": 1,
        "pad_to_bits": None,
        "tag": Tag.HARDWARE,
        "debug_only": False,
        "title": "Activation and quantization",
        "description": (
            "Activation and quantization: apply activation to R_ACC and write quantized "
            "INT8 bytes to POST_AAQ_REG."
        ),
        "doc_heading": "AAQ Instructions",
        "doc_intro": (
            "Activation and quantization: apply activation to R_ACC elements and write "
            "quantized INT8 bytes to POST_AAQ_REG."
        ),
        "field_prefix": "aaq_inst",
        "opcode_enum": "aaq_inst_opcode",
        "sv_struct_base": "aaq_slot",
        "asm_opcode_class": "AaqInstOpcode",
        "legend_colour": "#9B59B6",
        "legend_label": "AaqInst (Activation and Quantization)",
    },
    Slot.LR: {
        "count": 3,
        "pad_to_bits": 20,
        "tag": Tag.HARDWARE,
        "debug_only": False,
        "title": "Loop register",
        "description": (
            "Loop register instructions for controlling addresses, counters, and scalars. "
            "Three independent sub-slots per cycle."
        ),
        "doc_heading": "LR Instructions",
        "doc_intro": (
            "Loop register manipulation instructions for controlling loop counters and addresses."
        ),
        "field_prefix": "lr_inst",
        "opcode_enum": "lr_inst_opcode",
        "sv_struct_base": "lr_slot",
        "asm_opcode_class": "LrInstOpcode",
        "legend_colour": "#FFA07A",
        "legend_label": "LrInst (Link Register)",
    },
    Slot.COND: {
        "count": 1,
        "pad_to_bits": None,
        "tag": Tag.HARDWARE,
        "debug_only": False,
        "title": "Conditional",
        "description": (
            "Conditional and unconditional branches; control flow based on register comparisons."
        ),
        "doc_heading": "Conditional Branch Instructions",
        "doc_intro": (
            "Control flow instructions for branching based on conditions or unconditionally."
        ),
        "field_prefix": "cond_inst",
        "opcode_enum": "cond_inst_opcode",
        "sv_struct_base": "cond_slot",
        "asm_opcode_class": "CondInstOpcode",
        "legend_colour": "#98D8C8",
        "legend_label": "CondInst (Conditional)",
    },
}

#: Slots in the instruction word, most significant first.
LAYOUT_ORDER: tuple[Slot, ...] = (
    Slot.COND,
    Slot.LR,
    Slot.LOAD,
    Slot.MULT,
    Slot.ACC,
    Slot.AAQ,
    Slot.STORE,
    Slot.ACC_STORE,
    Slot.BREAK,
)


def _load(spec: Mapping[Slot, dict], layout_order: tuple[Slot, ...]) -> Mapping[Slot, SlotInfo]:
    """Check the authored entries and freeze them into records."""
    if set(spec) != set(Slot):
        raise schema.SpecError(f"SLOTS: entries {sorted(spec)} do not match Slot {sorted(Slot)}")
    if sorted(layout_order) != sorted(Slot):
        raise schema.SpecError("LAYOUT_ORDER: must list every slot exactly once")
    records = {}
    for slot, entry in spec.items():
        path = f"SLOTS.{slot.value}"
        schema.check_keys(path, entry, ENTRY_KEYS)
        if not isinstance(entry["count"], int) or entry["count"] < 1:
            raise schema.SpecError(f"{path}.count: must be a positive int")
        if not isinstance(entry["tag"], Tag) or entry["tag"] is Tag.ASSEMBLER:
            raise schema.SpecError(f"{path}.tag: a slot is Tag.HARDWARE or Tag.EMULATOR")
        if not isinstance(entry["debug_only"], bool):
            raise schema.SpecError(f"{path}.debug_only: must be a bool")
        if entry["debug_only"] and entry["tag"] is not Tag.EMULATOR:
            raise schema.SpecError(f"{path}.debug_only: only an EMULATOR slot can be debug-only")
        records[slot] = SlotInfo(slot=slot, **entry)
    tags = [records[slot].tag for slot in layout_order]
    if Tag.EMULATOR in tags and Tag.HARDWARE in tags[tags.index(Tag.EMULATOR) :]:
        late = layout_order[len(tags) - 1 - tags[::-1].index(Tag.HARDWARE)]
        raise schema.SpecError(
            f"LAYOUT_ORDER: HARDWARE slot {late.value!r} follows an EMULATOR slot; "
            "the hardware slots must be the most significant (Q41)"
        )
    for key in ("field_prefix", "opcode_enum", "sv_struct_base", "asm_opcode_class"):
        schema.check_unique("SLOTS", key, (getattr(info, key) for info in records.values()))
    return MappingProxyType(records)


#: Every slot, in authoring order.
SLOTS: Mapping[Slot, SlotInfo] = _load(_SPEC, LAYOUT_ORDER)
