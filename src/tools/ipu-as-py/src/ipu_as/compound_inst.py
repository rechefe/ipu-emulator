import functools
import warnings

import ipu_as.inst as inst
import ipu_as.utils as utils
from ipu_as.errors import SlotError
from ipu_isa.codec.word import WORD_LAYOUT
from ipu_isa.schema import SpecError
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot

INST_SEPARATOR = ";"


#: The assembler's instruction class of each slot. E05-02 replaces the
#: classes with the slot codec and deletes this table.
SLOT_TO_INST: dict[Slot, type[inst.Inst]] = {
    Slot.COND: inst.CondInst,
    Slot.LR: inst.LrInst,
    Slot.LOAD: inst.LoadInst,
    Slot.STORE: inst.StoreInst,
    Slot.ACC_STORE: inst.AccStoreInst,
    Slot.MULT: inst.MultInst,
    Slot.ACC: inst.AccInst,
    Slot.AAQ: inst.AaqInst,
    Slot.BREAK: inst.BreakInst,
}


def check_slot_to_inst(slot_to_inst: dict) -> None:
    """Raise ``SpecError`` unless ``slot_to_inst`` maps every registered slot,
    and nothing else, to the instruction class of that slot."""
    missing = sorted(slot.value for slot in SLOTS if slot not in slot_to_inst)
    extra = sorted(str(key) for key in slot_to_inst if key not in SLOTS)
    if missing or extra:
        raise SpecError(
            f"SLOT_TO_INST must cover exactly the registered slots; "
            f"missing {missing}, unknown {extra}"
        )
    for slot, inst_class in slot_to_inst.items():
        if inst_class._slot_type_name() != slot.value:
            raise SpecError(
                f"SLOT_TO_INST[{slot.value}] is {inst_class.__name__}, "
                f"a {inst_class._slot_type_name()} class"
            )


check_slot_to_inst(SLOT_TO_INST)


class CompoundInst:
    def __init__(
        self,
        instructions: list[dict[str, any]],
    ):
        # instructions is now a list ordered by instruction_types()
        self.instructions = [None] * len(self.instruction_types())
        self._fill_out_nop(self._fill_instructions(instructions))

    def _fill_instructions(self, instructions: list[dict[str, any]]) -> int:
        address = None
        inst_types_list = self.instruction_types()

        for instruction in instructions["instructions"]:
            opcode_str = instruction["opcode"].token.value
            if address is None:
                address = instruction["opcode"].instr_id

            # NOP is context-aware: fill the next available unfilled slot.
            if opcode_str.lower() == "nop":
                slot_filled = False
                for i, expected_type in enumerate(inst_types_list):
                    if self.instructions[i] is None:
                        self.instructions[i] = expected_type(instruction)
                        slot_filled = True
                        break
                if not slot_filled:
                    raise SlotError(
                        f"NOP: all slots are already filled\n"
                        f"At: {instruction['opcode'].get_location_string()}",
                        location=instruction["opcode"].location(),
                        token=opcode_str,
                    )
                continue

            inst_type = inst.Inst.find_inst_type_by_opcode(
                opcode_str, instruction["opcode"].location()
            )

            # Find the first available slot for this instruction type
            slot_filled = False
            for i, expected_type in enumerate(inst_types_list):
                if expected_type == inst_type and self.instructions[i] is None:
                    self.instructions[i] = inst_type(instruction)
                    slot_filled = True
                    break

            if not slot_filled:
                # No available slot - either all slots are filled or no slot exists
                available_slots = sum(1 for t in inst_types_list if t == inst_type)

                if available_slots == 0:
                    raise SlotError(
                        f"Instruction type {inst_type.__name__} is not allowed in compound instruction\n"
                        f"At: {instruction['opcode'].get_location_string()}",
                        location=instruction["opcode"].location(),
                        token=opcode_str,
                    )
                else:
                    raise SlotError(
                        f"Too many instructions of type {inst_type.__name__} (max {available_slots})\n"
                        f"At: {instruction['opcode'].get_location_string()}",
                        location=instruction["opcode"].location(),
                        token=opcode_str,
                    )

            if SLOTS[Slot(inst_type._slot_type_name())].debug_only:
                warnings.warn(
                    f"{instruction['opcode'].token.value} uses the simulation-only "
                    f"{inst_type._slot_type_name()} slot (not implemented in real IPU hardware)",
                    stacklevel=4,
                )
        return address

    def _fill_out_nop(self, address: int):
        inst_types_list = self.instruction_types()
        for i, inst_type in enumerate(inst_types_list):
            if self.instructions[i] is None:
                self.instructions[i] = inst_type.nop_inst(address)

    @classmethod
    @functools.cache
    def instruction_types(cls) -> tuple[type[inst.Inst], ...]:
        # Instruction slots in MSB → LSB order (encode places the last entry at LSB).
        # Execution order (break first, then LR, …) is handled in ipu.py, not here.
        # Slot order and counts come from the slot registry (ipu_isa.slots); built once.
        result = []
        for slot in LAYOUT_ORDER:
            result.extend([SLOT_TO_INST[slot]] * SLOTS[slot].count)
        return tuple(result)

    @classmethod
    @functools.cache
    def bits(cls) -> int:
        """The instruction word width. Static ISA data, so computed once."""
        return sum(instruction.bits() for instruction in cls.instruction_types())

    def encode(self) -> int:
        encoded_line = 0
        shift_amount = 0
        # Iterate through instructions in reverse order to maintain correct bit order
        for instruction in reversed(self.instructions):
            encoded_inst = instruction.encode()
            encoded_line |= encoded_inst << shift_amount
            shift_amount += instruction.bits()
        return encoded_line

    @classmethod
    def decode(cls, value: int) -> str:
        decoded_instructions = []
        shift_amount = 0
        for inst_type in reversed(cls.instruction_types()):
            inst_bits = inst_type.bits()
            inst_value = (value >> shift_amount) & ((1 << inst_bits) - 1)
            decoded_instructions.append(inst_type.decode(inst_value))
            shift_amount += inst_bits
        return ";\n\t\t\t".join(reversed(decoded_instructions)) + ";;"

    @classmethod
    def desc(cls) -> list[str]:
        res = []
        res.append(f"{cls.__name__} - {cls.bits()} bits:")
        res.append("Subitems:")
        for inst_type in cls.instruction_types():
            res.extend(f"\t{line}" for line in inst_type.desc())
        return res

    @classmethod
    @functools.cache
    def get_fields(cls) -> tuple[tuple[str, int], ...]:
        """``(name, bits)`` of every instruction word field, LSB first.

        Static ISA data, so computed once; the tuple is shared, hence immutable.
        ``CompoundInst.get_fields.cache_clear()`` recomputes it on the next call.
        """
        fields = []
        inst_types_list = cls.instruction_types()

        # Count occurrences of each type to generate unique names
        type_counts = {}
        type_indices = {}

        for inst_type in inst_types_list:
            type_counts[inst_type] = type_counts.get(inst_type, 0) + 1

        for inst_type in inst_types_list:
            # Determine the prefix for field names
            base_name = utils.camel_case_to_snake_case(inst_type.__name__)

            # Track which instance of this type we're on
            current_index = type_indices.get(inst_type, 0)
            type_indices[inst_type] = current_index + 1

            # If there are multiple instances of this type, add index to the name
            if type_counts[inst_type] > 1:
                inst_prefix = f"{base_name}_{current_index}"
            else:
                inst_prefix = base_name

            for i, token_type in enumerate(inst_type.all_tokens()):
                field_name = (
                    f"{inst_prefix}_token_{i}_{utils.camel_case_to_snake_case(token_type.__name__)}"
                )
                fields.append((field_name, token_type.bits()))
        return tuple(reversed(fields))


def check_word_fields(compound: type[CompoundInst]) -> None:
    """The assembler's word fields are the ISA's word layout (E03-16).

    ``get_fields()`` names fields after the token classes; ``WORD_LAYOUT`` is
    the wire format, built from the registries. Until E05-03 makes the
    assembler read ``WORD_LAYOUT``, the two must agree.
    """
    ours = list(compound.get_fields())
    isa_fields = [(f.name, f.bits) for f in WORD_LAYOUT]
    if ours != isa_fields:
        raise SpecError(
            "CompoundInst.get_fields() differs from ipu_isa.codec.word.WORD_LAYOUT: "
            f"{next((a, b) for a, b in zip(ours, isa_fields, strict=False) if a != b)}"
        )


check_word_fields(CompoundInst)
