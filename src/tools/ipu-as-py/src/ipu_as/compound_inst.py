import ipu_as.inst as inst
import ipu_as.utils as utils


INST_SEPARATOR = ";"


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
            inst_type = inst.Inst.find_inst_type_by_opcode(
                instruction["opcode"].token.value
            )
            if address is None:
                address = instruction["opcode"].instr_id

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
                    raise ValueError(
                        f"Instruction type {inst_type.__name__} is not allowed in compound instruction\n"
                        f"At: {instruction['opcode'].get_location_string()}"
                    )
                else:
                    raise ValueError(
                        f"Too many instructions of type {inst_type.__name__} (max {available_slots})\n"
                        f"At: {instruction['opcode'].get_location_string()}"
                    )
        return address

    def _fill_out_nop(self, address: int):
        inst_types_list = self.instruction_types()
        for i, inst_type in enumerate(inst_types_list):
            if self.instructions[i] is None:
                self.instructions[i] = inst_type.nop_inst(address)

    @classmethod
    def instruction_types(cls) -> list[type[inst.Inst]]:
        # Define the instruction slots in order (with duplicates for multiple instances)
        # Order matters for encoding/decoding
        return [
            inst.XmemInst,
            inst.MultInst,
            inst.AccInst,
            inst.LrInst,
            inst.LrInst,  # Second LrInst slot
            inst.CondInst,
        ]

    @classmethod
    def bits(cls) -> int:
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
    def get_fields(cls) -> list[tuple[str, int]]:
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
                    f"{inst_prefix}"
                    f"_token_{i}_{utils.camel_case_to_snake_case(token_type.__name__)}"
                )
                fields.append((field_name, token_type.bits()))
        return list(reversed(fields))
