import ipu_as.inst as inst
import ipu_as.utils as utils


INST_SEPARATOR = ";"


class CompoundInst:
    def __init__(
        self,
        instructions: list[dict[str, any]],
    ):
        self.instructions = {inst_type: None for inst_type in self.instruction_types()}
        self._fill_out_nop(self._fill_instructions(instructions))

    def _fill_instructions(self, instructions: list[dict[str, any]]) -> int:
        address = None
        for instruction in instructions["instructions"]:
            inst_type = inst.Inst.find_inst_type_by_opcode(
                instruction["opcode"].token.value
            )
            if address is None:
                address = instruction["opcode"].instr_id
            if not self.instructions[inst_type]:
                self.instructions[inst_type] = inst_type(instruction)
            else:
                raise ValueError(
                    f"Duplicate instruction of type {inst_type.__name__} in compound instruction\n"
                    f"First occurrence: {self.instructions[inst_type].opcode.annotated_token.get_location_string()}\n"
                    f"Second occurrence: {instruction['opcode'].get_location_string()}"
                )
        return address

    def _fill_out_nop(self, address: int):
        for inst_type, instruction in self.instructions.items():
            if not instruction:
                self.instructions[inst_type] = inst_type.nop_inst(address)

    @classmethod
    def instruction_types(cls) -> list[type[inst.Inst]]:
        return [inst_type for inst_type in inst.Inst.__subclasses__()]

    @classmethod
    def bits(cls) -> int:
        return sum(instruction.bits() for instruction in cls.instruction_types())

    def encode(self) -> int:
        encoded_line = 0
        shift_amount = 0
        for instruction in reversed(self.instructions.values()):
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
        for inst_type in cls.instruction_types():
            for i, token_type in enumerate(inst_type.all_tokens()):
                field_name = (
                    f"{utils.camel_case_to_snake_case(inst_type.__name__)}"
                    f"_token_{i}_{utils.camel_case_to_snake_case(token_type.__name__)}"
                )
                fields.append((field_name, token_type.bits()))
        return reversed(fields)
