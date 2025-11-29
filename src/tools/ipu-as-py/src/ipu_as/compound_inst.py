import ipu_as.inst as inst


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
        for instruction in instructions['instructions']:
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
                    f"First occurrence: {self.instructions[inst_type].opcode.get_location_string()}\n"
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

    @property
    def bits(self) -> int:
        return sum(instruction.bits() for instruction in self.instructions)

    def encode(self) -> int:
        encoded_line = 0
        shift_amount = 0
        for instruction in reversed(self.instructions.values()):
            encoded_inst = instruction.encode()
            encoded_line |= encoded_inst << shift_amount
            shift_amount += instruction.bits()
        return encoded_line
