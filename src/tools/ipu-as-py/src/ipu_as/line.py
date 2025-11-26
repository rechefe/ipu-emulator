import ipu_as.inst as inst


INST_SEPARATOR = ";"


class Line:
    def __init__(self, line_str: str):
        self.instructions = []
        if not (line_str.count(INST_SEPARATOR) == len(self.available_instructions) - 1):
            raise ValueError(
                f"Line '{line_str}' does not match expected instruction count of {len(self.available_instructions)}"
            )
        for inst_str, inst_type in zip(line_str.split(INST_SEPARATOR), self.available_instructions):
            try:
                self.instructions.append(inst_type(inst_str))
            except ValueError as e:
                raise ValueError(f"Invalid instruction in line: {line_str} - \n{e}") from e

    @property
    def available_instructions(self) -> list[type[inst.Inst]]:
        return [inst.LrInst, inst.XmemInst, inst.MacInst, inst.CondInst]

    @property
    def bits(self) -> int:
        return sum(instruction.bits for instruction in self.instructions)

    def encode(self) -> int:
        encoded_line = 0
        shift_amount = 0
        for instruction in reversed(self.instructions):
            encoded_inst = instruction.encode()
            encoded_line |= encoded_inst << shift_amount
            shift_amount += instruction.bits
        return encoded_line
