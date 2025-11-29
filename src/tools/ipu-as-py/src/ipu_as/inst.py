import lark
import ipu_as.opcodes as opcodes
import ipu_as.ipu_token as ipu_token
import ipu_as.reg as reg
import ipu_as.immediate as immediate


def validate_inst_structure(cls: type) -> type:
    """Class decorator to validate instruction structure."""
    cls._validate_instr_structure()
    return cls


class Inst:
    def __init__(self, inst: dict[str, any]):
        self.opcode = self.opcode_type()(inst["opcode"])
        if len(inst["operands"]) != len(
            self.struct_by_opcode_table()[inst["opcode"].token.value]
        ):
            raise ValueError(
                f"Instruction {inst['opcode'].token.value} expects {len(self.struct_by_opcode_table()[inst['opcode'].token.value])} operands, "
                f"got {len(inst['operands'])}, in Line {self.opcode.token.line}, Column {self.opcode.token.column}."
            )

        self.operands = [
            op_type(op)
            for op_type, op in zip(
                self.struct_by_opcode_table()[inst["opcode"].token.value],
                inst["operands"],
            )
        ]
        self.specific_operand_types = self.struct_by_opcode_table()[
            inst["opcode"].token.value
        ]

    def _get_full_token_list(self) -> list[ipu_token.IpuToken]:
        full_token_list = [None for _ in range(1 + len(self.operand_types()))]
        full_token_list[0] = self.opcode
        for i, operand in enumerate(self.operands):
            mapped_index = self._inst_mapping_table[self.opcode.token.value][i]
            full_token_list[mapped_index + 1] = operand
        for i in range(len(full_token_list)):
            if full_token_list[i] is None:
                full_token_list[i] = self.operand_types()[i - 1].default()
        return full_token_list

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        raise NotImplementedError(
            "struct_by_opcode_table property must be implemented by subclasses"
        )

    @classmethod
    def _validate_instr_structure(cls) -> None:
        cls._inst_mapping_table = dict()
        for opcode, token_list in cls.struct_by_opcode_table().items():
            assert (
                opcode in cls.opcode_type().enum_array()
            ), f"Configuration of {cls.__name__} is invalid, opcode key must be of type {cls.opcode_type().__name__}"

            cls._inst_mapping_table[opcode] = cls._find_instruction_inst_mapping(
                token_list
            )

    @classmethod
    def _find_instruction_inst_mapping(cls, token_list: list[type[ipu_token.IpuToken]]):
        inst_mapping = [None for _ in token_list]
        full_token_list = [False for _ in cls.operand_types()]
        for i, token in enumerate(token_list):
            for j, token_type in enumerate(cls.operand_types()):
                if token == token_type and not full_token_list[j]:
                    full_token_list[j] = True
                    inst_mapping[i] = j
                    break
        assert None not in inst_mapping, f"Configuration of {cls.__name__} is invalid"
        return inst_mapping

    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        raise NotImplementedError(
            "operand_types property must be implemented by subclasses"
        )

    @classmethod
    def nop_inst(cls) -> str:
        raise NotImplementedError("nop_inst method must be implemented by subclasses")

    @staticmethod
    def _reversed_inst_mapping_table(mapping: list[int]) -> dict[int, int]:
        return {j: i for i, j in enumerate(mapping)}

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        raise NotImplementedError(
            "opcode_type method must be implemented by subclasses"
        )

    def encode(self) -> int:
        encoded_inst = 0
        shift_amount = 0
        for token in reversed(self._get_full_token_list()):
            encoded_inst |= token.encode() << shift_amount
            shift_amount += token.bits()
        return encoded_inst

    @classmethod
    def bits(cls) -> int:
        return sum(token_type.bits() for token_type in cls.all_tokens())

    @classmethod
    def find_inst_type_by_opcode(cls, opcode: str) -> type["Inst"]:
        for subclass in cls.__subclasses__():
            if opcode in subclass.struct_by_opcode_table().keys():
                return subclass
        raise ValueError(f"Opcode '{opcode}' not found in any Inst subclass.")

    @classmethod
    def all_tokens(cls) -> list[type[ipu_token.IpuToken]]:
        return [cls.opcode_type()] + cls.operand_types()

    @classmethod
    def decode(cls, value: int) -> str:
        decoded_tokens = []
        shift_amount = 0
        for token_type in reversed(cls.all_tokens()):
            token_bits = token_type.bits()
            token_value = (value >> shift_amount) & ((1 << token_bits) - 1)
            decoded_tokens.append(token_type.decode(token_value))
            shift_amount += token_bits
        return " ".join(reversed(decoded_tokens))

    @classmethod
    def desc(cls) -> list[str]:
        res = []
        res.append(f"{cls.__name__} - {cls.bits()} bits:")
        for token_type in cls.all_tokens():
            res.append(f"\t{token_type.__name__} - {token_type.bits()} bits")
        return res


@validate_inst_structure
class XmemInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, reg.CrRegField]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.XmemInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        return {
            "ldr": [reg.LrRegField, reg.CrRegField],
            "str": [reg.LrRegField, reg.CrRegField],
            "xmem_nop": [],
        }

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return XmemInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "xmem_nop", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )


@validate_inst_structure
class MacInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.RxRegField, reg.RxRegField, reg.RxRegField, reg.LrRegField]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.MacInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        return {
            "mac.ee": [reg.RxRegField, reg.RxRegField, reg.RxRegField],
            "mac.ev": [reg.RxRegField, reg.RxRegField, reg.RxRegField, reg.LrRegField],
            "mac_nop": [],
        }

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return MacInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "mac_nop", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [],
            }
        )


@validate_inst_structure
class LrInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, immediate.LrImmediateType]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.LrInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        return {
            "incr": [reg.LrRegField, immediate.LrImmediateType],
            "set": [reg.LrRegField, immediate.LrImmediateType],
        }

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return LrInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "incr", line=0, column=0),
                    instr_id=addr,
                ),
                "operands": [
                    ipu_token.AnnotatedToken(
                        token=lark.Token("TOKEN", "lr0", line=0, column=0),
                        instr_id=addr,
                    ),
                    ipu_token.AnnotatedToken(
                        token=lark.Token("TOKEN", "0", line=0, column=0),
                        instr_id=addr,
                    ),
                ],
            }
        )


@validate_inst_structure
class CondInst(Inst):
    @classmethod
    def operand_types(cls) -> list[type[ipu_token.IpuToken]]:
        return [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken]

    @classmethod
    def opcode_type(cls) -> type[ipu_token.IpuToken]:
        return opcodes.CondInstOpcode

    @classmethod
    def struct_by_opcode_table(cls) -> dict[str, list[type[ipu_token.IpuToken]]]:
        return {
            "beq": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "bne": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "blt": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "bnz": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "bz": [reg.LrRegField, reg.LrRegField, ipu_token.LabelToken],
            "b": [ipu_token.LabelToken],
            "br": [reg.LrRegField],
            "bkpt": [],
        }

    @classmethod
    def nop_inst(cls, addr: int) -> str:
        return CondInst(
            {
                "opcode": ipu_token.AnnotatedToken(
                    token=lark.Token("TOKEN", "b", line=0, column=0), instr_id=addr
                ),
                "operands": [
                    ipu_token.AnnotatedToken(
                        token=lark.Token("TOKEN", "+1", line=0, column=0), instr_id=addr
                    )
                ],
            }
        )
