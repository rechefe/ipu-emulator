import lark
import ipu_as.ipu_token as ipu_token


class Opcode(ipu_token.EnumToken):
    @classmethod
    def find_opcode_class(cls, opcode: lark.Token) -> type["Opcode"]:
        for subclass in cls.__subclasses__():
            if opcode.value in subclass.enum_array():
                return subclass
        raise ValueError(
            f"Opcode '{opcode.value}' declared in Line {opcode.line} and Column {opcode.column} not found in any Opcode subclass."
        )


class XmemInstOpcode(Opcode):
    @classmethod
    def enum_array(cls):
        return [
            "str_acc_reg",
            "ldr_mult_reg",
            "ldr_cyclic_mult_reg",
            "ldr_mult_mask_reg",
            "xmem_nop",
        ]


class LrInstOpcode(Opcode):
    @classmethod
    def enum_array(cls):
        return [
            "incr",
            "set",
            "add",
            "sub",
        ]


class MultInstOpcode(Opcode):
    @classmethod
    def enum_array(cls):
        return [
            "mult.ee",
            "mult.ev",
            "mult.ve",
            "mult_nop",
        ]
class AccInstOpcode(Opcode):
    @classmethod
    def enum_array(cls):
        return [
            "acc",
            "reset_acc",
            "acc_nop",
        ]

class MacInstOpcode(Opcode):
    @classmethod
    def enum_array(cls):
        return [
            "mac.ee",
            "mac.ev",
            "mac.agg",
            "zero_rq",
            "mac_nop",
        ]


class CondInstOpcode(Opcode):
    @classmethod
    def enum_array(cls):
        return [
            "beq",
            "bne",
            "blt",
            "bnz",
            "bz",
            "b",
            "br",
            "bkpt",
        ]


def validate_unique_opcodes():
    opcodes_subclasses = Opcode.__subclasses__()
    opcode_to_class = {}

    for cls in opcodes_subclasses:
        for opcode in cls.enum_array():
            if opcode in opcode_to_class:
                existing_class = opcode_to_class[opcode]
                error_msg = (
                    f"Duplicate opcode '{opcode}' found in classes "
                    f"'{existing_class.__name__}' and '{cls.__name__}'"
                )
                raise AssertionError(error_msg)
            opcode_to_class[opcode] = cls


# Call it to validate on module load
validate_unique_opcodes()
