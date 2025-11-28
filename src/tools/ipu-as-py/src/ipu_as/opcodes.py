import ipu_as.ipu_token as token


class Opcode(token.EnumToken):
    pass


class XmemInstOpcode(Opcode):
    @property
    def enum_array(self):
        return [
            "str",
            "ldr",
        ]


class LrInstOpcode(Opcode):
    @property
    def enum_array(self):
        return [
            "incr",
            "set",
        ]


class MacInstOpcode(Opcode):
    @property
    def enum_array(self):
        return [
            "mac.ee",
            "mac.ev",
        ]


class CondInstOpcode(Opcode):
    @property
    def enum_array(self):
        return [
            "beq",
            "bne",
            "blt",
            "bge",
            "bnz",
            "bz",
            "b",
            "bkpt",
        ]
