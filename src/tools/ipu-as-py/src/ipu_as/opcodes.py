import ipu_as.ipu_token as token


class XmemInstOpcode(token.EnumToken):
    @property
    def enum_array(self):
        return [
            "str",
            "ldr",
        ]


class LrInstOpcode(token.EnumToken):
    @property
    def enum_array(self):
        return [
            "incr",
            "add",
            "set",
        ]


class MacInstOpcode(token.EnumToken):
    @property
    def enum_array(self):
        return [
            "mac",
            "macu",
            "macs",
            "macus",
        ]


class CondInstOpcode(token.EnumToken):
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
