"""The root of every error the IPU tooling packages raise, and the ISA's own."""


class IpuError(Exception):
    """Root of every error raised by IPU tooling packages."""


class IsaLookupError(IpuError, KeyError):
    """A name or number the ISA does not have: an instruction, opcode, operand,
    register or operand type. A ``KeyError`` too, so callers that catch that keep
    working."""


class IsaValueError(IpuError, ValueError):
    """A value the ISA cannot represent, or a question it cannot answer (a
    mnemonic in several slots, a target that is neither emulator nor hardware).
    A ``ValueError`` too, so callers that catch that keep working."""
