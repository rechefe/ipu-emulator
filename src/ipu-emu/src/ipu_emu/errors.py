"""Exceptions raised by the IPU emulator, all under ``ipu_isa.errors.IpuError``.

Two kinds: an :class:`EmulatorFault` is the hardware faulting while a program
runs; a :class:`ConfigurationError` is a bad setup, raised before any cycle,
or a CR content no layout allows (:class:`CrContentError`), raised in the
check phase of the word that reads it, before any of its writes (ADR 0007).
"""

from ipu_isa.errors import IpuError
from ipu_isa.faults import FaultId


class EmulatorError(IpuError):
    """Root of every error the emulator raises; also an invalid operation."""


class EmulatorFault(EmulatorError):
    """A hardware fault (F-1 to F-9) raised while a program runs (E04-19).

    The cycle that raises it records ``IpuState.halt`` before it propagates.

    Attributes:
        fault: Which fault (``ipu_isa.faults.FaultId``; ``FAULTS[fault]``
            describes it).
        pc: The PC of the faulting word.
        detail: What was illegal.
    """

    def __init__(self, fault: FaultId, pc: int, detail: str) -> None:
        super().__init__(f"{fault.label} at PC {pc}: {detail}")
        self.fault = fault
        self.pc = pc
        self.detail = detail

    def __reduce__(self) -> tuple[type, tuple[FaultId, int, str]]:
        return (type(self), (self.fault, self.pc, self.detail))


class UndeclaredRead(EmulatorError):
    """A handler read a register its instruction does not declare.

    Every register a handler reads is a ``read`` operand or one of the
    instruction's ``implicit_reads`` in ``instruction_spec.py``; the declaration
    gives the read its view (cycle start or forwarded). Raised by
    :meth:`ipu_emu.cycle.CycleView.source` (E04-12).
    """


class ConfigurationError(EmulatorError, ValueError):
    """A bad setup: the element format, the program size, prototyping without
    its build flag, or invalid API arguments. Raised at setup time, except
    :class:`CrContentError`, raised in a word's check phase. Also a
    ``ValueError``, as these checks raised before."""


class CrContentError(ConfigurationError):
    """A CR holds an invalid QF half, DS half or MULT constant (ADR 0007).

    The architects' layouts (``ipu_isa.cr_halves``) leave some contents
    meaningless: a QF with ``num_bits`` 0, a DS with ``valid_partitions`` 0 or
    ``mask_type`` NaN, a constant with ``fe`` 0. The hardware's behaviour for
    them is undefined, so the emulator refuses them. A configuration error
    (ADR 0005's kind, amended by ADR 0007), raised in the word's check phase,
    so the faulting word leaves the state as a fault would: nothing of it
    retires. The same contents are also refused before a run starts by
    tools that check a setup.
    """


class XMemBoundsError(ConfigurationError):
    """An XMEM access beyond the allocation. Outside a cycle (loading inputs,
    reading results) it is a setup error; inside one the cycle reports it as
    fault F-6, so it never leaves a run under this type."""


# Register-file validation errors (E00-02). Each one also derives from the
# matching built-in, so callers that catch ``IndexError`` / ``ValueError`` /
# ``TypeError`` keep working.


class RegisterIndexError(EmulatorError, IndexError):
    """Register or word index outside the register's declared count."""


class RegisterSizeError(EmulatorError, ValueError):
    """Data written to a register does not match the register's byte size."""


class RegisterKindError(EmulatorError, TypeError):
    """Scalar accessor used on a vector register."""


class DebugOnlyInstructionWarning(UserWarning, IpuError):
    """A debug-only instruction (no hardware equivalent) was executed.

    Emitted by an instruction of a debug-only slot (``SlotInfo.debug_only``).
    No slot is debug-only since ADR 0007 removed ``ST.ACC``; the class stays
    for the next one. Tests turn it into an error, so a kernel cannot start
    relying on such an instruction by accident.
    """
