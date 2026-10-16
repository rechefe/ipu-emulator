"""Exceptions raised by the IPU emulator, all under ``ipu_isa.errors.IpuError``.

Two kinds: an :class:`EmulatorFault` is the hardware faulting while a program
runs; a :class:`ConfigurationError` is a bad setup, raised before any cycle.
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


class ConfigurationError(EmulatorError, ValueError):
    """A bad setup: the element format, the program size, prototyping without
    its build flag, or invalid API arguments. Raised at setup time, never
    mid-run. Also a ``ValueError``, as these checks raised before."""


class XMemBoundsError(EmulatorError, ValueError):
    """An XMEM access beyond the allocation. Raised by ``XMem`` at any time;
    inside a cycle the cycle reports it as fault F-6. Also a ``ValueError``,
    as the check raised before."""


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
    """An emulator-only instruction (no hardware equivalent) was executed.

    Emitted by ``STR_ACC_REG`` (the ``acc_store`` slot). Tests that run such a
    program assert it with ``pytest.warns``; every other test turns it into an
    error, so a kernel cannot start relying on it by accident.
    """
