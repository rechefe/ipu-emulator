"""Exceptions raised by the IPU emulator."""


class EmulatorError(RuntimeError):
    """Raised when the emulator detects an invalid operation."""


# Temporary register-file validation errors (E00-02). Each one also derives
# from the matching built-in so callers that catch ``IndexError`` /
# ``ValueError`` / ``TypeError`` keep working when E04-03 re-parents them.


class RegisterIndexError(EmulatorError, IndexError):
    """Register or word index outside the register's declared count."""


class RegisterSizeError(EmulatorError, ValueError):
    """Data written to a register does not match the register's byte size."""


class RegisterKindError(EmulatorError, TypeError):
    """Scalar accessor used on a vector register."""


class DebugOnlyInstructionWarning(UserWarning):
    """An emulator-only instruction (no hardware equivalent) was executed.

    Emitted by ``STR_ACC_REG`` (the ``acc_store`` slot). Tests that run such a
    program assert it with ``pytest.warns``; every other test turns it into an
    error, so a kernel cannot start relying on it by accident.
    """
