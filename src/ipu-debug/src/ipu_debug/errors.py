"""Exceptions raised by the debugger, all under ``ipu_isa.errors.IpuError``.

Each also derives from the builtin it replaced, so an existing
``except ValueError`` or ``except RuntimeError`` still catches it.
"""

from __future__ import annotations

from ipu_isa.errors import IpuError

__all__ = ["DebugError", "TerminalUnavailableError"]


class DebugError(IpuError, ValueError):
    """A user-facing debugger failure; the message is the text shown."""


class TerminalUnavailableError(IpuError, RuntimeError):
    """The debugger was launched without an interactive input and output terminal."""
