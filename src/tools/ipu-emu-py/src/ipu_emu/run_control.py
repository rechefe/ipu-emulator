"""What :func:`ipu_emu.emulator.run_with_debug` reads and reports while it runs.

The emulator's one seam for a debugger: the caller that wants to share these
with a frontend creates a :class:`RunControl` (or a subclass) and passes it in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunControl:
    """Where a debug run stops, and what it last did.

    ``breakpoints`` is read-only here; a debugger's subclass decides how they
    change. ``stop_reason``, ``last_completed_pc`` and
    ``last_completed_instruction`` are written by the run.
    """

    _breakpoints: set[int] = field(default_factory=set, init=False)
    run_target: int | None = None
    stop_reason: str = "paused"
    last_completed_pc: int | None = None
    last_completed_instruction: dict[str, int] | None = None

    @property
    def breakpoints(self) -> frozenset[int]:
        return frozenset(self._breakpoints)
