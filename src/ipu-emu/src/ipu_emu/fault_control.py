"""Per-fault enable bits and sticky fault flags (E04-22, IPU-FLT-5, IPU-FLT-6).

An enabled fault halts the run and records its code and PC (E04-19). A
disabled one only sets its sticky flag, and the word goes on: F-1's slot runs
as NOP (the proposal on record in OQ-2), and a run past F-9's limit continues.
What a word does when F-2 to F-6 is disabled is open (OQ-2), so disabling one
is refused at setup rather than modelled by guess. F-7 and F-8 are not
modelled: their bits may be cleared and change nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ipu_emu.errors import ConfigurationError
from ipu_isa.faults import FaultId

__all__ = ["UNDECIDED", "FaultControl"]

#: Faults whose behaviour when disabled OQ-2 has not decided.
UNDECIDED = frozenset({FaultId.F2, FaultId.F3, FaultId.F4, FaultId.F5, FaultId.F6})


def _all_enabled() -> dict[FaultId, bool]:
    return dict.fromkeys(FaultId, True)


@dataclass
class FaultControl:
    """The fault control registers of one IPU.

    Attributes:
        enable: Each fault's enable bit; all set after reset.
        sticky: The faults that fired since the last reset, enabled or not.
    """

    enable: dict[FaultId, bool] = field(default_factory=_all_enabled)
    sticky: set[FaultId] = field(default_factory=set)

    def reset(self) -> None:
        """Set every enable bit and clear every sticky flag (IPU-FLT-6)."""
        self.enable.update(_all_enabled())
        self.sticky.clear()

    def set_enabled(self, fault: FaultId, enabled: bool) -> None:
        """Set or clear one enable bit.

        Raises:
            ConfigurationError: Clearing F-2 to F-6, whose disabled behaviour
                is not defined (OQ-2).
        """
        if not enabled and fault in UNDECIDED:
            raise ConfigurationError(_undecided(fault))
        self.enable[fault] = enabled

    def check(self) -> None:
        """Refuse a run with an undecided fault disabled (an ``enable`` write
        that bypassed :meth:`set_enabled`).

        Raises:
            ConfigurationError: F-2 to F-6 is disabled.
        """
        for fault in sorted(UNDECIDED):
            if not self.enable.get(fault, True):
                raise ConfigurationError(_undecided(fault))


def _undecided(fault: FaultId) -> str:
    return (
        f"{fault.label} cannot be disabled: what a word does with it disabled is not defined (OQ-2)"
    )
