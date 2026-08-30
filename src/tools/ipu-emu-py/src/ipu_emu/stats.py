"""Per-run execution statistics (issue #84).

Tracks mult/accumulate stage occupancy and memory access counts over a
single emulator run.  An instance is held on ``IpuState.stats`` and updated
by ``Ipu.dispatch_instruction`` during execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunStats:
    """Counters accumulated during one emulator run."""

    total_cycles: int = 0
    mult_active_cycles: int = 0
    acc_active_cycles: int = 0
    xmem_reads: int = 0
    xmem_writes: int = 0

    @property
    def mult_utilization(self) -> float:
        """Fraction of total cycles the mult stage was active (0.0–1.0)."""
        if self.total_cycles == 0:
            return 0.0
        return self.mult_active_cycles / self.total_cycles

    @property
    def acc_utilization(self) -> float:
        """Fraction of total cycles the accumulate stage was active (0.0–1.0)."""
        if self.total_cycles == 0:
            return 0.0
        return self.acc_active_cycles / self.total_cycles

    @property
    def xmem_accesses(self) -> int:
        return self.xmem_reads + self.xmem_writes

    def format_summary(self) -> str:
        """Return a multi-line human-readable summary."""
        tc = self.total_cycles
        mu = self.mult_utilization * 100
        au = self.acc_utilization * 100
        lines = [
            "=== Run summary ===",
            f"Total cycles:        {tc:>8}",
            f"Mult active:         {self.mult_active_cycles:>8}  ({mu:.1f}%)",
            f"Acc  active:         {self.acc_active_cycles:>8}  ({au:.1f}%)",
            f"XMEM reads:          {self.xmem_reads:>8}",
            f"XMEM writes:         {self.xmem_writes:>8}",
            f"XMEM accesses:       {self.xmem_accesses:>8}",
        ]
        return "\n".join(lines)
