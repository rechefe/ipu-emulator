"""Per-run execution statistics (issue #84).

Tracks mult/accumulate stage occupancy and memory access counts over a
single emulator run.  An instance is held on ``IpuState.stats`` and updated
by ``Ipu.dispatch_instruction`` during execution.

``mult_active_cycles`` counts *cycles*, which is not the same as MAC work: a
multiply against a constant 1.0 is how kernels emulate the vector move, add and
subtract the ISA lacks, and a 16-lane row still occupies all 128 lanes. The
lane- and identity-aware counters below exist so ``effective_mac_utilization``
can report what the multiplier actually retired. See
``docs/content/isa-alias-catalogue.md``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ipu_common.registers import get_lane_count

# The same constant ipu.LANES uses, from the same accessor, so the denominators
# here cannot drift from the lane counts accumulated there.
LANES = get_lane_count()


@dataclass
class RunStats:
    """Counters accumulated during one emulator run."""

    total_cycles: int = 0
    mult_active_cycles: int = 0
    acc_active_cycles: int = 0
    xmem_reads: int = 0
    xmem_writes: int = 0

    # Lanes that both passed the mult mask and lay within valid_elements,
    # summed over every multiply. Divided by total_cycles * LANES this is the
    # real occupancy of the 128-wide multiplier.
    mult_lane_ops: int = 0
    # Identity multiplies supplied by a CR or a declared constant ONES vector.
    mult_identity_cycles: int = 0
    # The lane-weighted counterpart, so identity work can be subtracted from
    # mult_lane_ops without re-assuming 128 lanes.
    mult_identity_lane_ops: int = 0
    # Count at issue time so debugger stops do not depend on the runner's
    # end-of-run total_cycles update.
    mult_idle_cycles: int = 0
    # Per-alias hit counts, keyed by ISA_ALIAS_SPEC id.
    alias_hits: Counter[str] = field(default_factory=Counter)
    # Optional independent measurements; overlaps never change MAC accounting.
    alias_profile: object | None = field(default=None, repr=False, compare=False)

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
    def lane_occupancy(self) -> float:
        """Fraction of the 128-wide multiplier that carried data (0.0–1.0)."""
        if self.total_cycles == 0:
            return 0.0
        return self.mult_lane_ops / (self.total_cycles * LANES)

    @property
    def effective_mac_utilization(self) -> float:
        """Fraction of peak MACs actually retired (0.0–1.0).

        Lane-weighted and net of identity multiplies, so a kernel that only
        moves data through the multiplier scores 0.0 however busy it looks.
        """
        if self.total_cycles == 0:
            return 0.0
        useful = self.mult_lane_ops - self.mult_identity_lane_ops
        return useful / (self.total_cycles * LANES)

    @property
    def identity_fraction(self) -> float:
        """Fraction of active mult cycles that were pure data routing."""
        if self.mult_active_cycles == 0:
            return 0.0
        return self.mult_identity_cycles / self.mult_active_cycles

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
            "--- MAC accounting ---",
            f"Mult idle:           {self.mult_idle_cycles:>8}",
            f"Identity mult:       {self.mult_identity_cycles:>8}"
            f"  ({self.identity_fraction * 100:.1f}% of active)",
            f"Lane ops:            {self.mult_lane_ops:>8}"
            f"  ({self.lane_occupancy * 100:.1f}% occupancy)",
            f"Effective MAC:       {self.effective_mac_utilization * 100:>7.1f}%",
        ]
        if self.mult_active_cycles and self.mult_lane_ops == self.mult_active_cycles * LANES:
            lines.append(
                "  note: every multiply used all 128 lanes — if this kernel pads a"
                " narrower row,"
            )
            lines.append(
                "        occupancy is overstated until it declares CR15.valid_elements."
            )
        if self.alias_hits:
            lines.append("--- ISA alias hits ---")
            for alias_id, count in self.alias_hits.most_common():
                lines.append(f"{alias_id + ':':<21}{count:>8}")
        if self.alias_profile is not None:
            lines.append(self.alias_profile.format_summary())
        return "\n".join(lines)
