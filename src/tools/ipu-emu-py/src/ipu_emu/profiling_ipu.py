"""Profiling IPU that tracks XMEM access patterns without running math.

``ProfilingIpu`` is a subclass of ``Ipu`` that:
- Intercepts XMEM slot instructions to record (cr_idx, addr) in a
  ``ProfilingXMem`` without actually reading/writing memory.
- No-ops the mult, acc, and aaq slots (skips all math calculations).
- Runs LR, COND, and BREAK slots normally (needed for control flow).

``run_profile`` is the top-level entry point.
"""

from __future__ import annotations

from typing import Any

from ipu_emu.ipu import Ipu, BreakResult, _INSTRUCTION_FIELD_MAP, _SLOT_FIELD_PREFIX
from ipu_emu.ipu_state import IpuState
from ipu_emu.profiling_xmem import ProfilingXMem
from ipu_emu.emulator import load_program_from_binary
from ipu_common.instruction_spec import get_instruction_by_opcode
from ipu_apps.base import IpuApp

# Opcode field key for the xmem slot
_XMEM_OPCODE_FIELD = (
    f"{_SLOT_FIELD_PREFIX['xmem']}_token_0_xmem_inst_opcode"
)


class ProfilingIpu(Ipu):
    """Ipu subclass that records XMEM access patterns and skips math."""

    def __init__(
        self,
        state: IpuState,
        profiling_xmem: ProfilingXMem,
        tracked_cr_indices: set[int] | None = None,
    ) -> None:
        super().__init__(state)
        self._profiling_xmem = profiling_xmem
        # Only record accesses for these CR indices (None = record all)
        self._tracked_cr_indices = tracked_cr_indices

    def dispatch_instruction(self, slot_type: str, inst: dict[str, int]) -> Any:
        if slot_type in ("mult", "acc", "aaq"):
            return None
        if slot_type == "xmem":
            return self._dispatch_xmem_profiling(inst)
        return super().dispatch_instruction(slot_type, inst)

    def _dispatch_xmem_profiling(self, inst: dict[str, int]) -> None:
        """Record the XMEM access address without executing the instruction."""
        opcode = inst[_XMEM_OPCODE_FIELD]
        instruction_name, _spec = get_instruction_by_opcode("xmem", opcode)

        field_map = _INSTRUCTION_FIELD_MAP.get(("xmem", instruction_name), {})
        if "base" not in field_map or "offset" not in field_map:
            return  # xmem_nop or instructions without base/offset

        raw_cr_idx = inst[field_map["base"]]    # CR index (not yet resolved)

        # Skip CRs that aren't data regions (e.g. parameter CRs)
        if self._tracked_cr_indices is not None and raw_cr_idx not in self._tracked_cr_indices:
            return

        raw_lr_idx = inst[field_map["offset"]]  # LR index (not yet resolved)

        # get_lr returns unsigned uint32; sign-extend to signed 32-bit
        offset_val = self.state.regfile.get_lr(raw_lr_idx)
        if offset_val >= 0x80000000:
            offset_val -= 0x100000000
        base_val = self.state.regfile.get_cr(raw_cr_idx)
        addr = offset_val + base_val

        self._profiling_xmem.record_access(raw_cr_idx, addr)


def run_profile(
    app: IpuApp,
    cr_names: dict[int, str],
    max_cycles: int = 1_000_000,
) -> dict[str, int]:
    """Run *app* in profiling mode and return peak lookahead per region.

    Math slots (mult, acc, aaq) are no-op'd; only LR, XMEM address tracking,
    COND (branches), and BREAK execute.

    Args:
        app:        An IpuApp instance (inst_path must be set, setup() is called).
        cr_names:   Maps CR index → region label, e.g. {0: "inputs", 1: "weights"}.
        max_cycles: Safety limit to catch infinite loops.

    Returns:
        ``{region_label: peak_lookahead_rows}`` for every CR index that was
        accessed during the run.  CR indices not in *cr_names* appear as
        ``"cr<idx>"``.
    """
    state = IpuState()
    load_program_from_binary(state, app.inst_path)
    app.setup(state)

    profiling_xmem = ProfilingXMem()
    state.xmem = profiling_xmem

    ipu = ProfilingIpu(state, profiling_xmem, tracked_cr_indices=set(cr_names.keys()))
    cycles = 0
    while not state.is_halted:
        if cycles >= max_cycles:
            raise RuntimeError(
                f"Exceeded {max_cycles} cycles — possible infinite loop "
                f"(PC={state.program_counter})"
            )
        result = ipu.execute_vliw_cycle()
        if result == BreakResult.BREAK:
            # In profiling mode, ignore breakpoints and complete the cycle
            ipu.execute_vliw_cycle_skip_break()
        cycles += 1

    raw = profiling_xmem.results()
    return {cr_names.get(k, f"cr{k}"): v for k, v in raw.items()}
