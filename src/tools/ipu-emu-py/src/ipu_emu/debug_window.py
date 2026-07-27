"""Cycle-by-cycle VLIW debug trace capture for the IPU emulator.

Data model + tracer for the debug window (see planning/debug_window_spec.md).
Captures one :class:`CycleSnapshot` per executed VLIW cycle by driving
``Ipu.execute_vliw_cycle()`` and recording, per slot, which operands were
resolved, from which register file (live vs snapshot), and what addresses
were touched (XMEM, r_cyclic windows, R0/R1 broadcast elements, effective
mult masks).

This module is strictly read-only against the emulator: it observes state,
it never changes how instructions execute. Two renderers consume the model:

- :mod:`ipu_emu.debug_window_text` — deterministic text/chat dump
- :mod:`ipu_emu.debug_window_html` — self-contained interactive HTML stepper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ipu_common.instruction_spec import SLOT_COUNT, get_instruction_by_opcode
from ipu_common.activations import ACTIVATION_FN_NAMES
from ipu_emu.ipu import (
    Ipu,
    BreakResult,
    LR_REG_COUNT,
    R_REG_SIZE,
    R_CYCLIC_SIZE,
    R_ACC_SIZE,
    _INSTRUCTION_FIELD_MAP,
    _MULT_STAGE_MAP,
    _SLOT_FIELD_PREFIX,
)
from ipu_emu.ipu_config import LR_CR_SCALAR_BITS, Partition, decode_dstructure
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState
from ipu_emu.regfile import RegFile

# Slot rendering order == VLIW dispatch order (see Ipu.execute_vliw_cycle)
SLOT_DISPATCH_ORDER: tuple[str, ...] = (
    "break", "lr0", "lr1", "lr2",
    "load", "mult", "acc", "aaq", "store", "acc_store", "cond",
)

_128_BIT_MASK = (1 << R_REG_SIZE) - 1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OperandRead:
    """One value read while executing a slot, tagged with its source regfile."""

    name: str
    value: int | bytes | None
    source: Literal["live", "snapshot", "immediate"]
    # For register operands: which register / address was actually touched.
    register: str | None = None
    # e.g. xmem byte address, or an (start, length) window into r_cyclic / Ra.
    address_or_index: int | tuple[int, int] | None = None
    note: str | None = None


@dataclass
class WriteEffect:
    """One state write performed by a slot (always to the LIVE state)."""

    register: str  # "lr3", "r_cyclic", "mult_res", "xmem", "pc", ...
    address_or_index: int | tuple[int, int] | None = None
    note: str | None = None


@dataclass
class SlotExecution:
    slot_type: str
    instruction_name: str | None  # None if slot idle (NOP) this cycle
    operand_tokens: list[str] = field(default_factory=list)  # as-written form
    operands: dict[str, OperandRead] = field(default_factory=dict)
    writes: list[WriteEffect] = field(default_factory=list)
    # mult slots only: the 128-bit mask actually applied to mult_res
    effective_mask: int | None = None
    base_mask: int | None = None  # raw R_MASK slot before shift
    mask_note: str | None = None

    @property
    def assembly(self) -> str:
        if self.instruction_name is None:
            return "NOP"
        if not self.operand_tokens:
            return self.instruction_name
        return f"{self.instruction_name} " + ", ".join(self.operand_tokens)


@dataclass
class CycleSnapshot:
    cycle: int
    program_counter: int
    slots: dict[str, SlotExecution]  # keyed by SLOT_DISPATCH_ORDER names
    # Full register dump, LIVE state AFTER this cycle completed.
    regfile: dict[str, bytes]
    # The snapshot state used for "snapshot" reads THIS cycle (pre-cycle values).
    regfile_snapshot: dict[str, bytes]
    pc_after: int = 0
    break_hit: bool = False
    empty: bool = False  # inst_mem[pc] was None (pure NOP cycle)
    # Arithmetic mode for THIS cycle (DType name, e.g. "INT8", "E4") — governs
    # how R0/R1/r_cyclic bytes should be interpreted by dtype-aware renderers.
    dtype: str = "INT8"


# ---------------------------------------------------------------------------
# Effective mask — pure replication of Ipu._mult_mask_and_shift steps 1–4
# ---------------------------------------------------------------------------

def compute_effective_mask(
    mask_offset: int,
    mask_shift: int,
    partition: Partition,
    r_mask_bytes: bytes | bytearray,
) -> tuple[int, int]:
    """Return (base_mask, effective_mask) as 128-bit ints.

    Replicates the mask selection + sequential shift-and-AND of
    ``Ipu._mult_mask_and_shift`` WITHOUT the final zeroing step, so the debug
    tool can show the mask that will gate mult_res without touching state.
    """
    shift = mask_shift
    if shift >= (1 << (LR_CR_SCALAR_BITS - 1)):
        shift = shift - (1 << LR_CR_SCALAR_BITS)
    shift = max(-3, min(3, shift))

    mask_slot = mask_offset % (R_REG_SIZE // 16)  # 8 slots of 128 bits
    offset = mask_slot * 16
    base_mask = (
        int.from_bytes(bytes(r_mask_bytes[offset : offset + 16]), byteorder="little")
        & _128_BIT_MASK
    )

    mask_int = base_mask
    if shift < 0:
        pv = Ipu._build_inverse_partition_vector(partition)
        for _ in range(-shift):
            mask_int = (mask_int >> 1) & pv
    elif shift > 0:
        pv = Ipu._build_partition_vector(partition)
        for _ in range(shift):
            mask_int = (mask_int << 1) & pv & _128_BIT_MASK

    return base_mask, mask_int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dump_regfile(regfile: RegFile) -> dict[str, bytes]:
    """Immutable copy of every register's backing bytes."""
    return {name: bytes(regfile.raw(name)) for name in regfile.register_names}


def _operand_token(op_type: str, raw: int) -> str:
    """Render a raw encoded operand field the way it is written in assembly."""
    if op_type == "LrIdx":
        return f"lr{raw}"
    if op_type in ("CrIdx", "DstructureCrIdx"):
        return f"cr{raw}"
    if op_type == "LcrIdx":
        return f"lr{raw}" if raw < LR_REG_COUNT else f"cr{raw - LR_REG_COUNT}"
    if op_type == "MultStageReg":
        if raw < len(_MULT_STAGE_MAP):
            reg_name, elem = _MULT_STAGE_MAP[raw]
            return f"{reg_name}{elem}" if reg_name == "r" else reg_name
        return f"mult_stage({raw})"
    if op_type == "ActivationFn":
        if 0 <= raw < len(ACTIVATION_FN_NAMES):
            return ACTIVATION_FN_NAMES[raw]
        return f"activation({raw})"
    return str(raw)


def _register_name_for(op_type: str, raw: int) -> str | None:
    if op_type in ("LrIdx", "CrIdx", "DstructureCrIdx", "LcrIdx", "MultStageReg"):
        return _operand_token(op_type, raw)
    return None


def _ra_window_bytes(r_buf: bytes | bytearray, start: int, length: int) -> bytes:
    """Read from combined R0++R1 (256 bytes) with mod-256 wrap."""
    total = 2 * R_REG_SIZE
    start %= total
    out = bytearray(length)
    for i in range(length):
        out[i] = r_buf[(start + i) % total]
    return bytes(out)


# ---------------------------------------------------------------------------
# TracingIpu — records operand resolution without altering execution
# ---------------------------------------------------------------------------

class TracingIpu(Ipu):
    """An Ipu whose dispatch records what each slot read and wrote.

    Recording duplicates the (side-effect-free) operand resolution that the
    parent dispatcher performs, at the same point in the cycle, so "live"
    values reflect writes from earlier slots exactly as the handler sees them.
    Execution itself is delegated unchanged to :class:`Ipu`.
    """

    def __init__(self, state: IpuState):
        super().__init__(state)
        self._rec: dict[str, SlotExecution] | None = None

    # -- recording control ---------------------------------------------------

    def begin_recording(self) -> None:
        self._rec = {}

    def end_recording(self) -> dict[str, SlotExecution]:
        rec = self._rec or {}
        self._rec = None
        return rec

    # -- dispatch hooks --------------------------------------------------------

    def dispatch_instruction(self, slot_type: str, inst: dict[str, int]) -> Any:
        if self._rec is not None:
            slot = self._record_slot(slot_type, inst)
            if slot is not None:
                self._rec[slot_type] = slot
        return super().dispatch_instruction(slot_type, inst)

    def _dispatch_lr_slots(self, inst: dict[str, int]) -> None:
        if self._rec is not None:
            for slot_idx in range(SLOT_COUNT["lr"]):
                slot = self._record_lr_slot(slot_idx, inst)
                if slot is not None:
                    self._rec[f"lr{slot_idx}"] = slot
        super()._dispatch_lr_slots(inst)

    # -- slot recorders --------------------------------------------------------

    def _record_lr_slot(self, slot_idx: int, inst: dict[str, int]) -> SlotExecution | None:
        prefix = f"lr_inst_{slot_idx}"
        opcode = inst[f"{prefix}_token_0_lr_inst_opcode"]
        inst_name, spec = get_instruction_by_opcode("lr", opcode)
        if inst_name == "NOP":
            return None
        field_map = _INSTRUCTION_FIELD_MAP[("lr", inst_name, slot_idx)]
        return self._build_slot_execution(f"lr{slot_idx}", "lr", inst_name, spec,
                                          field_map, inst)

    def _record_slot(self, slot_type: str, inst: dict[str, int]) -> SlotExecution | None:
        prefix = _SLOT_FIELD_PREFIX[slot_type]
        opcode = inst[f"{prefix}_token_0_{slot_type}_inst_opcode"]
        inst_name, spec = get_instruction_by_opcode(slot_type, opcode)
        if inst_name == "NOP":
            return None
        field_map = _INSTRUCTION_FIELD_MAP[(slot_type, inst_name)]
        return self._build_slot_execution(slot_type, slot_type, inst_name, spec,
                                          field_map, inst)

    def _build_slot_execution(
        self,
        slot_key: str,
        slot_type: str,
        inst_name: str,
        spec: dict,
        field_map: dict[str, str],
        inst: dict[str, int],
    ) -> SlotExecution:
        live = self.state.regfile
        snap = self.snapshot
        raw: dict[str, int] = {name: inst[key] for name, key in field_map.items()}

        slot = SlotExecution(slot_type=slot_key, instruction_name=inst_name)
        resolved: dict[str, int | bytes] = {}

        for op in spec["operands"]:
            name, op_type = op["name"], op["type"]
            raw_val = raw[name]
            slot.operand_tokens.append(_operand_token(op_type, raw_val))
            read = op.get("read")
            if read is not None:
                source_rf = snap if read == "snapshot" else live
                try:
                    value = self._resolve_operand(op_type, raw_val, source_rf)
                except Exception:
                    value = None
                if isinstance(value, bytearray):
                    value = bytes(value)
                resolved[name] = value
                slot.operands[name] = OperandRead(
                    name=name,
                    value=value,
                    source=read,
                    register=_register_name_for(op_type, raw_val),
                )
            else:
                slot.operands[name] = OperandRead(
                    name=name,
                    value=raw_val,
                    source="immediate",
                    register=_register_name_for(op_type, raw_val),
                )

        self._annotate(slot, spec["execute_fn"], raw, resolved)
        return slot

    # -- per-instruction semantic annotation -----------------------------------

    def _annotate(
        self,
        slot: SlotExecution,
        execute_fn: str,
        raw: dict[str, int],
        res: dict[str, int | bytes],
    ) -> None:
        """Add reads the handler performs internally + write effects.

        Everything here mirrors the corresponding ``execute_*`` handler in
        ``ipu.py`` — the addresses/sources recorded are exactly what the
        handler computes, captured at the same point in the cycle.
        """
        live = self.state.regfile
        snap = self.snapshot
        wide = self._wide_vector_active()

        def _int(name: str) -> int:
            v = res.get(name, raw.get(name, 0))
            return int(v) if isinstance(v, int) else 0

        # ---- load slot -----------------------------------------------------
        if execute_fn == "execute_ldr_mult_reg":
            addr = _int("offset") + _int("base")
            size = R_CYCLIC_SIZE if wide else R_REG_SIZE
            slot.operands["xmem_read"] = OperandRead(
                "xmem_read", None, "live", "xmem", (addr, size))
            dest = raw.get("dest", 0)
            slot.writes.append(WriteEffect(_operand_token("MultStageReg", dest)))
        elif execute_fn == "execute_ldr_cyclic_mult_reg":
            addr = _int("offset") + _int("base")
            size = R_CYCLIC_SIZE if wide else R_REG_SIZE
            index = _int("index")
            slot.operands["xmem_read"] = OperandRead(
                "xmem_read", None, "live", "xmem", (addr, size))
            slot.writes.append(WriteEffect("r_cyclic", (index, size)))
        elif execute_fn == "execute_ldr_mult_mask_reg":
            addr = _int("offset") + _int("base")
            slot.operands["xmem_read"] = OperandRead(
                "xmem_read", None, "live", "xmem", (addr, R_REG_SIZE))
            slot.writes.append(WriteEffect("r_mask"))

        # ---- store slots ---------------------------------------------------
        elif execute_fn == "execute_str_post_aaq_reg":
            addr = _int("offset") + _int("base")
            narrow = self.state.dtype == DType.INT8 and not wide
            size = 128 if narrow else 512
            slot.operands["post_aaq_reg"] = OperandRead(
                "post_aaq_reg", bytes(live.raw("post_aaq_reg")[:size]),
                "live", "post_aaq_reg")
            slot.writes.append(WriteEffect("xmem", (addr, size)))
        elif execute_fn == "execute_str_acc_reg":
            addr = _int("offset") + _int("base")
            slot.operands["r_acc"] = OperandRead(
                "r_acc", bytes(live.raw("r_acc")), "live", "r_acc")
            slot.writes.append(WriteEffect("xmem", (addr, R_ACC_SIZE)))

        # ---- mult slot -------------------------------------------------------
        elif execute_fn in ("execute_mult_rc_vv", "execute_mult_rc_ve",
                            "execute_mult_rc_vs", "execute_mult_ve",
                            "execute_mult_ee"):
            self._annotate_mult(slot, execute_fn, raw, res, wide)

        # ---- acc slot --------------------------------------------------------
        elif execute_fn in ("execute_acc_add", "execute_acc_sub", "execute_acc_max"):
            slot.operands["mult_res"] = OperandRead(
                "mult_res", bytes(live.raw("mult_res")), "live", "mult_res")
            slot.operands["r_acc"] = OperandRead(
                "r_acc", bytes(snap.raw("r_acc")), "snapshot", "r_acc",
                note="the recurring-bug read: r_acc comes from the cycle-start "
                     "snapshot, NOT from a same-cycle write")
            slot.writes.append(WriteEffect("r_acc"))
        elif execute_fn in ("execute_acc_add_first", "execute_acc_sub_first",
                            "execute_acc_max_first"):
            slot.operands["mult_res"] = OperandRead(
                "mult_res", bytes(live.raw("mult_res")), "live", "mult_res")
            slot.writes.append(WriteEffect("r_acc", note="overwrite (no old r_acc read)"))
        elif execute_fn == "execute_acc_stride":
            slot.operands["mult_res"] = OperandRead(
                "mult_res", bytes(live.raw("mult_res")), "live", "mult_res")
            base = (_int("offset") % 4) * 32
            slot.writes.append(WriteEffect("r_acc", (base * 4, None),
                                           note=f"words starting at r_acc[{base}]"))
        elif execute_fn in ("execute_agg_sum", "execute_agg_sum_first",
                            "execute_agg_max", "execute_agg_max_first"):
            dest = _int("dest_slot") % (R_ACC_SIZE // 4)
            cr_idx = raw.get("cr_idx", 15)
            valid = decode_dstructure(live.get_cr(cr_idx)).valid_elements
            active = min(int(valid) & 0xFFFFFFFF, R_ACC_SIZE // 4)
            slot.operands["mult_res"] = OperandRead(
                "mult_res", bytes(live.raw("mult_res")[: active * 4]),
                "live", "mult_res", (0, active * 4),
                note=f"{active} active lanes (valid_elements from cr{cr_idx}, live)")
            if not execute_fn.endswith("_first"):
                slot.operands["r_acc_dest"] = OperandRead(
                    "r_acc_dest", bytes(snap.raw("r_acc")[dest * 4 : dest * 4 + 4]),
                    "snapshot", "r_acc", (dest * 4, 4),
                    note=f"seed word r_acc[{dest}] from SNAPSHOT")
            slot.writes.append(WriteEffect("r_acc", (dest * 4, 4),
                                           note=f"word r_acc[{dest}]"))

        # ---- aaq slot --------------------------------------------------------
        elif execute_fn == "execute_activate_quantize":
            cr_idx = raw.get("cr_idx", 15)
            valid = decode_dstructure(live.get_cr(cr_idx)).valid_elements
            active = min(int(valid) & 0xFFFFFFFF, R_ACC_SIZE // 4)
            slot.operands["r_acc"] = OperandRead(
                "r_acc", bytes(snap.raw("r_acc")[: active * 4]),
                "snapshot", "r_acc", (0, active * 4),
                note=f"{active} active lanes read from SNAPSHOT "
                     f"(valid_elements from cr{cr_idx}, live)")
            slot.writes.append(WriteEffect("post_aaq_reg"))

        # ---- cond slot -------------------------------------------------------
        elif execute_fn in ("execute_beq", "execute_bne", "execute_blt",
                            "execute_bge", "execute_br"):
            slot.writes.append(WriteEffect("pc", note="branch reads are SNAPSHOT: "
                                           "same-cycle LR writes are not seen"))

    def _annotate_mult(
        self,
        slot: SlotExecution,
        execute_fn: str,
        raw: dict[str, int],
        res: dict[str, int | bytes],
        wide: bool,
    ) -> None:
        live = self.state.regfile
        snap = self.snapshot

        def _int(name: str) -> int:
            v = res.get(name, raw.get(name, 0))
            return int(v) if isinstance(v, int) else 0

        # --- data reads (all SNAPSHOT — issue #157) --------------------------
        if execute_fn in ("execute_mult_rc_vv", "execute_mult_rc_ve",
                          "execute_mult_rc_vs"):
            rc_start = _int("rc_idx") % R_CYCLIC_SIZE
            value = None if wide else bytes(snap.get_r_cyclic_at(rc_start, R_REG_SIZE))
            slot.operands["rc_data"] = OperandRead(
                "rc_data", value, "snapshot", "r_cyclic", (rc_start, R_REG_SIZE),
                note=f"r_cyclic[{rc_start}:{rc_start + R_REG_SIZE}] (wraps at "
                     f"{R_CYCLIC_SIZE}); window index from LIVE lr, DATA from SNAPSHOT")

        if execute_fn == "execute_mult_rc_vv":
            # 'ra' operand already recorded via spec read=snapshot; add usage note
            ra_read = slot.operands.get("ra")
            if ra_read is not None:
                ra_read.address_or_index = (0, R_REG_SIZE)
                ra_read.note = "whole 128B vector, DATA from SNAPSHOT (issue #157)"
        elif execute_fn == "execute_mult_rc_ve":
            src = raw.get("src", 0)
            if src < LR_REG_COUNT:
                idx = live.get_lr(src) % (2 * R_REG_SIZE)
                scalar = None if wide else snap.raw("r")[idx]
                which = "r0" if idx < R_REG_SIZE else "r1"
                slot.operands["scalar"] = OperandRead(
                    "scalar", scalar, "snapshot", which,
                    (idx if idx < R_REG_SIZE else idx - R_REG_SIZE, 1),
                    note=f"single byte Ra[{idx}] — index from LIVE lr{src}, "
                         f"DATA from SNAPSHOT; broadcast to 128 lanes")
            else:
                cr_idx = src - LR_REG_COUNT
                slot.operands["scalar"] = OperandRead(
                    "scalar", live.get_cr(cr_idx) & 0xFF, "live", f"cr{cr_idx}",
                    note="CR low byte, LIVE; broadcast to 128 lanes")
        elif execute_fn == "execute_mult_ve":
            ra_start = _int("ra_idx") % (2 * R_REG_SIZE)
            value = None if wide else _ra_window_bytes(snap.raw("r"), ra_start, R_REG_SIZE)
            slot.operands["ra_data"] = OperandRead(
                "ra_data", value, "snapshot", "r0++r1", (ra_start, R_REG_SIZE),
                note=f"128B window of combined R0++R1 starting at byte {ra_start} "
                     f"(mod 256) — index from LIVE lr, DATA from SNAPSHOT (issue #157)")
        elif execute_fn == "execute_mult_ee":
            idx = _int("ra_idx") % (2 * R_REG_SIZE)
            scalar = None if wide else snap.raw("r")[idx]
            which = "r0" if idx < R_REG_SIZE else "r1"
            slot.operands["ra_data"] = OperandRead(
                "ra_data", scalar, "snapshot", which,
                (idx if idx < R_REG_SIZE else idx - R_REG_SIZE, 1),
                note=f"single broadcast byte Ra[{idx}], DATA from SNAPSHOT (issue #157)")

        if execute_fn in ("execute_mult_ve", "execute_mult_ee"):
            cr_idx = raw.get("cr_idx", 0)
            slot.operands["cr_scalar"] = OperandRead(
                "cr_scalar", live.get_cr(cr_idx) & 0xFF, "live", f"cr{cr_idx}",
                note="scalar multiplier = CR low byte, LIVE")

        # --- effective mask ----------------------------------------------------
        dcr = raw.get("dstructure_cr_idx", raw.get("cr_idx", 15))
        if wide:
            slot.mask_note = "wide-vector debug mode: masking is skipped entirely"
        else:
            mask_offset = raw.get("mask_offset", 0)
            mask_shift = _int("mask_shift")
            partition = decode_dstructure(live.get_cr(dcr)).partition
            base_mask, eff = compute_effective_mask(
                mask_offset, mask_shift, partition, live.get_r_mask())
            slot.base_mask = base_mask
            slot.effective_mask = eff
            slot.mask_note = (
                f"slot {mask_offset % 8} of LIVE r_mask, shift={mask_shift} "
                f"(sign-extended/clamped to [-3,3]), partition={int(partition)} "
                f"from cr{dcr} (live)")

        slot.writes.append(WriteEffect("mult_res"))


# ---------------------------------------------------------------------------
# Trace capture driver
# ---------------------------------------------------------------------------

def capture_trace(
    state: IpuState,
    *,
    start: int = 0,
    limit: int = 1000,
    max_cycles: int = 1_000_000,
    trim_trailing_nops: bool = True,
) -> list[CycleSnapshot]:
    """Run *state* to completion, capturing CycleSnapshots for cycles
    [start, start+limit).

    The state must already hold a loaded program (and any XMEM data). Cycles
    before *start* execute without capture; capture stops after *limit*
    snapshots or when the program halts, whichever comes first. Execution
    mirrors ``run_until_complete`` (breakpoints are skipped, not honoured).

    When the program runs off the end of its code (no BKPT), the PC NOP-walks
    the rest of instruction memory; *trim_trailing_nops* drops that empty tail
    from the returned trace.
    """
    ipu = TracingIpu(state)
    snapshots: list[CycleSnapshot] = []
    cycle = 0
    prev_after: dict[str, bytes] | None = None

    while not state.is_halted:
        if cycle >= max_cycles:
            raise RuntimeError(
                f"Exceeded {max_cycles} cycles — possible infinite loop "
                f"(PC={state.program_counter})")

        capturing = start <= cycle and len(snapshots) < limit
        if not capturing:
            if len(snapshots) >= limit:
                break
            result = ipu.execute_vliw_cycle()
            if result == BreakResult.BREAK:
                ipu.execute_vliw_cycle_skip_break()
            cycle += 1
            continue

        pc = state.program_counter
        inst = state.inst_mem[pc]
        # Cycle N's pre-state == cycle N-1's post-state; share the dump.
        before = prev_after if prev_after is not None else _dump_regfile(state.regfile)

        ipu.begin_recording()
        result = ipu.execute_vliw_cycle()
        if result == BreakResult.BREAK:
            # mirror run_until_complete: skip the break, execute the cycle anyway
            ipu.execute_vliw_cycle_skip_break()
        slots = ipu.end_recording()

        after = _dump_regfile(state.regfile)
        prev_after = after
        snapshots.append(CycleSnapshot(
            cycle=cycle,
            program_counter=pc,
            slots=slots,
            regfile=after,
            regfile_snapshot=before,
            pc_after=state.program_counter,
            break_hit=result == BreakResult.BREAK,
            empty=inst is None,
            dtype=state.dtype.name,
        ))
        cycle += 1

    if trim_trailing_nops:
        while snapshots and snapshots[-1].empty:
            snapshots.pop()
    return snapshots


# ---------------------------------------------------------------------------
# JSON-friendly serialisation (shared by the HTML renderer / --format json)
# ---------------------------------------------------------------------------

def _value_to_jsonable(value: int | bytes | None) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return {"hex": bytes(value).hex()}
    return value


def operand_to_dict(op: OperandRead) -> dict[str, Any]:
    return {
        "name": op.name,
        "value": _value_to_jsonable(op.value),
        "source": op.source,
        "register": op.register,
        "address_or_index": list(op.address_or_index)
        if isinstance(op.address_or_index, tuple) else op.address_or_index,
        "note": op.note,
    }


def slot_to_dict(slot: SlotExecution) -> dict[str, Any]:
    return {
        "slot_type": slot.slot_type,
        "instruction": slot.instruction_name,
        "assembly": slot.assembly,
        "operands": {k: operand_to_dict(v) for k, v in slot.operands.items()},
        "writes": [
            {
                "register": w.register,
                "address_or_index": list(w.address_or_index)
                if isinstance(w.address_or_index, tuple) else w.address_or_index,
                "note": w.note,
            }
            for w in slot.writes
        ],
        "effective_mask": f"{slot.effective_mask:032x}"
        if slot.effective_mask is not None else None,
        "base_mask": f"{slot.base_mask:032x}" if slot.base_mask is not None else None,
        "mask_note": slot.mask_note,
    }


def snapshot_to_dict(snap: CycleSnapshot, *, include_regfile_snapshot: bool = True) -> dict[str, Any]:
    d: dict[str, Any] = {
        "cycle": snap.cycle,
        "pc": snap.program_counter,
        "pc_after": snap.pc_after,
        "break_hit": snap.break_hit,
        "empty": snap.empty,
        "dtype": snap.dtype,
        "slots": {
            key: slot_to_dict(snap.slots[key])
            for key in SLOT_DISPATCH_ORDER if key in snap.slots
        },
        "regfile": {name: data.hex() for name, data in snap.regfile.items()},
    }
    if include_regfile_snapshot:
        d["regfile_snapshot"] = {
            name: data.hex() for name, data in snap.regfile_snapshot.items()
        }
    return d
