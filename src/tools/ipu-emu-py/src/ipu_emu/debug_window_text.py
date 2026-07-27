"""Text/chat renderer for debug-window cycle snapshots.

Deterministic, compact serialisation of :class:`CycleSnapshot` — one line per
active slot per cycle, designed to be pasted into a chat window and read
without tooling. No prose, no LLM formatting.
"""

from __future__ import annotations

import struct

from ipu_emu.debug_window import (
    SLOT_DISPATCH_ORDER,
    CycleSnapshot,
    OperandRead,
    SlotExecution,
    WriteEffect,
)

_PREVIEW_BYTES = 8
_PREVIEW_WORDS = 4
_WORD_VIEW_REGS = {"r_acc", "mult_res"}


def _fmt_bytes(data: bytes, as_words: bool = False) -> str:
    if as_words and len(data) % 4 == 0 and len(data) >= 4:
        n = min(_PREVIEW_WORDS, len(data) // 4)
        words = struct.unpack_from(f"<{n}I", data, 0)
        s = " ".join(f"{w:08x}" for w in words)
        return f"[{s}{' ..' if len(data) // 4 > n else ''}]w"
    n = min(_PREVIEW_BYTES, len(data))
    s = "".join(f"{b:02x}" for b in data[:n])
    return f"[{s}{'..' if len(data) > n else ''}]"


def _fmt_value(op: OperandRead) -> str:
    if op.value is None:
        return ""
    if isinstance(op.value, (bytes, bytearray)):
        as_words = op.register in _WORD_VIEW_REGS
        return "=" + _fmt_bytes(bytes(op.value), as_words=as_words)
    return f"={op.value}"


def _fmt_addr(addr) -> str:
    if isinstance(addr, tuple) or isinstance(addr, list):
        start, length = addr
        if length is None:
            return f"[{start}:]"
        return f"[{start}:{start + length}]"
    return f"@{addr}"


def _fmt_operand(op: OperandRead) -> str:
    src = "SNAPSHOT" if op.source == "snapshot" else op.source
    where = op.register or ""
    if op.register == "xmem" and op.address_or_index is not None:
        start, length = op.address_or_index
        return f"{op.name}=xmem@0x{start:x}({length}B)"
    if op.address_or_index is not None and op.register not in (None, "xmem"):
        where += _fmt_addr(op.address_or_index)
    if op.source == "immediate":
        return f"{op.name}={op.register}" if op.register else f"{op.name}={op.value}"
    body = f"{where}({src})" if where else f"({src})"
    return f"{op.name}={body}{_fmt_value(op)}"


def _fmt_write(w: WriteEffect) -> str:
    if w.register == "xmem" and isinstance(w.address_or_index, tuple):
        start, length = w.address_or_index
        return f"xmem@0x{start:x}({length}B)(live-write)"
    where = w.register
    if w.address_or_index is not None:
        where += _fmt_addr(w.address_or_index)
    return f"{where}(live-write)"


def render_slot(slot: SlotExecution) -> str:
    parts: list[str] = [slot.assembly, "|"] if slot.operands or slot.writes else [slot.assembly]
    for op in slot.operands.values():
        parts.append(_fmt_operand(op))
    if slot.effective_mask is not None:
        parts.append(f"effective_mask=0x{slot.effective_mask:032x}")
    elif slot.mask_note:
        parts.append(f"mask:{slot.mask_note}")
    if slot.writes:
        parts.append("->")
        parts.append(" ".join(_fmt_write(w) for w in slot.writes))
    return " ".join(parts)


def render_cycle(snap: CycleSnapshot, *, notes: bool = False) -> str:
    """Render one cycle as a small text block."""
    header = f"cycle {snap.cycle} pc={snap.program_counter} -> pc={snap.pc_after}"
    if snap.break_hit:
        header += " [BREAK]"
    if snap.empty:
        return header + "  (empty instruction slot: NOP cycle)"
    lines = [header]
    for key in SLOT_DISPATCH_ORDER:
        slot = snap.slots.get(key)
        if slot is None:
            continue
        lines.append(f"  {key:<5}: {render_slot(slot)}")
        if notes:
            for op in slot.operands.values():
                if op.note:
                    lines.append(f"         . {op.name}: {op.note}")
            if slot.mask_note and slot.effective_mask is not None:
                lines.append(f"         . mask: {slot.mask_note}")
            for w in slot.writes:
                if w.note:
                    lines.append(f"         . write {w.register}: {w.note}")
    return "\n".join(lines)


def render_trace(snapshots: list[CycleSnapshot], *, notes: bool = False) -> str:
    """Render a cycle range as a paste-friendly text dump."""
    return "\n".join(render_cycle(s, notes=notes) for s in snapshots)
