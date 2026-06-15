"""SVG renderer for the IPU per-slot union layout.

Produces a diagram that shows, for every slot:

* the union fields packed by ``union_layout.py`` (one coloured box per field,
  width ∝ bit-width, labelled with the canonical type and bit-count), and
* a small grid below the field row showing, for each opcode, which operand
  lands in which field (or ``—`` when the field is unused).

The diagram is the canonical visual for the "Compound Instruction Layout"
section of the docs.  Slot order follows the binary layout MSB → LSB so the
picture matches what an encoded VLIW word looks like in memory.
"""
from __future__ import annotations

from ipu_common.union_layout import SlotUnion

# Slot palette — one stable colour per slot for cross-slot recognisability.
_SLOT_COLORS: dict[str, str] = {
    "cond": "#98D8C8",
    "lr": "#FFA07A",
    "load": "#FF6B6B",
    "store": "#E74C3C",
    "acc_store": "#C0392B",
    "mult": "#4ECDC4",
    "acc": "#45B7D1",
    "aaq": "#9B59B6",
    "break": "#FFD93D",
}
_OPCODE_FILL = "#D3D3D3"
_UNUSED_FILL = "#F8F8F8"

_FIELD_ROW_H = 64
_OPCODE_ROW_H = 14
_TITLE_BAND_H = 24
_BLOCK_PAD_H = 14


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _slot_block_height(su: SlotUnion) -> int:
    return (
        _TITLE_BAND_H
        + _FIELD_ROW_H
        + 6
        + len(su.opcode_bindings) * _OPCODE_ROW_H
        + _BLOCK_PAD_H
    )


def _render_slot(
    su: SlotUnion,
    slot_count: int,
    x: int,
    y: int,
    width: int,
    color: str,
) -> list[str]:
    out: list[str] = []
    field_widths = [su.opcode_bits] + [f.bits for f in su.fields]
    total_bits = sum(field_widths)

    count_suffix = f" (×{slot_count})" if slot_count > 1 else ""
    field_breakdown = "+".join(str(b) for b in field_widths[1:])
    title = (
        f"{su.slot.upper()}{count_suffix}  "
        f"({total_bits} bits = {su.opcode_bits}b opcode + {field_breakdown}b fields)"
    )
    out.append(
        f'  <text x="{x + width // 2}" y="{y + 16}" text-anchor="middle" '
        f'font-size="11" font-weight="bold" font-family="Arial">{_escape(title)}</text>'
    )

    field_row_y = y + _TITLE_BAND_H
    col_widths = [max(int(width * b / total_bits), 26) for b in field_widths]
    col_widths[-1] = width - sum(col_widths[:-1])

    labels = [("opcode", su.opcode_bits, _OPCODE_FILL)]
    labels += [(f.canonical_type, f.bits, color) for f in su.fields]

    px = x
    for (label, bits, fill), cw in zip(labels, col_widths):
        out.append(
            f'  <rect x="{px}" y="{field_row_y}" width="{cw}" height="{_FIELD_ROW_H}" '
            f'fill="{fill}" stroke="#333" stroke-width="1" rx="3"/>'
        )
        mid = px + cw // 2
        out.append(
            f'  <text x="{mid}" y="{field_row_y + _FIELD_ROW_H // 2 - 4}" '
            f'text-anchor="middle" font-size="9" font-weight="bold" '
            f'font-family="Arial">{_escape(label)}</text>'
        )
        out.append(
            f'  <text x="{mid}" y="{field_row_y + _FIELD_ROW_H // 2 + 10}" '
            f'text-anchor="middle" font-size="8" font-family="Arial">{bits}b</text>'
        )
        px += cw

    grid_y = field_row_y + _FIELD_ROW_H + 6
    label_col = 110
    grid_left = x + label_col
    grid_width = width - label_col
    data_widths = [int(grid_width * cw / width) for cw in col_widths[1:]]
    data_widths[-1] = grid_width - sum(data_widths[:-1])

    for oi, opcode in enumerate(sorted(su.opcode_bindings.keys())):
        row_y = grid_y + oi * _OPCODE_ROW_H
        out.append(
            f'  <text x="{x + 4}" y="{row_y + _OPCODE_ROW_H - 4}" '
            f'font-size="8.5" font-family="monospace" fill="#222">{_escape(opcode)}</text>'
        )
        binding_map = dict(su.opcode_bindings[opcode])
        cx = grid_left
        for fi, dw in enumerate(data_widths):
            field = su.fields[fi]
            if fi in binding_map:
                op_name = binding_map[fi]
                actual_type = field.users.get(opcode, (op_name, "?"))[1]
                cell_text = (
                    op_name
                    if actual_type == field.canonical_type
                    else f"{op_name}:{actual_type}"
                )
                cell_fill = color + "55"
            else:
                cell_text = "—"
                cell_fill = _UNUSED_FILL
            out.append(
                f'  <rect x="{cx}" y="{row_y + 1}" width="{dw}" height="{_OPCODE_ROW_H - 2}" '
                f'fill="{cell_fill}" stroke="#bbb" stroke-width="0.5"/>'
            )
            out.append(
                f'  <text x="{cx + dw // 2}" y="{row_y + _OPCODE_ROW_H - 4}" '
                f'text-anchor="middle" font-size="7.5" font-family="Arial">'
                f'{_escape(cell_text)}</text>'
            )
            cx += dw
    return out


def render_union_layout_svg(
    slot_unions: dict[str, SlotUnion],
    slot_order: list[str],
    slot_counts: dict[str, int] | None = None,
    width: int = 660,
    margin: int = 24,
    title: str = "IPU VLIW — Union Field Layout (per slot, MSB → LSB)",
) -> str:
    """Render a per-slot union layout diagram and return the SVG source.

    Args:
        slot_unions: ``{slot_name: SlotUnion}`` from ``compute_slot_layouts``.
        slot_order: Slot names in the order they should appear top-to-bottom
            (typically the MSB → LSB order of the compound instruction).
        slot_counts: Optional ``{slot_name: count}`` for slots that appear
            multiple times in the binary stream (e.g. ``lr`` ×3).  Annotates
            the slot title with the multiplicity; omit or pass 1 to suppress.
        width: SVG canvas width in pixels.
        margin: Outer margin around the diagram in pixels.
        title: Top-of-canvas title text.
    """
    slot_counts = slot_counts or {}
    inner_w = width - 2 * margin

    title_h = 28
    cursor = margin + title_h
    body: list[str] = []
    for slot in slot_order:
        su = slot_unions.get(slot)
        if su is None:
            continue
        color = _SLOT_COLORS.get(slot, "#CCCCCC")
        body.extend(
            _render_slot(
                su,
                slot_count=slot_counts.get(slot, 1),
                x=margin,
                y=cursor,
                width=inner_w,
                color=color,
            )
        )
        cursor += _slot_block_height(su) + margin

    canvas_h = cursor + margin
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{canvas_h}" viewBox="0 0 {width} {canvas_h}">',
        f'  <rect x="0" y="0" width="{width}" height="{canvas_h}" fill="white"/>',
        f'  <text x="{width // 2}" y="{margin + 14}" text-anchor="middle" '
        f'font-size="14" font-weight="bold" font-family="Arial">{_escape(title)}</text>',
    ]
    out.extend(body)
    out.append("</svg>")
    return "\n".join(out)
