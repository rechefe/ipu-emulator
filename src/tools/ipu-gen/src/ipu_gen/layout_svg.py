"""SVG renderers for the instruction word layout (E08-14).

``struct_layout_svg`` draws the whole instruction word as rows of 32 bits, every
field in its bit position. ``union_layout_svg`` (over ``render_union_layout_svg``)
shows, for every slot:

* the union fields packed by ``union_layout.py`` (one coloured box per field,
  width ∝ bit-width, labelled with the canonical type and bit-count), and
* a small grid below the field row showing, for each opcode, which operand
  lands in which field (or ``—`` when the field is unused).

The diagram is the canonical visual for the "Compound Instruction Layout"
section of the docs.  Slot order follows the binary layout MSB → LSB so the
picture matches what an encoded instruction word looks like in memory. Both read
the ISA (``LAYOUT_ORDER``, ``SLOTS``, ``SLOT_UNIONS``); E05-03 moves them onto its
word codec (``WORD_LAYOUT``).
"""

from __future__ import annotations

import re

from ipu_isa.layout import SLOT_UNIONS, word_bits
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot
from ipu_isa.union_layout import SlotUnion

# Slot palette, from the slot registry: one stable colour per slot.
_SLOT_COLORS: dict[str, str] = {slot.value: info.legend_colour for slot, info in SLOTS.items()}
_OPCODE_FILL = "#D3D3D3"
_UNUSED_FILL = "#F8F8F8"

_FIELD_ROW_H = 64
_OPCODE_ROW_H = 14
_TITLE_BAND_H = 24
_BLOCK_PAD_H = 14


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _slot_block_height(su: SlotUnion) -> int:
    return _TITLE_BAND_H + _FIELD_ROW_H + 6 + len(su.opcode_bindings) * _OPCODE_ROW_H + _BLOCK_PAD_H


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
    for (label, bits, fill), cw in zip(labels, col_widths, strict=True):
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
                    op_name if actual_type == field.canonical_type else f"{op_name}:{actual_type}"
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
                f"{_escape(cell_text)}</text>"
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
        slot_unions: ``{slot_name: SlotUnion}`` (``ipu_isa.layout.SLOT_UNIONS``).
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


#: The word diagram's title names the assembler class that used to draw it;
#: kept so the generated pages stay byte-identical.
_WORD_TITLE = "CompoundInst"


def _word_slots() -> list[Slot]:
    """Every slot copy of the instruction word, MSB first."""
    return [slot for slot in LAYOUT_ORDER for _ in range(SLOTS[slot].count)]


def _snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _slot_tokens(slot: Slot) -> list[tuple[str, int]]:
    """``(snake_case token name, bits)`` of a slot's opcode, then its union fields."""
    union = SLOT_UNIONS[slot.value]
    tokens = [(SLOTS[slot].asm_opcode_class, union.opcode_bits)]
    tokens += [(OPERAND_TYPES[f.canonical_type].token_class, f.bits) for f in union.fields]
    return [(_snake_case(name), bits) for name, bits in tokens]


def union_layout_svg() -> str:
    """The per-slot union layout of the ISA, slots in word order (MSB → LSB).

    Returns:
        The SVG source.
    """
    return render_union_layout_svg(
        dict(SLOT_UNIONS),
        slot_order=[slot.value for slot in LAYOUT_ORDER],
        slot_counts={slot.value: SLOTS[slot].count for slot in LAYOUT_ORDER},
    )


# The whole-word diagram: one row of 32 bits, ``_BIT_W`` px per bit, per 32 bits.
_BIT_W = 24
_ROW_H = 80
_PAD = 20
_ROW_W = 32 * _BIT_W
_TITLE_PX = 14
_LABEL_PX = 8
_BITS_PX = 7
_ROW_X = _PAD + 60
_TEXT_LINE_H = 11
_LEGEND_BOX = 12


def _word_fields() -> list[dict]:
    """Every field of the instruction word, LSB first, with its bit range and slot colour."""
    colours = {slot: info.legend_colour for slot, info in SLOTS.items()}
    fields = []
    current_bit = 0
    for slot in reversed(_word_slots()):
        for token_name, token_bits in reversed(_slot_tokens(slot)):
            fields.append(
                {
                    "name": _smart_abbreviate_field_name(token_name),
                    "bits": token_bits,
                    "start_bit": current_bit,
                    "end_bit": current_bit + token_bits - 1,
                    "color": colours.get(slot, "#CCCCCC"),
                }
            )
            current_bit += token_bits
    return fields


def _label_lines(name: str) -> list[str]:
    """A field label split into lines; words over 10 characters are shortened."""
    short = {"immediate": "imm", "opcode": "op", "stage": "stg"}
    lines = []
    for word in name.split():
        if len(word) <= 10:
            lines.append(word)
            continue
        prefix = next((p for p in short if word.startswith(p)), None)
        lines.append(short[prefix] if prefix else word[:5])
    return lines


def _field_svg(field: dict, row_end_bit: int, row_start_bit: int, x0: float, y: float) -> list[str]:
    """One field's box, label and bit range, within one row."""
    field_start = max(field["start_bit"], row_start_bit)
    field_end = min(field["end_bit"], row_end_bit)
    width_px = (field_end - field_start + 1) * _BIT_W
    field_x = x0 + (row_end_bit - field_end) * _BIT_W
    out = [
        f'  <rect x="{field_x}" y="{y}" '
        f'width="{width_px}" height="{_ROW_H}" '
        f'fill="{field["color"]}" stroke="black" stroke-width="1" opacity="0.85"/>'
    ]
    center_x = field_x + width_px / 2
    center_y = y + _ROW_H / 2
    lines = _label_lines(field["name"])
    text_height = len(lines) * _TEXT_LINE_H
    start_y = center_y - text_height / 2
    for i, line in enumerate(lines):
        out.append(
            f'  <text x="{center_x}" y="{start_y + i * _TEXT_LINE_H}" '
            f'text-anchor="middle" class="field-label">{line}</text>'
        )
    out.append(
        f'  <text x="{center_x}" y="{center_y + text_height / 2 + 3}" '
        f'text-anchor="middle" class="field-label" style="font-weight: bold;">'
        f"[{field['end_bit']}:{field['start_bit']}]</text>"
    )
    return out


def _row_svg(row_idx: int, num_rows: int, total_bits: int, fields: list[dict]) -> list[str]:
    """One row of 32 bits (row 0 is bits 31..0), drawn top to bottom from the MSB."""
    row_start_bit = row_idx * 32
    row_end_bit = min(row_start_bit + 31, total_bits - 1)
    y = _PAD + _TITLE_PX * 2.5 + ((num_rows - 1 - row_idx) * _ROW_H)
    out = [
        f'  <rect x="{_ROW_X}" y="{y}" width="{_ROW_W}" height="{_ROW_H}" '
        f'fill="white" stroke="black" stroke-width="2"/>',
        f'  <text x="{_PAD + 30}" y="{y + _ROW_H / 2 + _LABEL_PX / 2}" '
        f'text-anchor="end" class="bit-range">[{row_end_bit}:{row_start_bit}]</text>',
    ]
    x0 = _ROW_X + (32 - (row_end_bit - row_start_bit + 1)) * _BIT_W
    for bit_pos in range(row_end_bit, row_start_bit - 1, -1):
        bit_x = x0 + (row_end_bit - bit_pos + 0.5) * _BIT_W
        out.append(
            f'  <text x="{bit_x}" y="{y + 12}" text-anchor="middle" '
            f'class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">{bit_pos}</text>'
        )
    for field in fields:
        if field["start_bit"] <= row_end_bit and field["end_bit"] >= row_start_bit:
            out += _field_svg(field, row_end_bit, row_start_bit, x0, y)
    return out


def _legend_svg(legend_y: float) -> list[str]:
    """The slot colour key below the rows."""
    out = [
        f'  <text x="{_ROW_X}" y="{legend_y}" class="inst-label" '
        f'style="font-weight: bold;">Instruction Type Colors:</text>'
    ]
    for i, info in enumerate(SLOTS.values()):
        y_pos = legend_y + 15 + i * 15
        out.append(
            f'  <rect x="{_ROW_X}" y="{y_pos - _LEGEND_BOX + 2}" '
            f'width="{_LEGEND_BOX}" height="{_LEGEND_BOX}" '
            f'fill="{info.legend_colour}" stroke="black" stroke-width="1" opacity="0.85"/>'
        )
        out.append(
            f'  <text x="{_ROW_X + _LEGEND_BOX + 8}" y="{y_pos + 2}" '
            f'class="field-label" style="font-size: 9px;">{info.legend_label}</text>'
        )
    return out


def struct_layout_svg() -> str:
    """Render the whole instruction word as bit rows of 32.

    Shows every operand field in its actual bit position, coloured by the
    owning slot.  Complements ``union_layout_svg`` by giving the whole-word
    picture (what an encoded instruction word looks like in memory) without
    folding cross-opcode field sharing. Field labels are the token-class names
    the ISA records (``SlotInfo.asm_opcode_class``, ``OperandType.token_class``).

    Returns:
        The SVG source.
    """
    fields = _word_fields()
    total_bits = word_bits()
    num_rows = (total_bits + 31) // 32
    legend_height = 20 + len(SLOTS) * 15
    svg_width = _ROW_W + _PAD * 2 + 80
    svg_height = (num_rows * _ROW_H) + _TITLE_PX * 4 + _PAD * 3 + legend_height
    svg_lines = [
        f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
        "  <defs>",
        "    <style>",
        f"      .inst-title {{ font-size: {_TITLE_PX}px; font-weight: bold; font-family: Arial; }}",
        f"      .inst-label {{ font-size: {_LABEL_PX}px; font-weight: bold; font-family: Arial; }}",
        f"      .field-label {{ font-size: {_BITS_PX}px; font-family: Arial; }}",
        f"      .bit-range {{ font-size: {_LABEL_PX}px; font-family: Arial; }}",
        "    </style>",
        "  </defs>",
        f'  <rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="white" stroke="none"/>',
        f'  <text x="{svg_width / 2}" y="{_PAD + _TITLE_PX}" '
        f'text-anchor="middle" class="inst-title">'
        f"{_WORD_TITLE} Layout - {total_bits} total bits</text>",
    ]
    for row_idx in range(num_rows - 1, -1, -1):
        svg_lines += _row_svg(row_idx, num_rows, total_bits, fields)
    svg_lines += _legend_svg(_PAD + _TITLE_PX * 2.5 + (num_rows * _ROW_H) + 20)
    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def _smart_abbreviate_field_name(name: str) -> str:
    """Convert a snake_case token-class name into a readable space-separated label."""
    if name.endswith("_field"):
        name = name[:-6]
    if name.endswith("_type"):
        name = name[:-5]
    formatted_parts: list[str] = []
    for part in name.split("_"):
        words = re.findall("[A-Z][a-z]*|[a-z]+", part)
        formatted_parts.extend(words if words else [part])
    return " ".join(formatted_parts)
