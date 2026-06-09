import warnings

import ipu_as.inst as inst
import ipu_as.utils as utils
from ipu_common.instruction_spec import (
    COMPOUND_LAYOUT_SLOT_ORDER,
    SLOT_COUNT,
    SLOT_UNIONS,
    is_hardware_slot,
)
from ipu_common.union_layout_svg import render_union_layout_svg


INST_SEPARATOR = ";"


class CompoundInst:
    def __init__(
        self,
        instructions: list[dict[str, any]],
    ):
        # instructions is now a list ordered by instruction_types()
        self.instructions = [None] * len(self.instruction_types())
        self._fill_out_nop(self._fill_instructions(instructions))

    def _fill_instructions(self, instructions: list[dict[str, any]]) -> int:
        address = None
        inst_types_list = self.instruction_types()

        for instruction in instructions["instructions"]:
            inst_type = inst.Inst.find_inst_type_by_opcode(
                instruction["opcode"].token.value
            )
            if address is None:
                address = instruction["opcode"].instr_id

            # Find the first available slot for this instruction type
            slot_filled = False
            for i, expected_type in enumerate(inst_types_list):
                if expected_type == inst_type and self.instructions[i] is None:
                    self.instructions[i] = inst_type(instruction)
                    slot_filled = True
                    break

            if not slot_filled:
                # No available slot - either all slots are filled or no slot exists
                available_slots = sum(1 for t in inst_types_list if t == inst_type)

                if available_slots == 0:
                    raise ValueError(
                        f"Instruction type {inst_type.__name__} is not allowed in compound instruction\n"
                        f"At: {instruction['opcode'].get_location_string()}"
                    )
                else:
                    raise ValueError(
                        f"Too many instructions of type {inst_type.__name__} (max {available_slots})\n"
                        f"At: {instruction['opcode'].get_location_string()}"
                    )

            if not is_hardware_slot(inst_type._slot_type_name()):
                warnings.warn(
                    f"{instruction['opcode'].token.value} uses the simulation-only "
                    f"{inst_type._slot_type_name()} slot (not implemented in real IPU hardware)",
                    stacklevel=4,
                )
        return address

    def _fill_out_nop(self, address: int):
        inst_types_list = self.instruction_types()
        for i, inst_type in enumerate(inst_types_list):
            if self.instructions[i] is None:
                self.instructions[i] = inst_type.nop_inst(address)

    @classmethod
    def instruction_types(cls) -> list[type[inst.Inst]]:
        # Instruction slots in MSB → LSB list order (encode places the last entry at LSB).
        # Execution order (break first, then LR, …) is handled in ipu.py, not here.
        # Slot counts come from SLOT_COUNT in instruction_spec (single source of truth).
        _slot_to_inst = {
            "cond": inst.CondInst,
            "lr": inst.LrInst,
            "load": inst.LoadInst,
            "store": inst.StoreInst,
            "acc_store": inst.AccStoreInst,
            "mult": inst.MultInst,
            "acc": inst.AccInst,
            "aaq": inst.AaqInst,
            "break": inst.BreakInst,
        }
        result = []
        for slot in COMPOUND_LAYOUT_SLOT_ORDER:
            result.extend([_slot_to_inst[slot]] * SLOT_COUNT[slot])
        return result

    @classmethod
    def bits(cls) -> int:
        return sum(instruction.bits() for instruction in cls.instruction_types())

    def encode(self) -> int:
        encoded_line = 0
        shift_amount = 0
        # Iterate through instructions in reverse order to maintain correct bit order
        for instruction in reversed(self.instructions):
            encoded_inst = instruction.encode()
            encoded_line |= encoded_inst << shift_amount
            shift_amount += instruction.bits()
        return encoded_line

    @classmethod
    def decode(cls, value: int) -> str:
        decoded_instructions = []
        shift_amount = 0
        for inst_type in reversed(cls.instruction_types()):
            inst_bits = inst_type.bits()
            inst_value = (value >> shift_amount) & ((1 << inst_bits) - 1)
            decoded_instructions.append(inst_type.decode(inst_value))
            shift_amount += inst_bits
        return ";\n\t\t\t".join(reversed(decoded_instructions)) + ";;"

    @classmethod
    def desc(cls) -> list[str]:
        res = []
        res.append(f"{cls.__name__} - {cls.bits()} bits:")
        res.append("Subitems:")
        for inst_type in cls.instruction_types():
            res.extend(f"\t{line}" for line in inst_type.desc())
        return res

    @classmethod
    def get_fields(cls) -> list[tuple[str, int]]:
        fields = []
        inst_types_list = cls.instruction_types()

        # Count occurrences of each type to generate unique names
        type_counts = {}
        type_indices = {}

        for inst_type in inst_types_list:
            type_counts[inst_type] = type_counts.get(inst_type, 0) + 1

        for inst_type in inst_types_list:
            # Determine the prefix for field names
            base_name = utils.camel_case_to_snake_case(inst_type.__name__)

            # Track which instance of this type we're on
            current_index = type_indices.get(inst_type, 0)
            type_indices[inst_type] = current_index + 1

            # If there are multiple instances of this type, add index to the name
            if type_counts[inst_type] > 1:
                inst_prefix = f"{base_name}_{current_index}"
            else:
                inst_prefix = base_name

            for i, token_type in enumerate(inst_type.all_tokens()):
                field_name = (
                    f"{inst_prefix}"
                    f"_token_{i}_{utils.camel_case_to_snake_case(token_type.__name__)}"
                )
                fields.append((field_name, token_type.bits()))
        return list(reversed(fields))

    @classmethod
    def generate_union_layout_svg(cls) -> str:
        """Render the per-slot union field layout as inline SVG.

        Slot order matches the binary layout (MSB → LSB) so the picture lines
        up with what an encoded VLIW word looks like in memory.  Multiplicity
        (e.g. ``lr`` ×3) is annotated in each slot's title.
        """
        seen: list[str] = []
        slot_counts: dict[str, int] = {}
        for inst_type in cls.instruction_types():
            slot = inst_type._slot_type_name()
            if slot not in slot_counts:
                seen.append(slot)
            slot_counts[slot] = slot_counts.get(slot, 0) + 1
        return render_union_layout_svg(
            SLOT_UNIONS,
            slot_order=seen,
            slot_counts=slot_counts,
        )

    @classmethod
    def generate_struct_layout_svg(cls) -> str:
        """Render the full compound-instruction word as bit rows of 32.

        Shows every operand token in its actual bit position, coloured by the
        owning slot.  Complements ``generate_union_layout_svg`` by giving the
        whole-word picture (what an encoded VLIW word looks like in memory)
        without folding cross-opcode field sharing.
        """
        legend_entries = [
            (inst.BreakInst, "#FFD93D", "BreakInst (Break / Debug)"),
            (inst.LoadInst, "#FF6B6B", "LoadInst (Memory Load)"),
            (inst.StoreInst, "#E74C3C", "StoreInst (Memory Store)"),
            (inst.AccStoreInst, "#C0392B", "AccStoreInst (STR_ACC_REG, simulation-only)"),
            (inst.MultInst, "#4ECDC4", "MultInst (Multiply)"),
            (inst.AccInst, "#45B7D1", "AccInst (Accumulator)"),
            (inst.AaqInst, "#9B59B6", "AaqInst (Activation and Quantization)"),
            (inst.LrInst, "#FFA07A", "LrInst (Link Register)"),
            (inst.CondInst, "#98D8C8", "CondInst (Conditional)"),
        ]
        color_map = {inst_type: color for inst_type, color, _ in legend_entries}

        inst_types_list = cls.instruction_types()

        fields_with_positions = []
        current_bit = 0
        for inst_type in reversed(inst_types_list):
            for token_type in reversed(inst_type.all_tokens()):
                token_bits = token_type.bits()
                token_name = utils.camel_case_to_snake_case(token_type.__name__)
                fields_with_positions.append({
                    "name": _smart_abbreviate_field_name(token_name),
                    "bits": token_bits,
                    "start_bit": current_bit,
                    "end_bit": current_bit + token_bits - 1,
                    "color": color_map.get(inst_type, "#CCCCCC"),
                })
                current_bit += token_bits

        bits_per_width = 24
        row_height = 80
        padding = 20
        row_width = 32 * bits_per_width
        font_size_title = 14
        font_size_label = 8
        font_size_bits = 7
        total_bits = cls.bits()
        num_rows = (total_bits + 31) // 32

        legend_height = 20 + len(legend_entries) * 15
        svg_width = row_width + padding * 2 + 80
        svg_height = (num_rows * row_height) + font_size_title * 4 + padding * 3 + legend_height

        svg_lines = [
            f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">',
            '  <defs>',
            '    <style>',
            f'      .inst-title {{ font-size: {font_size_title}px; font-weight: bold; font-family: Arial; }}',
            f'      .inst-label {{ font-size: {font_size_label}px; font-weight: bold; font-family: Arial; }}',
            f'      .field-label {{ font-size: {font_size_bits}px; font-family: Arial; }}',
            f'      .bit-range {{ font-size: {font_size_label}px; font-family: Arial; }}',
            '    </style>',
            '  </defs>',
            f'  <rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="white" stroke="none"/>',
            f'  <text x="{svg_width/2}" y="{padding + font_size_title}" '
            f'text-anchor="middle" class="inst-title">'
            f'{cls.__name__} Layout - {cls.bits()} total bits</text>',
        ]

        for row_idx in range(num_rows - 1, -1, -1):
            row_start_bit = row_idx * 32
            row_end_bit = min(row_start_bit + 31, total_bits - 1)
            display_row_idx = num_rows - 1 - row_idx
            current_y = padding + font_size_title * 2.5 + (display_row_idx * row_height)
            current_x = padding + 60
            svg_lines.append(
                f'  <rect x="{current_x}" y="{current_y}" '
                f'width="{row_width}" height="{row_height}" '
                f'fill="white" stroke="black" stroke-width="2"/>'
            )
            svg_lines.append(
                f'  <text x="{padding + 30}" y="{current_y + row_height/2 + font_size_label/2}" '
                f'text-anchor="end" class="bit-range">[{row_end_bit}:{row_start_bit}]</text>'
            )

            bit_label_y = current_y + 12
            bits_in_row = row_end_bit - row_start_bit + 1
            row_offset = (32 - bits_in_row) * bits_per_width
            for bit_pos in range(row_end_bit, row_start_bit - 1, -1):
                bit_x = current_x + row_offset + (row_end_bit - bit_pos + 0.5) * bits_per_width
                svg_lines.append(
                    f'  <text x="{bit_x}" y="{bit_label_y}" text-anchor="middle" '
                    f'class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">{bit_pos}</text>'
                )

            for field in fields_with_positions:
                if field["start_bit"] > row_end_bit or field["end_bit"] < row_start_bit:
                    continue
                field_start = max(field["start_bit"], row_start_bit)
                field_end = min(field["end_bit"], row_end_bit)
                field_width_px = (field_end - field_start + 1) * bits_per_width
                field_x = current_x + row_offset + (row_end_bit - field_end) * bits_per_width
                svg_lines.append(
                    f'  <rect x="{field_x}" y="{current_y}" '
                    f'width="{field_width_px}" height="{row_height}" '
                    f'fill="{field["color"]}" stroke="black" stroke-width="1" opacity="0.85"/>'
                )
                field_center_x = field_x + field_width_px / 2
                field_center_y = current_y + row_height / 2

                text_lines = []
                for word in field["name"].split():
                    if len(word) > 10:
                        if word.startswith('immediate'):
                            text_lines.append('imm')
                        elif word.startswith('opcode'):
                            text_lines.append('op')
                        elif word.startswith('stage'):
                            text_lines.append('stg')
                        else:
                            text_lines.append(word[:5])
                    else:
                        text_lines.append(word)

                line_height = 11
                total_text_height = len(text_lines) * line_height
                start_y = field_center_y - total_text_height / 2
                for i, line in enumerate(text_lines):
                    svg_lines.append(
                        f'  <text x="{field_center_x}" y="{start_y + i * line_height}" '
                        f'text-anchor="middle" class="field-label">{line}</text>'
                    )
                svg_lines.append(
                    f'  <text x="{field_center_x}" y="{field_center_y + total_text_height / 2 + 3}" '
                    f'text-anchor="middle" class="field-label" style="font-weight: bold;">'
                    f'[{field["end_bit"]}:{field["start_bit"]}]</text>'
                )

        legend_y = padding + font_size_title * 2.5 + (num_rows * row_height) + 20
        svg_lines.append(
            f'  <text x="{padding + 60}" y="{legend_y}" class="inst-label" '
            f'style="font-weight: bold;">Instruction Type Colors:</text>'
        )
        color_box_size = 12
        for i, (_, color, label) in enumerate(legend_entries):
            x_pos = padding + 60
            y_pos = legend_y + 15 + i * 15
            svg_lines.append(
                f'  <rect x="{x_pos}" y="{y_pos - color_box_size + 2}" '
                f'width="{color_box_size}" height="{color_box_size}" '
                f'fill="{color}" stroke="black" stroke-width="1" opacity="0.85"/>'
            )
            svg_lines.append(
                f'  <text x="{x_pos + color_box_size + 8}" y="{y_pos + 2}" '
                f'class="field-label" style="font-size: 9px;">{label}</text>'
            )

        svg_lines.append('</svg>')
        return '\n'.join(svg_lines)


def _smart_abbreviate_field_name(name: str) -> str:
    """Convert a snake_case token-class name into a readable space-separated label."""
    import re

    if name.endswith("_field"):
        name = name[:-6]
    if name.endswith("_type"):
        name = name[:-5]
    formatted_parts: list[str] = []
    for part in name.split('_'):
        words = re.findall('[A-Z][a-z]*|[a-z]+', part)
        formatted_parts.extend(words if words else [part])
    return ' '.join(formatted_parts)
