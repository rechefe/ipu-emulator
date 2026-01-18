import ipu_as.inst as inst
import ipu_as.utils as utils


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
        return address

    def _fill_out_nop(self, address: int):
        inst_types_list = self.instruction_types()
        for i, inst_type in enumerate(inst_types_list):
            if self.instructions[i] is None:
                self.instructions[i] = inst_type.nop_inst(address)

    @classmethod
    def instruction_types(cls) -> list[type[inst.Inst]]:
        # Define the instruction slots in order (with duplicates for multiple instances)
        # Order matters for encoding/decoding
        return [
            inst.XmemInst,
            inst.MultInst,
            inst.AccInst,
            inst.LrInst,
            inst.LrInst,  # Second LrInst slot
            inst.CondInst,
        ]

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
    def generate_fields_svg(cls) -> str:
        """
        Generate an SVG visualization of instruction fields organized in 32-bit rows.
        Shows bits from bottom to top: Row 0 (31:0), Row 1 (63:32), Row 2 (95:64), Row 3 (127:96)
        Each instruction type gets a unique color.
        """
        # Define colors for each instruction type
        color_map = {
            inst.XmemInst: "#FF6B6B",     # Red
            inst.MultInst: "#4ECDC4",     # Teal
            inst.AccInst: "#45B7D1",      # Blue
            inst.LrInst: "#FFA07A",       # Light Salmon
            inst.CondInst: "#98D8C8",     # Mint
        }

        inst_types_list = cls.instruction_types()
        
        # Build list of fields with their bit positions
        fields_with_positions = []
        current_bit = 0
        
        for inst_type in inst_types_list:
            all_tokens = inst_type.all_tokens()
            for token_type in all_tokens:
                token_bits = token_type.bits()
                token_name = utils.camel_case_to_snake_case(token_type.__name__)
                display_name = _smart_abbreviate_field_name(token_name, 0)
                
                fields_with_positions.append({
                    "name": display_name,
                    "bits": token_bits,
                    "start_bit": current_bit,
                    "end_bit": current_bit + token_bits - 1,
                    "color": color_map.get(inst_type, "#CCCCCC"),
                    "token_type": inst_type.__name__,
                })
                current_bit += token_bits
        
        # SVG dimensions and styling
        bits_per_width = 24  # pixels per bit (doubled for wider layout)
        row_height = 80  # Increased to fit multiple text lines
        padding = 20
        row_width = 32 * bits_per_width  # 32 bits per row
        font_size_title = 14
        font_size_label = 8
        font_size_bits = 7
        
        total_bits = cls.bits()
        num_rows = (total_bits + 31) // 32  # Round up to nearest 32
        
        # Calculate height with space for color legend
        legend_height = 100  # Space for color legend below
        svg_width = row_width + padding * 2 + 80  # Extra space for bit labels
        svg_height = (num_rows * row_height) + font_size_title * 4 + padding * 3 + legend_height

        # Start building SVG
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
        ]

        # Add title
        svg_lines.append(
            f'  <text x="{svg_width/2}" y="{padding + font_size_title}" '
            f'text-anchor="middle" class="inst-title">'
            f'{cls.__name__} Layout - {cls.bits()} total bits</text>'
        )

        # Draw rows from top to bottom (highest bits first)
        for row_idx in range(num_rows - 1, -1, -1):
            row_start_bit = row_idx * 32
            row_end_bit = min(row_start_bit + 31, total_bits - 1)
            display_row_idx = num_rows - 1 - row_idx
            
            current_y = padding + font_size_title * 2.5 + (display_row_idx * row_height)
            current_x = padding + 60
            
            # Draw row background and label
            svg_lines.append(
                f'  <rect x="{current_x}" y="{current_y}" '
                f'width="{row_width}" height="{row_height}" '
                f'fill="white" stroke="black" stroke-width="2"/>'
            )
            
            # Add bit range label on the left
            svg_lines.append(
                f'  <text x="{padding + 30}" y="{current_y + row_height/2 + font_size_label/2}" '
                f'text-anchor="end" class="bit-range">[{row_end_bit}:{row_start_bit}]</text>'
            )
            
            # Add bit position labels across the top (right-to-left: high bits on left)
            bit_label_y = current_y + 12
            
            # For partial rows, shift them to the right to align with full rows
            bits_in_row = row_end_bit - row_start_bit + 1
            row_offset = (32 - bits_in_row) * bits_per_width
            
            for bit_pos in range(row_end_bit, row_start_bit - 1, -1):
                # Centered positioning: center the number in each bit's space, right-aligned with offset
                bit_x = current_x + row_offset + (row_end_bit - bit_pos + 0.5) * bits_per_width
                svg_lines.append(
                    f'  <text x="{bit_x}" y="{bit_label_y}" text-anchor="middle" '
                    f'class="field-label" style="font-size: 6px; fill: #000000; font-weight: bold;">{bit_pos}</text>'
                )
            
            # Draw fields in this row
            for field in fields_with_positions:
                if field["start_bit"] <= row_end_bit and field["end_bit"] >= row_start_bit:
                    # Calculate position within row (right-to-left: high bits on left, low bits on right)
                    field_start = max(field["start_bit"], row_start_bit)
                    field_end = min(field["end_bit"], row_end_bit)
                    
                    # Reverse positioning with consistent bits_per_width and row offset for partial rows
                    field_width_px = (field_end - field_start + 1) * bits_per_width
                    field_x = current_x + row_offset + (row_end_bit - field_end) * bits_per_width
                    
                    # Draw field block
                    svg_lines.append(
                        f'  <rect x="{field_x}" y="{current_y}" '
                        f'width="{field_width_px}" height="{row_height}" '
                        f'fill="{field["color"]}" stroke="black" stroke-width="1" opacity="0.85"/>'
                    )
                    
                    # Add field label with text wrapping support
                    field_center_x = field_x + field_width_px / 2
                    field_center_y = current_y + row_height / 2
                    
                    # Split field name into words - put one word per line for consistency
                    words = field["name"].split()
                    text_lines = []
                    
                    # Create consistent wrapping: one word per line
                    for word in words:
                        # Further split long words (e.g., "stage_reg_r" is already split, but handle others)
                        if len(word) > 10:
                            # Abbreviate very long words
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
                    
                    # Render text lines centered vertically
                    line_height = 11
                    total_text_height = len(text_lines) * line_height
                    start_y = field_center_y - total_text_height / 2
                    
                    for i, line in enumerate(text_lines):
                        svg_lines.append(
                            f'  <text x="{field_center_x}" y="{start_y + i * line_height}" '
                            f'text-anchor="middle" class="field-label">'
                            f'{line}</text>'
                        )
                    
                    # Add bit range label at bottom
                    bit_label_y = field_center_y + total_text_height / 2 + 3
                    svg_lines.append(
                        f'  <text x="{field_center_x}" y="{bit_label_y}" '
                        f'text-anchor="middle" class="field-label" style="font-weight: bold;">'
                        f'[{field["end_bit"]}:{field["start_bit"]}]</text>'
                    )
        
        # Add color legend at the bottom of the SVG
        legend_y = padding + font_size_title * 2.5 + (num_rows * row_height) + 20
        svg_lines.append(f'  <text x="{padding + 60}" y="{legend_y}" class="inst-label" style="font-weight: bold;">Instruction Type Colors:</text>')
        
        # Color legend items
        color_legend = [
            ("#FF6B6B", "XmemInst (Extended Memory)"),
            ("#4ECDC4", "MultInst (Multiply)"),
            ("#45B7D1", "AccInst (Accumulator)"),
            ("#FFA07A", "LrInst (Link Register)"),
            ("#98D8C8", "CondInst (Conditional)"),
        ]
        
        legend_item_y = legend_y + 15
        legend_item_height = 15
        color_box_size = 12
        
        for i, (color, label) in enumerate(color_legend):
            x_pos = padding + 60
            y_pos = legend_item_y + (i * legend_item_height)
            
            # Draw color box
            svg_lines.append(
                f'  <rect x="{x_pos}" y="{y_pos - color_box_size + 2}" '
                f'width="{color_box_size}" height="{color_box_size}" '
                f'fill="{color}" stroke="black" stroke-width="1" opacity="0.85"/>'
            )
            
            # Add label
            svg_lines.append(
                f'  <text x="{x_pos + color_box_size + 8}" y="{y_pos + 2}" '
                f'class="field-label" style="font-size: 9px;">{label}</text>'
            )
        
        svg_lines.append('</svg>')

        return '\n'.join(svg_lines)


def _smart_abbreviate_field_name(name: str, available_width_px: float) -> str:
    """
    Format field names for display by converting class names to readable labels.
    For example: AccInstOpcode -> Acc Inst Opcode
    """
    # Remove "Field" suffix if present
    if name.endswith("_field"):
        name = name[:-6]
    if name.endswith("_type"):
        name = name[:-5]
    
    # Convert snake_case to Title Case
    # First handle the case where we have camelCase embedded in snake_case
    parts = name.split('_')
    formatted_parts = []
    
    for part in parts:
        # Split camelCase into separate words
        import re
        # Insert space before uppercase letters (camelCase handling)
        words = re.findall('[A-Z][a-z]*|[a-z]+', part)
        if words:
            formatted_parts.extend(words)
        else:
            formatted_parts.append(part)
    
    # Join with spaces and title case
    display_name = ' '.join(formatted_parts)
    
    return display_name


def _create_multiline_text_elements(
    text: str, center_x: float, start_y: float, line_height: int, font_size: int, 
    max_width_px: float, bold: bool = False
) -> list[str]:
    """
    Create multiple text elements for a string that needs to wrap across multiple lines.
    Returns list of SVG text element strings.
    """
    # Estimate characters per line (roughly 6-7 pixels per character at 6px font)
    chars_per_line = max(2, int(max_width_px / 5.5))
    
    # Split text into words
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        if len(test_line) <= chars_per_line:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # If no space split worked, try splitting by underscore
    if len(lines) == 1 and '_' in text:
        words = text.split('_')
        lines = []
        current_line = []
        for word in words:
            test_line = '_'.join(current_line + [word])
            if len(test_line) <= chars_per_line:
                current_line.append(word)
            else:
                if current_line:
                    lines.append('_'.join(current_line))
                current_line = [word]
        if current_line:
            lines.append('_'.join(current_line))
    
    # Create text elements for each line - use small font
    text_elements = []
    for i, line in enumerate(lines):
        y_pos = start_y + (i * line_height)
        bold_style = "font-weight: bold;" if bold else ""
        text_elements.append(
            f'  <text x="{center_x}" y="{y_pos}" text-anchor="middle" '
            f'class="field-name" style="font-size: 6px; {bold_style}">{line}</text>'
        )
    
    return text_elements
