import os
import jinja2
from ipu_as import compound_inst, ipu_token, utils

WORD_WIDTH = 32

SRC_PATH = os.path.dirname(__file__)
TEMPLATE_DIR = os.path.join(SRC_PATH, "templates")
C_HEADER_TEMPLATE = jinja2.Template(
    open(os.path.join(TEMPLATE_DIR, "inst_parser.h.j2")).read()
)
C_TEMPLATE = jinja2.Template(
    open(os.path.join(TEMPLATE_DIR, "inst_parser.c.j2")).read()
)


def get_instruction_slots_info() -> list[dict]:
    """
    Get information about each instruction slot for disassembly.
    Returns a list of dicts with:
    - name: slot name (e.g., 'break_inst', 'xmem_inst')
    - prefix: field prefix in the struct
    - opcode_field: name of the opcode field
    - opcode_enum: name of the opcode enum type
    """
    slots = []
    inst_types_list = compound_inst.CompoundInst.instruction_types()
    
    # Count occurrences of each type to generate unique names
    type_counts = {}
    type_indices = {}
    
    for inst_type in inst_types_list:
        type_counts[inst_type] = type_counts.get(inst_type, 0) + 1
    
    for inst_type in inst_types_list:
        base_name = utils.camel_case_to_snake_case(inst_type.__name__)
        
        current_index = type_indices.get(inst_type, 0)
        type_indices[inst_type] = current_index + 1
        
        if type_counts[inst_type] > 1:
            prefix = f"{base_name}_{current_index}"
        else:
            prefix = base_name
        
        # Get opcode type name
        opcode_type = inst_type.opcode_type()
        opcode_enum = utils.camel_case_to_snake_case(opcode_type.__name__)
        opcode_field = f"{prefix}_token_0_{opcode_enum}"
        
        # Get operand fields
        operand_fields = []
        for i, token_type in enumerate(inst_type.operand_types()):
            field_name = f"{prefix}_token_{i+1}_{utils.camel_case_to_snake_case(token_type.__name__)}"
            operand_fields.append({
                'name': field_name,
                'type': utils.camel_case_to_snake_case(token_type.__name__)
            })
        
        slots.append({
            'name': base_name,
            'prefix': prefix,
            'opcode_field': opcode_field,
            'opcode_enum': opcode_enum,
            'operand_fields': operand_fields,
        })
    
    return slots


def expand_template_to_file(out_dir: str):
    fields = compound_inst.CompoundInst.get_fields()
    if compound_inst.CompoundInst.bits() % WORD_WIDTH != 0:
        fields.append(
            (
                "reserved",
                WORD_WIDTH - (compound_inst.CompoundInst.bits() % WORD_WIDTH),
            )
        )

    enums = ipu_token.EnumToken.get_all_enum_descriptors()
    slots = get_instruction_slots_info()

    rendered_h = C_HEADER_TEMPLATE.render(
        {
            "enums": enums,
            "inst_bit_fields": fields,
        }
    )
    rendered_c = C_TEMPLATE.render(
        {
            "inst_bit_fields": fields,
            "enums": enums,
            "slots": slots,
        }
    )

    with open(os.path.join(out_dir, "inst_parser.h"), "w") as f:
        f.write(rendered_h)
    with open(os.path.join(out_dir, "inst_parser.c"), "w") as f:
        f.write(rendered_c)
