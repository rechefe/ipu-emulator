import os
import jinja2
from ipu_as import compound_inst, ipu_token

WORD_WIDTH = 32

SRC_PATH = os.path.dirname(__file__)
TEMPLATE_DIR = os.path.join(SRC_PATH, "templates")
C_HEADER_TEMPLATE = jinja2.Template(
    open(os.path.join(TEMPLATE_DIR, "inst_parser.h.j2")).read()
)
C_TEMPLATE = jinja2.Template(
    open(os.path.join(TEMPLATE_DIR, "inst_parser.h.j2")).read()
)


def expand_template_to_file(out_dir: str):
    fields = compound_inst.CompoundInst.get_fields()
    if compound_inst.CompoundInst.bits() % WORD_WIDTH != 0:
        fields.append(
            (
                "reserved",
                WORD_WIDTH - (compound_inst.CompoundInst.bits() % WORD_WIDTH),
            )
        )

    rendered_h = C_HEADER_TEMPLATE.render(
        {
            "enums": ipu_token.EnumToken.get_all_enum_descriptors(),
            "inst_bit_fields": fields,
        }
    )
    rendered_c = C_TEMPLATE.render(
        {
            "inst_bit_fields": fields,
        }
    )

    with open(os.path.join(out_dir, "inst_parser.h"), "w") as f:
        f.write(rendered_h)
    with open(os.path.join(out_dir, "inst_parser.c"), "w") as f:
        f.write(rendered_c)
