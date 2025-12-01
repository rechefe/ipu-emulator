import jinja2
from ipu_as import compound_inst, ipu_token, opcodes


def expand_template():
    template_loader = jinja2.FileSystemLoader(searchpath="src/ipu_as/templates")
    template_env = jinja2.Environment(loader=template_loader)

    template_file = "inst_parser.h.j2"
    template = template_env.get_template(template_file)

    rendered_code = template.render(
        {
            "enums": ipu_token.EnumToken.get_all_enum_descriptors(),
            "inst_bit_fields": compound_inst.CompoundInst.get_fields(),
        }
    )

    with open("generated/inst_parser.h", "w") as f:
        f.write(rendered_code)

    for line in compound_inst.CompoundInst.desc():
        print(line)


if __name__ == "__main__":
    expand_template()
