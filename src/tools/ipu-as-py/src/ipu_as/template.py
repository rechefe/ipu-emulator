from . import compound_inst
import jinja2


def expand_template():
    template_loader = jinja2.FileSystemLoader(searchpath="src/ipu_as/templates")
    template_env = jinja2.Environment(loader=template_loader)

    template_file = "inst_parser.h.j2"
    template = template_env.get_template(template_file)

    rendered_code = template.render(
        {
            "enums": compound_inst.get_enums(),
            "inst_bit_fields": compound_inst.get_fields(),
        }
    )

    with open("generated/inst_parser.h", "w") as f:
        f.write(rendered_code)


if __name__ == "__main__":
    expand_template()
