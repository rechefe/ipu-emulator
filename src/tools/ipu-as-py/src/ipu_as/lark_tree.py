import os
import lark
import ipu_as.label as ipu_label
import ipu_as.compound_inst as compound_inst
import ipu_as.ipu_token as ipu_token

IPU_INSTR_ADDR_JUMP = 1


class ASTBuilder(lark.Transformer):
    def __init__(self):
        super().__init__()
        self.instr_index = 0

    def __default_token__(self, token: lark.Token):
        return ipu_token.AnnotatedToken(token=token, instr_id=self.instr_index)

    def program(self, items):
        return items

    def compound_instr_list(self, items):
        return self.compound_instruction(items)

    def compound_instruction(self, items):
        label = None
        instructions = []

        if items and isinstance(items[0], dict) and items[0].get("type") == "label":
            label = items[0]["name"]
            if label:
                ipu_label.ipu_labels.add_label(label)
            items = items[1:]

        for it in items:
            if isinstance(it, list):
                instructions.extend(it)
            else:
                instructions.append(it)

        self.instr_index += IPU_INSTR_ADDR_JUMP
        return {
            "label": label,
            "instructions": instructions,
        }

    def instr_list(self, items):
        return items

    def instruction(self, items):
        opcode = items[0]
        operands = [tok for tok in items[1:]]
        return {"opcode": opcode, "operands": operands}

    def label(self, items):
        return {"type": "label", "name": items[0]}

    def operand(self, items):
        return items[0]


def parse(text):
    script_dir = os.path.dirname(__file__)
    parser = lark.Lark.open(
        os.path.join(script_dir, "asm_grammar.lark"), start="start", parser="lalr"
    )

    tree = parser.parse(text)
    ast = ASTBuilder().transform(tree)
    return ast


if __name__ == "__main__":
    code = """
start:
    beq lr0 lr15 end;;
    
    
end:
    bkpt;
    """
    ast = parse(code)

    program = [compound_inst.CompoundInst(instr) for instr in ast]

    for inst in program:
        print(inst.encode())
