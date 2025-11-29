import os
import lark
import dataclasses
import ipu_as.label as ipu_label

IPU_INSTR_ADDR_JUMP = 1


@dataclasses.dataclass
class AnnotatedToken:
    token: lark.Token
    instr_id: int

    def get_location_string(self) -> str:
        return f"Line {self.token.line}, Column {self.token.column}"


class ASTBuilder(lark.Transformer):
    def __init__(self):
        super().__init__()
        self.instr_index = 0

    def __default_token__(self, token: lark.Token):
        return AnnotatedToken(token=token, instr_id=self.instr_index)

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
    l1:
        add r1 r2 r3;
        mov rq3 rq4;
        sub r5 r6 r7; ;;
        
    l2:
        load rq8 100; beq lr1 lr2 l1; store rq9 200; nop; ;;

        mul r10 r11 r12;
        div r13 r14 r15;
        div rq16 rq17 rq18
    """
    ast = parse(code)

    print(ast)
    print(ipu_label.ipu_labels.labels)
