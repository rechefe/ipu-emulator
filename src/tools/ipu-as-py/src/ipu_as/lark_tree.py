import os
import lark
import ipu_as.compound_inst as compound_inst
import ipu_as.ipu_token as ipu_token
import ipu_as.label as ipu_label

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


def parse(text: str) -> list[dict[str, any]]:
    script_dir = os.path.dirname(__file__)
    parser = lark.Lark.open(
        os.path.join(script_dir, "asm_grammar.lark"), start="start", parser="lalr"
    )

    try:
        tree = parser.parse(text)
        ast = ASTBuilder().transform(tree)
        return ast
    except lark.exceptions.LarkError as e:
        print(f"Error parsing code: {e}")
        exit(1)


def assemble(text: str) -> list[int]:
    ast = parse(text)
    try:
        program = [compound_inst.CompoundInst(instr).encode() for instr in ast]
        return program
    except ValueError as e:
        print(f"Error assembling code: {e}")
        exit(1)


def assemble_to_mem_file(text: str, output_path: str):
    program = assemble(text)
    with open(output_path, "w") as f:
        for word in program:
            f.write(f"0x0{word:08x}\n")


BYTE_SIZE = 8


def instruction_aligned_bytes_len() -> int:
    val = compound_inst.CompoundInst.bits() // BYTE_SIZE
    if val % BYTE_SIZE != 0:
        val += 1
    return val


def assemble_to_bin_file(text: str, output_path: str):
    program = assemble(text)
    with open(output_path, "wb") as f:
        for word in program:
            f.write(
                word.to_bytes(
                    compound_inst.CompoundInst.bits()
                    // instruction_aligned_bytes_len(),
                    byteorder="little",
                )
            )


def disassemble(program: list[int]) -> list[str]:
    disassembled_instructions = []
    for i, encoded_inst in enumerate(program):
        disassembled_instructions.append(
            f"{i:04x}:\t\t{compound_inst.CompoundInst.decode(encoded_inst)}"
        )
    return disassembled_instructions


def disassemble_from_mem_file(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        program = [int(line.strip(), 0) for line in f.readlines()]
    with open(output_path, "w") as f:
        for line in disassemble(program):
            f.write(f"{line}\n")


def disassemble_from_bin_file(input_path: str, output_path: str):
    with open(input_path, "rb") as f:
        byte_data = f.read()
        instruction_size_bytes = instruction_aligned_bytes_len()
        program = []
        for i in range(0, len(byte_data), instruction_size_bytes):
            word_bytes = byte_data[i : i + instruction_size_bytes]
            word_int = int.from_bytes(word_bytes, byteorder="little")
            program.append(word_int)
    with open(output_path, "w") as f:
        for line in disassemble(program):
            f.write(f"{line}\n")


if __name__ == "__main__":
    code = """
# The provided code snippet is a simple assembly-like code written in a custom language. Here's a
# breakdown of what it does:
start:
    beq lr13 lr15 end; 
    mac.ee rq4 r1 r2;
    ldr lr3 cr9;                     ;;
    
    
end:
    mac.ev rq8 r5 r9 lr0; 
    beq lr0 lr1 +2;
    incr lr2 15;
    ;;
    
    b start; mac.ev rq0 r3 mem_bypass lr15; ;;
    """

    description = compound_inst.CompoundInst.desc()
    print("\n".join(description))
    assembled = assemble(code)
    for line in assembled:
        print(hex(line))

    for line in disassemble(assembled):
        print(line)
