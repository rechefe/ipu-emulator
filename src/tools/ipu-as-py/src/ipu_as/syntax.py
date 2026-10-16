"""The assembly grammar: the parser, pseudo-instruction expansion and the AST.

It parses rendered text only; `ipu_as.session.AssemblySession` renders once
and then calls into this module.
"""

import functools
import os

import lark
from ipu_isa.instruction_spec import (
    INSTRUCTION_SPEC,
    PSEUDO_INSTRUCTION_SPEC,
    find_pseudo_instruction,
)

import ipu_as.ipu_token as ipu_token
from ipu_as.errors import OperandError
from ipu_as.symbols import SymbolTable

IPU_INSTR_ADDR_JUMP = 1


@functools.lru_cache(maxsize=1)
def get_parser() -> lark.Lark:
    """The single parser-construction site, shared by the assembler and tooling.

    Cached because building a LALR parser is not cheap and `parse()` is called
    once per assembled file; it used to be rebuilt on every call.
    """
    script_dir = os.path.dirname(__file__)
    return lark.Lark.open(
        os.path.join(script_dir, "asm_grammar.lark"), start="start", parser="lalr"
    )


def parse_tree(text: str) -> lark.Tree:
    """Parse already-rendered assembly text, **raising** `lark.exceptions.LarkError`.

    It never renders Jinja: rendering happens once, in `AssemblySession.render`.
    """
    return get_parser().parse(text)


def _is_real_instruction_name(name: str) -> bool:
    lowered = name.lower()
    return any(
        any(real_name.lower() == lowered for real_name in slot_instructions)
        for slot_instructions in INSTRUCTION_SPEC.values()
    )


def _expand_pseudo_instructions(instructions: list[dict]) -> list[dict]:
    """Expand pseudo-instructions into their real-instruction equivalents.

    Matched by (opcode name, operand count) via ``find_pseudo_instruction``,
    so a pseudo-instruction can share a name with a real instruction of a
    different arity without ambiguity. Pseudo-instructions never reach
    ``CompoundInst`` — by the time this returns, every instruction is a
    real one.
    """
    expanded = []
    for instr in instructions:
        opcode_token = instr["opcode"]
        operands = instr["operands"]
        opcode_name = opcode_token.token.value

        pseudo_def = find_pseudo_instruction(opcode_name, len(operands))
        if pseudo_def is None:
            if not _is_real_instruction_name(opcode_name):
                for pseudo_name, candidate in PSEUDO_INSTRUCTION_SPEC.items():
                    if pseudo_name.lower() == opcode_name.lower():
                        expected_operands = candidate["operands"]
                        names = ", ".join(op["name"] for op in expected_operands)
                        raise OperandError(
                            f"Pseudo-instruction '{pseudo_name}' expects "
                            f"{len(expected_operands)} operand(s) ({names}), "
                            f"got {len(operands)}\n"
                            f"At: {opcode_token.get_location_string()}",
                            location=opcode_token.location(),
                            token=opcode_name,
                        )
            expanded.append(instr)
            continue

        expansion = pseudo_def["expands_to"]
        operand_by_name = dict(
            zip((op["name"] for op in pseudo_def["operands"]), operands, strict=True)
        )
        # Borrow the position of the pseudo-instruction being expanded. A bare
        # lark.Token() carries no line or column, so any later error about the
        # expansion reported "Line None, Column None" and could not be placed —
        # `B loop;;` expanding to BEQ is invisible to the diagnostics.
        real_opcode = ipu_token.AnnotatedToken(
            token=lark.Token.new_borrow_pos(
                opcode_token.token.type, expansion["instruction"], opcode_token.token
            ),
            instr_id=opcode_token.instr_id,
            symbols=opcode_token.symbols,
        )
        real_operands = [
            operand_by_name.get(
                arg,
                ipu_token.AnnotatedToken(
                    token=lark.Token.new_borrow_pos(
                        opcode_token.token.type, arg, opcode_token.token
                    ),
                    instr_id=opcode_token.instr_id,
                    symbols=opcode_token.symbols,
                ),
            )
            for arg in expansion["args"]
        ]
        expanded.append({"opcode": real_opcode, "operands": real_operands})
    return expanded


class ASTBuilder(lark.Transformer):
    """Builds the compound-instruction list and defines each label in `symbols`.

    Every token it makes carries `symbols`, so label operands resolve against
    this assembly's table and no other.
    """

    def __init__(self, symbols: SymbolTable | None = None):
        super().__init__()
        self.instr_index = 0
        self.symbols = SymbolTable() if symbols is None else symbols

    def __default_token__(self, token: lark.Token):
        return ipu_token.AnnotatedToken(
            token=token, instr_id=self.instr_index, symbols=self.symbols
        )

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
                self.symbols.define(label)
            items = items[1:]

        for it in items:
            if isinstance(it, list):
                instructions.extend(it)
            else:
                instructions.append(it)

        instructions = _expand_pseudo_instructions(instructions)

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
