"""File entry points of the assembler, over `ipu_as.session`.

The pipeline lives in `ipu_as.session`, and the parser and AST builder in
`ipu_as.syntax` (re-exported here for existing importers). These functions
never print or exit: a bad source raises an `AssemblerError` subclass. Only
`ipu_as.cli` turns errors into text and exit codes.
"""

import ipu_as.compound_inst as compound_inst
from ipu_as import session
from ipu_as.session import instruction_aligned_bytes_len
from ipu_as.syntax import ASTBuilder, get_parser, parse_tree
from ipu_isa.codec.word import unpack_words

__all__ = [
    "ASTBuilder",
    "assemble",
    "assemble_to_bin_file",
    "assemble_to_mem_file",
    "disassemble",
    "disassemble_from_bin_file",
    "disassemble_from_mem_file",
    "get_parser",
    "instruction_aligned_bytes_len",
    "parse",
    "parse_tree",
]


def parse(text: str) -> list[dict]:
    """The compound instructions of `text`. Raises an AssemblerError subclass."""
    return session.AssemblySession(text).parse()


def assemble(text: str) -> list[int]:
    """The instruction words of `text`. Raises an AssemblerError subclass."""
    return list(session.assemble(text).words)


def assemble_to_mem_file(text: str, output_path: str):
    result = session.assemble(text)
    with open(output_path, "w") as f:
        f.write(result.to_mem())


def assemble_to_bin_file(text: str, output_path: str):
    result = session.assemble(text)
    with open(output_path, "wb") as f:
        f.write(result.to_bin())


def disassemble(program: list[int]) -> list[str]:
    disassembled_instructions = []
    for i, encoded_inst in enumerate(program):
        disassembled_instructions.append(
            f"{i:04x}:\t\t{compound_inst.CompoundInst.decode(encoded_inst)}"
        )
    return disassembled_instructions


def disassemble_from_mem_file(input_path: str, output_path: str):
    with open(input_path) as f:
        program = [int(line.strip(), 0) for line in f.readlines()]
    with open(output_path, "w") as f:
        for line in disassemble(program):
            f.write(f"{line}\n")


def disassemble_from_bin_file(input_path: str, output_path: str):
    with open(input_path, "rb") as f:
        program = unpack_words(f.read())
    with open(output_path, "w") as f:
        for line in disassemble(program):
            f.write(f"{line}\n")
