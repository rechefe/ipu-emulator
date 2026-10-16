"""File and CLI entry points of the assembler, adapted to `AssemblySession`.

The pipeline lives in `ipu_as.session`; the parser and AST builder in
`ipu_as.syntax` (re-exported here for existing importers). These wrappers keep
their historical behaviour: a template error propagates as the jinja2 exception,
and a parse or assemble error prints a message and exits with status 1.
"""

import lark

import ipu_as.compound_inst as compound_inst
from ipu_as import session
from ipu_as.errors import AssemblerError, ParseError, TemplateError
from ipu_as.session import instruction_aligned_bytes_len
from ipu_as.syntax import ASTBuilder, get_parser, parse_tree

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


def _parse_failure(error: AssemblerError) -> lark.exceptions.LarkError | AssemblerError:
    """What the parse stage reported before the session: lark's own exception text."""
    cause = error.__cause__
    return cause if isinstance(cause, lark.exceptions.LarkError) else error


def parse(text: str) -> list[dict[str, any]]:
    """The compound instructions of `text`; prints and exits on a parse error."""
    try:
        return session.AssemblySession(text).parse()
    except TemplateError as error:
        # The jinja2 exception itself, as before the session existed.
        raise (error.__cause__ or error) from None
    except AssemblerError as error:
        if isinstance(error, ParseError) or isinstance(error.__cause__, lark.exceptions.LarkError):
            print(f"Error parsing code: {_parse_failure(error)}")
            exit(1)
        raise


def _assemble(text: str) -> session.AssemblyResult:
    try:
        return session.assemble(text)
    except TemplateError as error:
        # The jinja2 exception itself, as before the session existed.
        raise (error.__cause__ or error) from None
    except AssemblerError as error:
        if isinstance(error, ParseError) or isinstance(error.__cause__, lark.exceptions.LarkError):
            print(f"Error parsing code: {_parse_failure(error)}")
        else:
            print(f"Error assembling code: {error}")
        exit(1)
    except ValueError as error:
        print(f"Error assembling code: {error}")
        exit(1)


def assemble(text: str) -> list[int]:
    """The instruction words of `text`; prints and exits on a parse or assemble error."""
    return list(_assemble(text).words)


def assemble_to_mem_file(text: str, output_path: str):
    with open(output_path, "w") as f:
        f.write(_assemble(text).to_mem())


def assemble_to_bin_file(text: str, output_path: str):
    with open(output_path, "wb") as f:
        f.write(_assemble(text).to_bin())


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
