"""File and CLI entry points of the assembler, adapted to `AssemblySession`.

The pipeline lives in `ipu_as.session`; the parser and AST builder in
`ipu_as.syntax` (re-exported here for existing importers). These wrappers keep
their historical behaviour: a template error propagates as the jinja2 exception,
and a parse or assemble error prints a message and exits with status 1.
"""

from typing import NoReturn

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


def _report_and_exit(error: AssemblerError) -> NoReturn:
    """Print the line each stage printed before the session existed, and exit 1.

    The session chains every parse-stage error to the lark exception it came
    from (`raise ... from error`); that exception's text is what was printed.
    """
    cause = error.__cause__
    if isinstance(error, ParseError) or isinstance(cause, lark.exceptions.LarkError):
        reported = cause if isinstance(cause, lark.exceptions.LarkError) else error
        print(f"Error parsing code: {reported}")
    else:
        print(f"Error assembling code: {error}")
    exit(1)


def _jinja_error(error: TemplateError) -> BaseException:
    """The jinja2 exception a TemplateError wraps, which is what callers got before."""
    return error.__cause__ or error


def parse(text: str) -> list[dict]:
    """The compound instructions of `text`; prints and exits on a parse error."""
    try:
        return session.AssemblySession(text).parse()
    except TemplateError as error:
        raise _jinja_error(error) from None
    except AssemblerError as error:
        if isinstance(error, ParseError) or isinstance(error.__cause__, lark.exceptions.LarkError):
            _report_and_exit(error)
        raise


def _assemble(text: str) -> session.AssemblyResult:
    try:
        return session.assemble(text)
    except TemplateError as error:
        raise _jinja_error(error) from None
    except AssemblerError as error:
        _report_and_exit(error)
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
