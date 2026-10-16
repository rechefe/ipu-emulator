"""File entry points of the assembler, over `ipu_as.session`.

The pipeline lives in `ipu_as.session`, and the parser and AST builder in
`ipu_as.syntax` (re-exported here for existing importers). These functions
never print or exit: a bad source raises an `AssemblerError` subclass. Only
`ipu_as.cli` turns errors into text and exit codes.
"""

from ipu_as import encode, session
from ipu_as.session import instruction_aligned_bytes_len
from ipu_as.syntax import ASTBuilder, get_parser, parse_tree
from ipu_isa.codec.word import unpack_words, word_bytes
from ipu_isa.tags import Tag

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


def disassemble(program: list[int], target: str = "emulator") -> list[str]:
    """One `<index>:\t\t<text>` line per word; `target` says which word they are."""
    return [
        f"{i:04x}:\t\t{encode.disassemble_word(word, target)}" for i, word in enumerate(program)
    ]


def _write_lines(output_path: str, lines: list[str]) -> None:
    with open(output_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def disassemble_from_mem_file(input_path: str, output_path: str, target: str = "emulator"):
    with open(input_path) as f:
        program = [int(line.strip(), 0) for line in f.readlines()]
    _write_lines(output_path, disassemble(program, target))


def _bin_words(data: bytes, target: str) -> list[int]:
    """The words of a `.bin` file of `target` words; a trailing partial word is ignored."""
    if encode.target_tag(target) is Tag.EMULATOR:
        return unpack_words(data)
    width = word_bytes(Tag.HARDWARE)
    return [
        int.from_bytes(data[offset : offset + width], "little")
        for offset in range(0, len(data) - width + 1, width)
    ]


def disassemble_from_bin_file(input_path: str, output_path: str, target: str = "emulator"):
    with open(input_path, "rb") as f:
        program = _bin_words(f.read(), target)
    _write_lines(output_path, disassemble(program, target))
