"""File entry points of the assembler, over `ipu_as.session`.

The pipeline lives in `ipu_as.session`, and the parser and AST builder in
`ipu_as.syntax` (re-exported here for existing importers). These functions
never print or exit: a bad source raises an `AssemblerError` subclass. Only
`ipu_as.cli` turns errors into text and exit codes.
"""

from ipu_as import encode, session
from ipu_as.session import instruction_aligned_bytes_len
from ipu_as.syntax import ASTBuilder, get_parser, parse_tree
from ipu_isa.codec.disasm import program_text
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
    """The parsed instruction words of `text`. Raises an AssemblerError subclass."""
    return session.AssemblySession(text).parse()


def assemble(text: str) -> list[int]:
    """The instruction words of `text`. Raises an AssemblerError subclass."""
    return list(session.assemble(text).words)


def assemble_to_mem_file(text: str, output_path: str) -> None:
    """Assemble `text` into a `.mem` file: one `0x`-prefixed hex word per line.

    Args:
        text: The assembly source.
        output_path: The file to write.

    Raises:
        AssemblerError: the source does not assemble.
    """
    result = session.assemble(text)
    with open(output_path, "w") as f:
        f.write(result.to_mem())


def assemble_to_bin_file(text: str, output_path: str) -> None:
    """Assemble `text` into a `.bin` file of little-endian emulator words.

    Args:
        text: The assembly source.
        output_path: The file to write.

    Raises:
        AssemblerError: the source does not assemble.
    """
    result = session.assemble(text)
    with open(output_path, "wb") as f:
        f.write(result.to_bin())


#: How ``disassemble`` writes a program: every field of every slot, by word
#: index (``fields``), or source the assembler reads back (``assembly``).
SYNTAXES = ("fields", "assembly")


def disassemble(program: list[int], target: str = "emulator", syntax: str = "fields") -> list[str]:
    """The program as text lines; `target` says which word they are.

    Args:
        program: The words.
        target: `"emulator"` or `"hardware"`.
        syntax: `"fields"`: one `<index>:\t\t<text>` line per word with every
            field of every slot; `"assembly"`: source that assembles back to
            `program` (``ipu_isa.codec.disasm.program_text``).
    """
    if syntax == "assembly":
        return program_text(program, target=target).split("\n")
    if syntax != "fields":
        raise ValueError(f"syntax must be one of {SYNTAXES}, not {syntax!r}")
    return [
        f"{i:04x}:\t\t{encode.disassemble_word(word, target)}" for i, word in enumerate(program)
    ]


def _write_lines(output_path: str, lines: list[str]) -> None:
    with open(output_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def disassemble_from_mem_file(
    input_path: str, output_path: str, target: str = "emulator", syntax: str = "fields"
) -> None:
    """Disassemble a `.mem` file into one text line per word (`disassemble`).

    Args:
        input_path: The `.mem` file: one hex word per line.
        output_path: The text file to write.
        target: Which word the file holds, `"emulator"` or `"hardware"`.
        syntax: How to write it (``SYNTAXES``).
    """
    with open(input_path) as f:
        program = [int(line.strip(), 0) for line in f.readlines()]
    _write_lines(output_path, disassemble(program, target, syntax))


def _bin_words(data: bytes, target: str) -> list[int]:
    """The words of a `.bin` file of `target` words; a trailing partial word is ignored."""
    if encode.target_tag(target) is Tag.EMULATOR:
        return unpack_words(data)
    width = word_bytes(Tag.HARDWARE)
    return [
        int.from_bytes(data[offset : offset + width], "little")
        for offset in range(0, len(data) - width + 1, width)
    ]


def disassemble_from_bin_file(
    input_path: str, output_path: str, target: str = "emulator", syntax: str = "fields"
) -> None:
    """Disassemble a `.bin` file into one text line per word (`disassemble`).

    Args:
        input_path: The `.bin` file of little-endian `target` words.
        output_path: The text file to write.
        target: Which word the file holds, `"emulator"` or `"hardware"`.
        syntax: How to write it (``SYNTAXES``).
    """
    with open(input_path, "rb") as f:
        program = _bin_words(f.read(), target)
    _write_lines(output_path, disassemble(program, target, syntax))
