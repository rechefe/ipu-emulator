"""Assemble a kernel's program in-process, through the assembler's session.

Kept apart from ``cases`` so a harness module can assemble its own program
without discovery importing the case machinery.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.abc import Traversable
from pathlib import Path
from tempfile import TemporaryDirectory

from ipu_as import session
from ipu_as.session import AssemblyResult


def assemble_program(
    source: Path | Traversable,
    *,
    defines: Mapping[str, object] | None = None,
    search_path: Sequence[Path] = (),
) -> AssemblyResult:
    """Assemble one ``.asm`` file through the assembler's session (and its cache).

    Args:
        source: The ``.asm`` file, a path or a package resource.
        defines: Values for the parameters its ``ipu:params`` header declares.
        search_path: Directories its ``{% include %}`` and ``{% import %}``
            resolve in.

    Returns:
        The assembled program; equal inputs share one cached result.

    Raises:
        AssemblerError: the source does not render or assemble.
    """
    return session.assemble(
        source.read_text(), defines=defines, search_path=search_path, name=source.name
    )


@contextmanager
def program_file(program: AssemblyResult) -> Iterator[Path]:
    """``program`` written as a ``.bin`` file, which exists until the block ends.

    Yields:
        The path of the ``.bin`` file, in a temporary directory.
    """
    with TemporaryDirectory(prefix="ipu_program_") as tmp:
        path = Path(tmp) / "program.bin"
        path.write_bytes(program.to_bin())
        yield path
