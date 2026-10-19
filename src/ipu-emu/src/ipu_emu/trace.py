"""Traces: a header line, then one commit record per line (JSON Lines, E04-29).

The header says what produced the records: the commit record schema's
version, the ISA (``ipu_isa.trace_schema.isa_digest``), the element format,
the program and the emulator.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

from ipu_emu.commit import CommitRecord
from ipu_emu.ipu_state import IpuState
from ipu_isa.codec.word import DecodedWord
from ipu_isa.trace_schema import SCHEMA_VERSION, isa_digest, program_digest

__all__ = ["EMULATOR_VERSION", "TraceWriter", "read_trace", "trace_header"]

#: ``ipu-emu``'s ``[project] version``.
EMULATOR_VERSION = "0.1.0"


def _program_digest(state: IpuState) -> str:
    """``program_digest`` of the loaded words, encoded (an empty entry is 0)."""
    return program_digest(
        0 if word is None else DecodedWord(word).to_word()
        for word in state.inst_mem[: state.program_length]
    )


def trace_header(state: IpuState) -> dict[str, Any]:
    """The header of a trace of *state*'s loaded program."""
    return {
        "schema_version": SCHEMA_VERSION,
        "isa_digest": isa_digest(),
        "format": state.fmt.value,
        "program_digest": _program_digest(state),
        "emulator_version": EMULATOR_VERSION,
    }


class TraceWriter:
    """Writes a trace to *path*: *header* first, then each record written.

    A context manager: leaving the ``with`` block closes the file.

    Args:
        path: The file to write (replaced).
        header: The header line's fields (``trace.v1.json``).
    """

    def __init__(self, path: str | Path, header: dict[str, Any]) -> None:
        self._file: IO[str] = Path(path).open("w", encoding="utf-8")
        self._line(header)

    def _line(self, doc: dict[str, Any]) -> None:
        self._file.write(json.dumps(doc, separators=(",", ":")) + "\n")

    def write(self, record: CommitRecord) -> None:
        """Write *record* as the next line."""
        self._line(record.to_json())

    def write_all(self, records: Iterable[CommitRecord]) -> None:
        """Write each of *records*, in order."""
        for record in records:
            self.write(record)

    def close(self) -> None:
        """Close the file."""
        self._file.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_trace(path: str | Path) -> tuple[dict[str, Any], list[CommitRecord]]:
    """A trace's header and its records."""
    with Path(path).open(encoding="utf-8") as lines:
        header = json.loads(next(lines))
        return header, [CommitRecord.from_json(json.loads(line)) for line in lines if line.strip()]
