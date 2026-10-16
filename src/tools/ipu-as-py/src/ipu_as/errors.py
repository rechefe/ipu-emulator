"""Errors the assembler raises for bad input, each with its source location.

Every assembler failure a user can cause is an ``AssemblerError``. It is a
setup-time error in the shared ``IpuError`` hierarchy, never a hardware fault.
It also subclasses ``ValueError`` so existing ``except ValueError`` callers keep
working. The message text is the one the assembler always printed; the
location and the offending token are also available as attributes, so callers
need not parse the message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import lark
from ipu_isa.errors import IpuError


@dataclass(frozen=True)
class SourceLocation:
    """Where in the rendered source an error is."""

    line: int  #: 1-based, in the rendered text
    column: int  #: 1-based
    end_column: int | None = None  #: 1-based, exclusive
    source: str = "<input>"

    @classmethod
    def of(cls, token: lark.Token) -> "SourceLocation":
        """The location of a lark token from the parse tree."""
        return cls(token.line, token.column, getattr(token, "end_column", None))

    def __str__(self) -> str:
        return f"Line {self.line}, Column {self.column}"


class AssemblerError(IpuError, ValueError):
    """A problem with the assembly source. ``stage`` is the ``check --json`` stage."""

    stage: ClassVar[str] = "assemble"

    def __init__(
        self,
        message: str,
        *,
        location: SourceLocation | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(message)
        self.location = location
        self.token = token


class TemplateError(AssemblerError):
    """The Jinja template failed to render."""

    stage = "template"


class ParseError(AssemblerError):
    """The rendered source does not match the assembly grammar."""

    stage = "parse"


class EncodeError(AssemblerError):
    """The source parses but cannot be encoded."""

    stage = "assemble"


class OperandError(EncodeError):
    """An operand has a bad value or is out of range, or the operand count is wrong."""


class SlotError(EncodeError):
    """An unknown mnemonic, or too many instructions for one slot kind."""


class LabelError(EncodeError):
    """A label is defined twice or used but never defined."""
