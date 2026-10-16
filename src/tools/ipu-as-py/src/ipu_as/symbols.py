"""The label table of one assembly."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import lark

from ipu_as.errors import LabelError, SourceLocation

if TYPE_CHECKING:
    from ipu_as.ipu_token import AnnotatedToken


class SymbolTable:
    """Label name -> address of the instruction word it labels.

    Each assembly owns one, so two assemblies in one process never see each
    other's labels.
    """

    def __init__(self) -> None:
        self._labels: dict[str, AnnotatedToken] = {}

    def define(self, name: AnnotatedToken) -> None:
        """Define the label `name` at its token's instruction word; LabelError if defined twice."""
        if name.token.value in self._labels:
            existing = self._labels[name.token.value].token
            raise LabelError(
                f"Label '{name.token.value}' is defined for the second time at "
                f"Line {name.token.line}, Column {name.token.column}. "
                f"Previous definition at Line {existing.line}, Column {existing.column}.",
                location=SourceLocation.of(name.token),
                token=name.token.value,
            )
        self._labels[name.token.value] = name

    def address(self, ref: lark.Token) -> int:
        """The address of the label `ref` names; LabelError if it is not defined."""
        if ref.value not in self._labels:
            raise LabelError(
                f"Label '{ref.value}' referenced in Line {ref.line}, Column {ref.column} not defined.",
                location=SourceLocation.of(ref),
                token=ref.value,
            )
        return self._labels[ref.value].instr_id

    def __contains__(self, name: object) -> bool:
        return name in self._labels

    def items(self) -> Iterator[tuple[str, int]]:
        """(name, address) for every label, in definition order."""
        return ((name, token.instr_id) for name, token in self._labels.items())
