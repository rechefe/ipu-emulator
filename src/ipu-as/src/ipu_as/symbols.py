"""The label table of one assembly."""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

import lark

from ipu_as.errors import LabelError, SourceLocation

if TYPE_CHECKING:
    from ipu_as.ipu_token import AnnotatedToken


#: `N:` with decimal digits only defines a local label (GNU as numeric labels).
_LOCAL_NAME = re.compile(r"[0-9]+")
#: `Nf` / `Nb` refer to the next / the nearest previous `N:`; a bare `N` is an error.
_LOCAL_REFERENCE = re.compile(r"(?P<number>[0-9]+)(?P<direction>[fb]?)")


def is_local_reference(name: str) -> bool:
    """True for `Nf`, `Nb` and a bare `N`: a reference that names a local label."""
    return _LOCAL_REFERENCE.fullmatch(name) is not None


class SymbolTable:
    """Label name -> address of the instruction word it labels.

    Each assembly owns one, so two assemblies in one process never see each
    other's labels. A named label is defined once. A local label (`1:`) may be
    defined many times; `1f` refers to the next definition after the referring
    word and `1b` to the nearest one at or before it.
    """

    def __init__(self) -> None:
        self._labels: dict[str, AnnotatedToken] = {}
        #: Local label number -> addresses of its definitions, ascending.
        self._locals: dict[int, list[int]] = {}

    def define(self, name: AnnotatedToken) -> None:
        """Define the label `name` at its token's instruction word.

        Raises:
            LabelError: a named (not local) label is defined a second time.
        """
        if _LOCAL_NAME.fullmatch(name.token.value):
            bisect.insort(self._locals.setdefault(int(name.token.value), []), name.instr_id)
            return
        if is_local_reference(name.token.value):
            raise LabelError(
                f"Label '{name.token.value}' at {SourceLocation.of(name.token)} looks like a "
                f"local label reference; a named label cannot be digits followed by f or b",
                location=SourceLocation.of(name.token),
                token=name.token.value,
            )
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

    def address(self, ref: lark.Token, *, at: int) -> int:
        """The address of the label `ref` names, referenced from instruction word `at`.

        Raises:
            LabelError: the label is not defined; for a local reference, no
                definition lies in its direction, or the reference has no `f`/`b`.
        """
        local = _LOCAL_REFERENCE.fullmatch(ref.value)
        if local:
            return self._local_address(ref, local, at)
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

    def _local_address(self, ref: lark.Token, local: re.Match, at: int) -> int:
        number, direction = int(local["number"]), local["direction"]
        location = SourceLocation.of(ref)
        if not direction:
            raise LabelError(
                f"Label {ref.value} is a local label reference without a direction; "
                f"use {number}f or {number}b",
                location=location,
                token=ref.value,
            )
        addresses = self._locals.get(number, [])
        # addresses[:index] are at or before `at`; addresses[index:] are after it.
        index = bisect.bisect_right(addresses, at)
        if direction == "f":
            if index < len(addresses):
                return addresses[index]
            where = "after"
        else:
            if index > 0:
                return addresses[index - 1]
            where = "at or before"
        raise LabelError(
            f"local label {ref.value} has no definition {where} line {ref.line}",
            location=location,
            token=ref.value,
        )
