"""The keys the instruction-spec dicts may hold, at every level (E03-03).

Instructions, operands, slot metadata and pseudo-instructions are authored as
nested dicts in ``instruction_spec.py``. Every key a dict there may carry is
listed here, required or optional, so a typo such as ``"raed": "cycle_start"`` is
an error at import instead of a silently dropped field. Adding a key to the
spec means adding it here first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ipu_isa.errors import IpuError


class SpecError(IpuError, ValueError):
    """The instruction spec breaks a schema rule.

    The message starts with the path of the offending entry, for example
    ``acc.ACC.RESHAPE.operands[source].read``.
    """


@dataclass(frozen=True)
class Keys:
    """The keys one level of the spec requires and may carry."""

    required: frozenset[str]
    optional: frozenset[str] = frozenset()


INSTRUCTION = Keys(
    frozenset({"operands", "doc", "execute_fn"}),
    frozenset({"tag", "implicit_reads", "implicit_writes"}),
)
OPERAND = Keys(frozenset({"name", "type"}), frozenset({"read", "write"}))
#: One entry of an instruction's ``implicit_reads`` (E03-14).
IMPLICIT_READ = Keys(frozenset({"register", "view"}), frozenset({"via_operand"}))
#: One entry of an instruction's ``implicit_writes`` (E03-14).
IMPLICIT_WRITE = Keys(frozenset({"register"}), frozenset({"via_operand"}))
SLOT_METADATA_ENTRY = Keys(frozenset({"description"}))
PSEUDO_INSTRUCTION = Keys(frozenset({"operands", "expands_to", "doc"}), frozenset({"tag"}))
EXPANSION = Keys(frozenset({"slot", "instruction", "args"}))

#: Values of an operand's ``read`` key and of an implicit read's ``view``: the
#: register's value at cycle start, or the value forwarded from an earlier
#: slot of the same word (GLOSSARY.md; ``ipu_isa.timing.View``).
READ_SOURCES = frozenset({"cycle_start", "forwarded"})


def check_keys(path: str, entry: object, keys: Keys) -> None:
    """Raise unless ``entry`` is a dict holding every required key and no
    key outside ``keys``."""
    if not isinstance(entry, Mapping):
        raise SpecError(f"{path}: must be a dict, got {type(entry).__name__}")
    missing = sorted(keys.required - entry.keys())
    if missing:
        raise SpecError(f"{path}: missing key(s) {', '.join(map(repr, missing))}")
    unknown = sorted(entry.keys() - keys.required - keys.optional)
    if unknown:
        allowed = sorted(keys.required | keys.optional)
        raise SpecError(
            f"{path}: unknown key(s) {', '.join(map(repr, unknown))}; allowed: {allowed}"
        )


def check_unique(path: str, what: str, names: Iterable[str]) -> None:
    """Raise if two of ``names`` are equal."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise SpecError(f"{path}: duplicate {what} {name!r}")
        seen.add(name)
