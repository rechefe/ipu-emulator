"""The ``[tables]`` section of a ``kernel.toml``: each XMEM table the program
uses, as the Cache Unit's compiler needs it (for :mod:`ipu_harness.manifest`).

A table is one region (an input, output or scratch tensor) reached through
its ``[abi]`` base CR. Its rows are the region's; the section adds what the
program does with them, as integers or shape-grammar expressions over
parameter and derived values (:mod:`ipu_harness.tables` evaluates them per
case and :mod:`ipu_harness.table_trace` checks them against the run).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ipu_harness._manifest_decls import TABLE_KEYS, ZERO_VIEW, TableDecl, TensorDecl
from ipu_harness._manifest_params import check_expression
from ipu_harness._toml_source import Reader, where

__all__ = ["read_tables"]


def _count(r: Reader, raw: object, names: Iterable[str], path: tuple[str, ...]) -> int | str:
    value = r.typed(raw, (int, str), path)
    if isinstance(value, str):
        check_expression(r, value, names, path)
    elif value < 0:
        raise r.fail(path, f"{where(path)} {value} must not be negative")
    return value


def _also_via(
    r: Reader,
    raw: object,
    name: str,
    abi: Mapping[str, str],
    placed: Mapping[str, frozenset[str]],
    path: tuple[str, ...],
) -> tuple[str, ...]:
    """Each entry is ``CR0`` or an ``[abi]`` CR binding of a value that reads ``base(name)``."""
    entries = r.typed(raw, list, path)
    for entry in entries:
        entry = r.text(entry, path)
        if entry == ZERO_VIEW:
            continue
        if not abi.get(entry, "").startswith("CR"):
            raise r.fail(path, f"{where(path)} {entry!r} is bound to no CR in [abi]")
        if name not in placed.get(entry, frozenset()):
            raise r.fail(path, f"{where(path)} {entry!r} does not read base({name})")
    if len(set(entries)) != len(entries):
        raise r.fail(path, f"{where(path)} lists an entry twice")
    return tuple(entries)


def read_tables(
    r: Reader,
    raw: object,
    tensors: Mapping[str, TensorDecl],
    scalars: Iterable[str],
    abi: Mapping[str, str],
    placed: Mapping[str, frozenset[str]],
) -> dict[str, TableDecl]:
    """The ``[tables]`` entries, keyed by tensor name, checked against the file.

    Args:
        r: The file's reader.
        raw: The ``[tables]`` section.
        tensors: The declared tensors.
        scalars: Parameter and derived names an expression may read (no
            placed value: a table's numbers do not depend on placement).
        abi: The ``[abi]`` bindings.
        placed: The placed derived values and the bases each reads.

    Returns:
        Each table's declaration.
    """
    known = [s for s in scalars if s not in placed]
    tables: dict[str, TableDecl] = {}
    for name, entry in r.table(raw, ("tables",)).items():
        path = ("tables", name)
        if name not in tensors:
            raise r.fail(path, f"{where(path)}: a table is a region; {name!r} is no tensor")
        if not abi.get(name, "").startswith("CR"):
            raise r.fail(path, f"{where(path)}: {name} is reached through no CR in [abi]")
        table: Mapping[str, Any] = r.table(entry, path)
        r.known(table, TABLE_KEYS, path)
        if table.get("whole_run") and ({"repetitions", "jump_back"} & set(table)):
            raise r.fail(
                path, f"{where(path)}: a whole_run table takes no repetitions or jump_back"
            )
        tables[name] = TableDecl(
            name=name,
            repetitions=_count(r, table.get("repetitions", 1), known, (*path, "repetitions")),
            jump_back=_count(r, table.get("jump_back", 0), known, (*path, "jump_back")),
            overhead_rows=_count(r, table.get("overhead_rows", 0), known, (*path, "overhead_rows")),
            whole_run=r.typed(table.get("whole_run", False), bool, (*path, "whole_run")),
            also_via=_also_via(
                r, table.get("also_via", []), name, abi, placed, (*path, "also_via")
            ),
        )
    return tables
