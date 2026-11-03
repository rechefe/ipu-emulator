"""The ``[tables]`` section of a ``kernel.toml``: each XMEM table the program
uses, as the Cache Unit's compiler needs it (for :mod:`ipu_harness.manifest`).
A multi-stage kernel has one sub-section per stage (``[tables.stage1]``).

A table is one region (an input, output or scratch tensor) reached through
its ``[abi]`` base CR. Its rows are the region's; the section adds what the
program does with them, as integers or shape-grammar expressions over
parameter and derived values (:mod:`ipu_harness.tables` evaluates them per
case and :mod:`ipu_harness.table_trace` checks them against the run).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ipu_harness._manifest_decls import TABLE_KEYS, ZERO_VIEW, TableDecl, TensorDecl
from ipu_harness._manifest_params import check_expression
from ipu_harness._toml_source import Reader, where

__all__ = ["STAGE_PREFIX", "read_tables"]

#: A stage's name is this and its number among the hook's stages, from 1.
STAGE_PREFIX = "stage"


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


def _entries(
    r: Reader,
    raw: object,
    tensors: Mapping[str, TensorDecl],
    known: Iterable[str],
    prefix: tuple[str, ...],
    *,
    also_via: bool,
) -> dict[str, TableDecl]:
    """The ``<tensor> = {...}`` entries of one section (without their ``also_via``)."""
    tables: dict[str, TableDecl] = {}
    for name, entry in r.table(raw, prefix).items():
        path = (*prefix, name)
        if name not in tensors:
            raise r.fail(path, f"{where(path)}: a table is a region; {name!r} is no tensor")
        table: Mapping[str, Any] = r.table(entry, path)
        r.known(table, TABLE_KEYS, path)
        if table.get("whole_run") and ({"repetitions", "jump_back"} & set(table)):
            raise r.fail(
                path, f"{where(path)}: a whole_run table takes no repetitions or jump_back"
            )
        if table.get("also_via") and not also_via:
            raise r.fail(path, f"{where(path)}: a stage's table takes no also_via")
        tables[name] = TableDecl(
            name=name,
            repetitions=_count(r, table.get("repetitions", 1), known, (*path, "repetitions")),
            jump_back=_count(r, table.get("jump_back", 0), known, (*path, "jump_back")),
            overhead_rows=_count(r, table.get("overhead_rows", 0), known, (*path, "overhead_rows")),
            whole_run=r.typed(table.get("whole_run", False), bool, (*path, "whole_run")),
            pending=r.typed(table.get("pending", ""), str, (*path, "pending")),
        )
    return tables


def _reaches(abi: Mapping[str, str], placed: Mapping[str, frozenset[str]], name: str) -> bool:
    """Whether a CR holds the tensor's base: its own binding, or a ``[derived]``
    value of its base alone (``x_band = base(x) + ...``)."""
    bound = abi.get(name, "").startswith("CR")
    return bound or any(abi.get(d, "").startswith("CR") and b == {name} for d, b in placed.items())


def _stage_name(r: Reader, stage: str) -> None:
    path = ("tables", stage)
    number = stage[len(STAGE_PREFIX) :]
    if not (stage.startswith(STAGE_PREFIX) and number.isdigit() and int(number) >= 1):
        raise r.fail(
            path,
            f"{where(path)}: a multi-stage kernel's tables are in [tables.stage<N>] "
            f"sections, not {stage!r}",
        )


def read_tables(
    r: Reader,
    raw: object,
    tensors: Mapping[str, TensorDecl],
    scalars: Iterable[str],
    abi: Mapping[str, str] | None,
    placed: Mapping[str, frozenset[str]],
) -> tuple[dict[str, TableDecl], dict[str, dict[str, TableDecl]]]:
    """The ``[tables]`` of a kernel: its tables by tensor name, or a
    multi-stage kernel's by stage name (``stage<N>``, its place among the
    stages the hook runs) and tensor name.

    A stage lists its own tables: a handoff buffer is in each stage it is
    reached in, with that stage's numbers. Which CR a stage reaches a tensor
    through is the stage's (the hook names them), so the run checks that.

    Args:
        r: The file's reader.
        raw: The ``[tables]`` section.
        tensors: The declared tensors.
        scalars: Parameter and derived names an expression may read (no
            placed value: a table's numbers do not depend on placement).
        abi: The ``[abi]`` bindings; ``None`` for a multi-stage kernel.
        placed: The placed derived values and the bases each reads.

    Returns:
        The tables of a one-program kernel, and those of a multi-stage
        kernel's stages; one of them is empty.
    """
    known = [s for s in scalars if s not in placed]
    if abi is None:
        stages = {}
        for stage, entries in r.table(raw, ("tables",)).items():
            _stage_name(r, stage)
            stages[stage] = _entries(r, entries, tensors, known, ("tables", stage), also_via=False)
        return {}, stages
    tables = _entries(r, raw, tensors, known, ("tables",), also_via=True)
    out = {}
    for name, decl in tables.items():
        path = ("tables", name)
        if not _reaches(abi, placed, name):
            raise r.fail(path, f"{where(path)}: {name} is reached through no CR in [abi]")
        via = _also_via(
            r,
            r.table(raw, ("tables",))[name].get("also_via", []),
            name,
            abi,
            placed,
            (*path, "also_via"),
        )
        out[name] = replace(decl, also_via=via)
    return out, {}
