"""Write a small dict as TOML text.

The standard library reads TOML (``tomllib``) but does not write it, and an
experiment has to be saved next to its results. This writer covers what an
experiment spec holds: text, numbers, true/false, lists of those, tables,
and lists of tables (``[[sweep]]``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

__all__ = ["to_toml"]


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, Mapping):  # an inline table: { a = 1, b = 2 }
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {type(value).__name__} as TOML")


def _key(key: str) -> str:
    plain = key.replace("_", "").replace("-", "")
    return key if plain.isalnum() and plain.isascii() else json.dumps(key)


def _is_table_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(v, Mapping) for v in value)
    )


def to_toml(data: Mapping[str, Any]) -> str:
    """``data`` as TOML: plain keys first, then each table, then each list of
    tables. Tables one level deep are written as ``[name]`` sections; anything
    deeper is written inline."""
    lines = [f"{_key(k)} = {_value(v)}" for k, v in data.items()
             if not isinstance(v, Mapping) and not _is_table_list(v)]  # fmt: skip
    for name, table in data.items():
        if isinstance(table, Mapping):
            lines += [
                "",
                f"[{_key(name)}]",
                *(f"{_key(k)} = {_value(v)}" for k, v in table.items()),
            ]
    for name, tables in data.items():
        if _is_table_list(tables):
            for table in tables:
                lines += [
                    "",
                    f"[[{_key(name)}]]",
                    *(f"{_key(k)} = {_value(v)}" for k, v in table.items()),
                ]
    return "\n".join(lines).lstrip("\n") + "\n"
