"""The lane count and R_CYCLIC's element count are ``ipu_isa``'s (E03-12):
no ``ipu_emu`` module writes them as a literal.

A literal 128 or 512 in ``ipu_emu`` is a copy of ``ipu_isa.geometry.LANES``
or ``R_CYCLIC_ELEMENTS`` unless it means something else: the INT8 minimum
(``-128``), a cache size, or an int8 sign test, which ``_NOT_A_LANE_COUNT``
lists by its line.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import ipu_emu

PACKAGE = Path(inspect.getsourcefile(ipu_emu)).parent
_COPIED = (128, 512)
#: Lines whose 128 is the int8 sign bit, not a lane count.
_NOT_A_LANE_COUNT = {
    ("geometry.py", "return b if b < 128 else b - 256"),
    ("numerics/codecs.py", "return byte - 0x100 if byte & 0x80 else byte"),
    ("slots/lr.py", "signed = byte_val - 256 if byte_val >= 128 else byte_val"),
}
MODULES = sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py"))


def _copies(module: str) -> list[str]:
    source = (PACKAGE / module).read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            allowed.add(id(node.operand))  # -128: the INT8 minimum
        elif isinstance(node, ast.keyword) and node.arg == "maxsize":
            allowed.add(id(node.value))  # an lru_cache size
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and node.value in _COPIED
            and id(node) not in allowed
        ):
            line = lines[node.lineno - 1].strip()
            if (module, line) not in _NOT_A_LANE_COUNT:
                found.append(f"{module}:{node.lineno}: {line}")
    return found


@pytest.mark.parametrize("module", MODULES)
def test_no_lane_count_literal(module):
    assert not _copies(module), "use ipu_isa.geometry.LANES / R_CYCLIC_ELEMENTS"
