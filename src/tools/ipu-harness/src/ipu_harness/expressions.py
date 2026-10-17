"""The shape grammar: integer expressions over parameter and derived names (E06-03).

A shape dimension or a ``[derived]`` value is an int literal, a name, or an
expression built from int literals, names, ``+ - * / // %``, unary minus, and
the calls ``ceil(x)``, ``min(a, b, ...)`` and ``max(a, b, ...)``. Nothing else is
evaluated. A ``[derived]`` value may also read ``base(<tensor>)``, a region's
base once it is placed (:func:`base_name`, :func:`placed_reads`): such a value
is known only when the registers are written, so only ``[abi]`` reads it.
Arithmetic is exact (fractions): ``/`` must end up inside ``ceil`` or divide
evenly, and the result must be a whole number.

:func:`names_in` checks an expression against the grammar without values (the
loader, at load time); :func:`evaluate` computes it (``plan_case``).
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping
from fractions import Fraction

_BINARY: Mapping[type, Callable[[Fraction, Fraction], Fraction]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: Fraction(a // b),
    ast.Mod: lambda a, b: a % b,
}
#: Each call: (fewest arguments, most arguments or None, the function).
_CALLS: Mapping[str, tuple[int, int | None, Callable[..., Fraction]]] = {
    "ceil": (1, 1, lambda x: Fraction(math.ceil(x))),
    "min": (2, None, min),
    "max": (2, None, max),
}
#: The call that reads a placed region's base (``base(x)``).
BASE = "base"

FUNCTIONS = frozenset({*_CALLS, BASE})


def base_name(tensor: str) -> str:
    """The name ``base(<tensor>)`` is looked up by, in :func:`evaluate`'s names
    and in what :func:`names_in` returns."""
    return f"{BASE}({tensor})"


def base_tensor(read: str) -> str | None:
    """The tensor a :func:`names_in` read of ``base(x)`` names; ``None`` for
    any other read."""
    prefix = f"{BASE}("
    return read[len(prefix) : -1] if read.startswith(prefix) and read.endswith(")") else None


def placed_reads(derived: Mapping[str, str]) -> dict[str, frozenset[str]]:
    """Each ``[derived]`` value that reads a region's base, directly or through
    an earlier such value, with the tensors whose bases it reads.

    Raises:
        ExpressionError: An expression is outside the grammar.
    """
    reads: dict[str, frozenset[str]] = {}
    for name, expression in derived.items():
        tensors: set[str] = set()
        placed = False
        for read in names_in(expression):
            tensor = base_tensor(read)
            if tensor is not None or read in reads:
                placed = True
                tensors |= {tensor} if tensor is not None else reads[read]
        if placed:
            reads[name] = frozenset(tensors)
    return reads


class ExpressionError(ValueError):
    """An expression outside the grammar, or one that does not evaluate. Callers
    turn it into a located configuration error (``ManifestError``, ``PlanError``)."""


def _parse(text: str) -> ast.Expression:
    try:
        return ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"{text!r} does not parse: {error.msg}") from None
    except (RecursionError, MemoryError):
        raise ExpressionError(f"{text[:40]!r}... is too deeply nested") from None


def _base(node: ast.Call, names: Mapping[str, Fraction] | None) -> Fraction:
    if node.keywords or len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
        raise ExpressionError("base() takes one tensor name")
    if names is None:
        return Fraction(0)
    key = base_name(node.args[0].id)
    if key not in names:
        raise ExpressionError(f"{key} is no placed region")
    return names[key]


def _call(node: ast.Call, names: Mapping[str, Fraction] | None) -> Fraction:
    if isinstance(node.func, ast.Name) and node.func.id == BASE:
        return _base(node, names)
    if not isinstance(node.func, ast.Name) or node.func.id not in _CALLS:
        raise ExpressionError("only ceil(), min(), max() and base() may be called")
    least, most, function = _CALLS[node.func.id]
    count = len(node.args)
    if node.keywords or count < least or (most is not None and count > most):
        wanted = f"{least}" if least == most else f"at least {least}"
        raise ExpressionError(f"{node.func.id}() takes {wanted} positional arguments, not {count}")
    values = [_walk(a, names) for a in node.args]
    return function(*values) if names is not None else Fraction(0)


def _walk(node: ast.AST, names: Mapping[str, Fraction] | None) -> Fraction:
    """Evaluate ``node``; with ``names`` None, only check the grammar."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value)
    if isinstance(node, ast.Name):
        if names is None:
            return Fraction(0)
        if node.id not in names:
            raise ExpressionError(f"{node.id!r} is no integer parameter or derived value")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_walk(node.operand, names)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _walk(node.left, names), _walk(node.right, names)
        if names is None:
            return Fraction(0)
        if right == 0 and isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod):
            raise ExpressionError("division by zero")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.Call):
        return _call(node, names)
    raise ExpressionError(f"{ast.dump(node)[:60]} is outside the shape grammar")


def names_in(text: str) -> list[str]:
    """The names ``text`` reads, after checking it against the grammar; a
    ``base(x)`` read is ``base_name("x")``.

    Raises:
        ExpressionError: ``text`` is outside the grammar.
    """
    tree = _parse(text)
    try:
        _walk(tree.body, None)
    except RecursionError:
        raise ExpressionError(f"{text[:40]!r}... is too deeply nested") from None
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    called = {id(n.func) for n in calls}
    bases = {
        id(n.args[0]): n.args[0]
        for n in calls
        if isinstance(n.func, ast.Name) and n.func.id == BASE
    }
    return [
        base_name(n.id) if id(n) in bases else n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and id(n) not in called
    ]


def evaluate(text: str, names: Mapping[str, int]) -> int:
    """``text`` over integer ``names``, as an exact int.

    Args:
        text: An expression in the shape grammar.
        names: Integer parameter and derived values.

    Returns:
        Its value.

    Raises:
        ExpressionError: Outside the grammar, an unknown name, division by zero,
            or a result that is not a whole number.
    """
    tree = _parse(text)
    try:
        value = _walk(tree.body, {k: Fraction(v) for k, v in names.items()})
    except RecursionError:
        raise ExpressionError(f"{text[:40]!r}... is too deeply nested") from None
    if value.denominator != 1:
        raise ExpressionError(f"{text!r} is {value}, not a whole number")
    return int(value)
