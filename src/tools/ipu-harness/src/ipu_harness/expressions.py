"""The shape grammar: integer expressions over parameter and derived names (E06-03).

A shape dimension or a ``[derived]`` value is an int literal, a name, or an
expression built from int literals, names, ``+ - * / // %``, unary minus, and
the calls ``ceil(x)``, ``min(a, b, ...)``, ``max(a, b, ...)`` and
``dstructure(valid, partition=0, pad_mode=0)`` (the last two positional or by
name; valid 0..128; a dstructure CR word, as
``ipu_isa.constants.encode_dstructure`` packs it, so a kernel binds one to any
CR without the bit layout in its file). Nothing else is evaluated. A
``[derived]`` value may also read ``base(<tensor>)``, a region's base once it
is placed (:func:`base_name`, :func:`placed_reads`): such a value is known
only when the registers are written, so only ``[abi]`` reads it. So is a
``[derived]`` value that reads :data:`ROW_BYTES` (``row_bytes``, the bytes of
one XMEM row in the case's format), which a ``row`` pin may also read: a
legacy layout at fixed byte addresses in every format (``fully_connected``).
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

from ipu_isa.constants import encode_dstructure
from ipu_isa.geometry import LANES

_BINARY: Mapping[type, Callable[[Fraction, Fraction], Fraction]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: Fraction(a // b),
    ast.Mod: lambda a, b: a % b,
}


#: ``dstructure()``'s arguments, in order; the last two may be given by name.
_DSTRUCTURE_ARGS = ("valid", "partition", "pad_mode")


def _dstructure(*args: Fraction) -> Fraction:
    """A dstructure CR word from (valid elements, partition, pad mode)."""
    if any(a.denominator != 1 for a in args):
        raise ExpressionError(f"dstructure() takes whole numbers, not {[str(a) for a in args]}")
    valid, partition, pad = (int(a) for a in (*args, Fraction(0), Fraction(0))[:3])
    if not 0 <= valid <= LANES:
        raise ExpressionError(f"dstructure({valid}, ...): valid elements must be 0..{LANES}")
    try:
        return Fraction(encode_dstructure(valid_elements=valid, partition=partition, pad_mode=pad))
    except ValueError as error:
        raise ExpressionError(f"dstructure({valid}, {partition}, {pad}): {error}") from None


#: Each call: (fewest arguments, most arguments or None, the function).
_CALLS: Mapping[str, tuple[int, int | None, Callable[..., Fraction]]] = {
    "ceil": (1, 1, lambda x: Fraction(math.ceil(x))),
    "min": (2, None, min),
    "max": (2, None, max),
    "dstructure": (1, 3, _dstructure),  # its arguments are checked by _dstructure_call
}
#: The call that reads a placed region's base (``base(x)``).
BASE = "base"

FUNCTIONS = frozenset({*_CALLS, BASE})

#: The name of the bytes of one XMEM row in the case's format (128 x the
#: element's bytes: 128 in an 8-bit format, 512 in FP32 and INT32). Known only
#: with the format, so a ``row`` pin and a placed ``[derived]`` value may read
#: it, and nothing that routing evaluates without a format.
ROW_BYTES = "row_bytes"


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
    """Each ``[derived]`` value that reads a region's base or :data:`ROW_BYTES`,
    directly or through an earlier such value, with the tensors whose bases it
    reads (none for one that reads only ``row_bytes``).

    Raises:
        ExpressionError: An expression is outside the grammar.
    """
    reads: dict[str, frozenset[str]] = {}
    for name, expression in derived.items():
        tensors: set[str] = set()
        placed = False
        for read in names_in(expression):
            tensor = base_tensor(read)
            if tensor is not None or read in reads or read == ROW_BYTES:
                placed = True
                if tensor is not None:
                    tensors.add(tensor)
                elif read in reads:
                    tensors |= reads[read]
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


def _dstructure_call(node: ast.Call, names: Mapping[str, Fraction] | None) -> Fraction:
    """``dstructure(valid, partition=0, pad_mode=0)``: positional, or the last
    two by name."""
    given = list(node.args)
    named = {k.arg: k.value for k in node.keywords}
    unknown = [k for k in named if k not in _DSTRUCTURE_ARGS[1:]]
    if unknown or None in named or not 1 <= len(given) <= 3:
        raise ExpressionError(
            "dstructure() takes valid, then partition and pad_mode (positional or by name)"
        )
    slots: list[ast.expr | None] = [*given, *[None] * (3 - len(given))]
    for i, key in enumerate(_DSTRUCTURE_ARGS):
        if key in named:
            if slots[i] is not None:
                raise ExpressionError(f"dstructure() got {key} twice")
            slots[i] = named[key]
    values = [Fraction(0) if a is None else _walk(a, names) for a in slots]
    return _dstructure(*values) if names is not None else Fraction(0)


def _call(node: ast.Call, names: Mapping[str, Fraction] | None) -> Fraction:
    if isinstance(node.func, ast.Name) and node.func.id == BASE:
        return _base(node, names)
    if isinstance(node.func, ast.Name) and node.func.id == "dstructure":
        return _dstructure_call(node, names)
    if not isinstance(node.func, ast.Name) or node.func.id not in _CALLS:
        raise ExpressionError("only ceil(), min(), max(), dstructure() and base() may be called")
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
            raise ExpressionError(f"{node.id!r} is no parameter or derived value this reads")
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


def evaluate_exact(text: str, names: Mapping[str, int | float]) -> Fraction:
    """``text`` over ``names``, exactly, whole or not (a scratch region's
    ``fill``: ``"1 / channels"``; a float name is its exact binary value).

    Raises:
        ExpressionError: Outside the grammar, an unknown name or division by
            zero.
    """
    tree = _parse(text)
    try:
        return _walk(tree.body, {k: Fraction(v) for k, v in names.items()})
    except RecursionError:
        raise ExpressionError(f"{text[:40]!r}... is too deeply nested") from None


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
