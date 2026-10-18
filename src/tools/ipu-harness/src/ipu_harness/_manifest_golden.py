"""The ``[golden]`` table of a ``kernel.toml``: the reference and the tolerance
tier per format (for :mod:`ipu_harness.manifest`)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ipu_harness._manifest_decls import GOLDEN_KEYS, TIER_KEYS, GoldenDecl, TensorDecl
from ipu_harness._toml_source import Reader, where
from ipu_harness.tolerance import TIERS
from ipu_harness.tolerance import Tier as ToleranceTier
from ipu_isa.formats import FORMATS, ElementFormat


def _named_tier(r: Reader, raw: object, path: tuple[str, ...]) -> ToleranceTier:
    name = r.text(raw, path)
    if name not in TIERS:
        raise r.fail(path, f"{where(path)} {name!r} is no tier; tiers: {', '.join(TIERS)}")
    return TIERS[name]


def _tier(r: Reader, raw: object, fmt: ElementFormat, path: tuple[str, ...]) -> ToleranceTier:
    """A tier by name, or a table overriding one's numbers with a ``why``.

    An override may set only the numbers ``compare`` reads for ``fmt``:
    ``ulps`` only for an 8-bit float format, and there not ``rtol`` or ``atol``
    when the tier counts codes.
    """
    if not isinstance(raw, dict):
        return _named_tier(r, raw, path)
    r.known(raw, TIER_KEYS, path)
    if "tier" not in raw or "why" not in raw:
        raise r.fail(path, f"{where(path)} overrides a tier: it needs 'tier' and a 'why'")
    base = _named_tier(r, raw["tier"], (*path, "tier"))
    fp8 = FORMATS[fmt].is_float and FORMATS[fmt].element_bits == 8
    unread = ["ulps"] if not fp8 else []
    if fp8 and (base.ulps is not None or "ulps" in raw):
        unread = ["rtol", "atol"]
    for key in unread:
        if key in raw:
            raise r.fail(
                (*path, key), f"{where((*path, key))} is never read for {fmt.value} at this tier"
            )
    rtol, atol = (_non_negative(r, raw, key, (int, float), path) for key in ("rtol", "atol"))
    ulps = _non_negative(r, raw, "ulps", int, path)
    return ToleranceTier(
        base.name,
        base.rtol if rtol is None else float(rtol),
        base.atol if atol is None else float(atol),
        base.ulps if ulps is None else int(ulps),
        r.text(raw["why"], (*path, "why")),
    )


def _non_negative(
    r: Reader,
    table: Mapping[str, Any],
    key: str,
    kind: type | tuple[type, ...],
    path: tuple[str, ...],
) -> float | None:
    """``table[key]`` when given: a number of ``kind`` that is not negative."""
    if key not in table:
        return None
    value = r.typed(table[key], kind, (*path, key))
    if value < 0:
        raise r.fail((*path, key), f"{where((*path, key))} {value} must not be negative")
    return float(value)


def _read_formats(
    formats: tuple[ElementFormat, ...], tensors: Mapping[str, TensorDecl]
) -> set[ElementFormat]:
    """The formats some output is compared in: a run's format for an element
    output, its accumulator format (FP32, INT32) for an accumulator output
    (as ``packing.stored_format``, which this module cannot import)."""
    read = set()
    for tensor in tensors.values():
        if tensor.role != "output":
            continue
        for fmt in formats:
            accumulated = ElementFormat(FORMATS[fmt].accumulator.value)
            read.add(accumulated if tensor.store == "accumulator" else fmt)
    return read


def read_golden(
    r: Reader,
    directory: Path,
    raw: object,
    formats: tuple[ElementFormat, ...],
    tensors: Mapping[str, TensorDecl],
) -> GoldenDecl:
    table = r.table(raw, ("golden",))
    r.known(table, GOLDEN_KEYS, ("golden",))
    if "reference" in table and "op" in table:
        raise r.fail(("golden", "op"), "[golden] names both a reference and an op; keep one")
    reference = None
    if "reference" in table:
        reference = r.function(directory, table["reference"], ("golden", "reference"))
    elif "op" not in table and (directory / "reference.py").is_file():
        reference = "reference.py:reference"
    op = r.text(table["op"], ("golden", "op")) if "op" in table else None
    tolerance: dict[ElementFormat, ToleranceTier] = {}
    if "tolerance" in table:
        tiers = r.table(table["tolerance"], ("golden", "tolerance"))
        # An accumulator output is compared in its accumulator format (FP32 or INT32).
        accumulated = {ElementFormat(FORMATS[f].accumulator.value) for f in formats}
        read = _read_formats(formats, tensors)
        for fmt, tier in tiers.items():
            path = ("golden", "tolerance", fmt)
            key = r.format(fmt, path)
            if key not in formats and key not in accumulated:
                raise r.fail(
                    path,
                    f"tolerance for {fmt}, a format the kernel does not run in or accumulate in",
                )
            if key not in read:
                compared = ", ".join(sorted(f.value for f in read)) or "none"
                raise r.fail(
                    path,
                    f"tolerance for {fmt} is never read: no output is compared in {fmt} "
                    f"(an accumulator output is compared in its accumulator format; "
                    f"outputs are compared in: {compared})",
                )
            tolerance[key] = _tier(r, tier, key, path)
    return GoldenDecl(reference, op, MappingProxyType(tolerance))
