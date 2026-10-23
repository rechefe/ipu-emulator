"""The ``[golden]`` table of a ``kernel.toml``: the reference and the tolerance
tier per variant, or per mode (``nx8``) for every variant of it without its
own (for :mod:`ipu_harness.manifest`)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ipu_harness._manifest_decls import GOLDEN_KEYS, TIER_KEYS, GoldenDecl, TensorDecl
from ipu_harness._toml_source import Reader, where
from ipu_harness.tolerance import TIERS
from ipu_harness.tolerance import Tier as ToleranceTier
from ipu_harness.variants import MODE_KEYS, Variant


def _named_tier(r: Reader, raw: object, path: tuple[str, ...]) -> ToleranceTier:
    name = r.text(raw, path)
    if name not in TIERS:
        raise r.fail(path, f"{where(path)} {name!r} is no tier; tiers: {', '.join(TIERS)}")
    return TIERS[name]


def _tier(r: Reader, raw: object, fmt: Variant, path: tuple[str, ...]) -> ToleranceTier:
    """A tier by name, or a table overriding one's numbers with a ``why``.

    An override may set only the numbers ``compare`` reads for ``fmt``:
    ``ulps`` (NX8 steps) only in NX8, and there not ``rtol`` or ``atol``
    when the tier counts steps.
    """
    if not isinstance(raw, dict):
        return _named_tier(r, raw, path)
    r.known(raw, TIER_KEYS, path)
    if "tier" not in raw or "why" not in raw:
        raise r.fail(path, f"{where(path)} overrides a tier: it needs 'tier' and a 'why'")
    base = _named_tier(r, raw["tier"], (*path, "tier"))
    unread = ["ulps"] if not fmt.nx8 else []
    if fmt.nx8 and (base.counts_steps or "ulps" in raw):
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


def _tolerances(
    r: Reader, tiers: Mapping[str, object], formats: tuple[Variant, ...]
) -> dict[Variant, ToleranceTier]:
    """Each variant's tier: its own key, else its mode's (``nx8``)."""
    tolerance: dict[Variant, ToleranceTier] = {}
    for key, tier in tiers.items():
        path = ("golden", "tolerance", key)
        if key in MODE_KEYS and key != "fp32":  # a whole mode; "fp32" is a variant too
            if not any(f.mode is MODE_KEYS[key] for f in formats):
                raise r.fail(path, f"tolerance for {key}, a mode the kernel does not run in")
            for fmt in formats:
                if fmt.mode is MODE_KEYS[key] and fmt.name not in tiers:
                    tolerance[fmt] = _tier(r, tier, fmt, path)
            continue
        fmt = r.format(key, path)
        if fmt not in formats:
            raise r.fail(path, f"tolerance for {key}, a variant the kernel does not run in")
        tolerance[fmt] = _tier(r, tier, fmt, path)
    return tolerance


def read_golden(
    r: Reader,
    directory: Path,
    raw: object,
    formats: tuple[Variant, ...],
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
    tolerance: dict[Variant, ToleranceTier] = {}
    if "tolerance" in table:
        if not any(t.role == "output" for t in tensors.values()):
            raise r.fail(("golden", "tolerance"), "a tolerance is never read: no output")
        tiers = r.table(table["tolerance"], ("golden", "tolerance"))
        tolerance = _tolerances(r, tiers, formats)
    return GoldenDecl(reference, op, MappingProxyType(tolerance))
