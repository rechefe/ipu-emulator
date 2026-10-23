"""What a kernel's generated docs page shows, as data (E06-21).

``//kernels:manifest`` adds :func:`page_data` to each kernel; E08 and E09
render it. Nothing here reads an ABI from a comment: the ABI table comes
from ``[abi]``, the design notes from the program's leading ``{#- ... -#}``
block (Q79), so neither can drift from what runs.
"""

from __future__ import annotations

import re
import textwrap
from typing import Any

from ipu_harness.manifest import KernelManifest
from ipu_harness.tolerance import DEFAULT_TIER

__all__ = ["abi_table", "design_notes", "page_data", "tolerances"]

#: The program's leading Jinja comment: its header (Q79).
_HEADER = re.compile(r"\A\s*\{#-?(?P<body>.*?)-?#\}", re.DOTALL)


def design_notes(program: str) -> str | None:
    """The leading ``{#- ... -#}`` block of a program, dedented.

    Args:
        program: The ``.asm`` text.

    Returns:
        The header's text, or None when the program does not start with one
        (or it is empty).
    """
    m = _HEADER.match(program)
    if m is None:
        return None
    first, _, rest = m.group("body").strip("\n").partition("\n")
    text = "\n".join([first.strip(), textwrap.dedent(rest)]).strip()
    return text or None


def _kind(manifest: KernelManifest, name: str) -> str:
    if name in manifest.tensors:
        return "region"
    if name in manifest.derived:
        return "derived"
    return "element scalar" if manifest.params[name].type == "float" else "int"


def abi_table(manifest: KernelManifest) -> list[dict[str, Any]]:
    """One row per ``[abi]`` binding, in its order.

    Args:
        manifest: The kernel.

    Returns:
        ``name``, ``register``, ``kind`` (``region``, ``int``, ``element
        scalar`` or ``derived``), ``offset`` (a region's, else 0) and ``when``
        (the condition it binds under, or None).
    """
    return [
        {
            "name": name,
            "register": register,
            "kind": _kind(manifest, name),
            "offset": manifest.abi_offsets.get(name, 0),
            "when": str(manifest.abi_when[name]) if name in manifest.abi_when else None,
        }
        for name, register in manifest.abi.items()
    ]


def _domain(manifest: KernelManifest) -> dict[str, dict[str, Any]]:
    """Each parameter's type and the bounds it declares (a fixed one has one choice)."""
    out = {}
    for name, p in manifest.params.items():
        fields = {"min": p.min, "max": p.max, "multiple_of": p.multiple_of, "default": p.default}
        domain: dict[str, Any] = {"type": p.type, "unit": p.unit}
        domain.update({k: v for k, v in fields.items() if v is not None})
        if p.choices is not None:
            domain["choices"] = list(p.choices)
            if p.default not in p.choices:  # a registry entry's: leaving it out is refused
                domain.pop("default", None)
        out[name] = domain
    return out


def tolerances(manifest: KernelManifest) -> dict[str, dict[str, str]]:
    """The tier each output is compared at, in each variant the kernel runs in
    (``compare.tier_for``).

    Args:
        manifest: The kernel.

    Returns:
        Variant name to output name to tier name; ``EXACT`` unless ``[golden]``
        names a tier.
    """
    tolerance = manifest.golden.tolerance
    outputs = [t for t in manifest.tensors.values() if t.role == "output"]
    return {
        fmt.value: {t.name: tolerance.get(fmt, DEFAULT_TIER).name for t in outputs}
        for fmt in manifest.formats
    }


def page_data(manifest: KernelManifest) -> dict[str, Any]:
    """What the kernel's page shows besides its cases and targets.

    Args:
        manifest: The kernel.

    Returns:
        ``design_notes`` (:func:`design_notes` of its program), ``abi``
        (:func:`abi_table`), ``params`` (each one's domain) and ``tolerance``
        (:func:`tolerances`).
    """
    return {
        "design_notes": design_notes(manifest.asm.read_text(encoding="utf-8")),
        "abi": abi_table(manifest),
        "params": _domain(manifest),
        "tolerance": tolerances(manifest),
    }
