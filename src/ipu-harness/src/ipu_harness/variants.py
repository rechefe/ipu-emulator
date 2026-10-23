"""A case's variant: the run mode and, in NX8, the element type (ADR 0007).

A kernel lists the variants it runs in (``[kernel] formats``), by name:

* ``fp32``: FP32 mode (emulator only): 512-byte rows of IEEE binary32, no
  metadata, no quantization.
* ``e1`` .. ``e7``: NX8 mode with a signed, 8-bit element of that many
  exponent bits (``fe``); ``dynamic``: NX8 with the signed 8-bit dynamic type
  (``fe`` 1 or 2 per row, ``ARCH-stage-aaq-str.md`` §4.3.2.3).
* ``s<sign>e<fe>w<bits>``: any other static NX8 type, ``s0e3w6`` for an
  unsigned 6-bit element with 3 exponent bits.

A variant replaces the old per-format :class:`~ipu_isa.formats.ElementFormat`
key of every harness table. A ``range`` or ``tolerance`` table may also key
the whole mode (``nx8``, ``fp32``): a variant's own entry wins.
"""

from __future__ import annotations

import re
from functools import cache

from ipu_isa.cr_halves import qf
from ipu_isa.errors import IsaValueError
from ipu_isa.formats import FORMATS, ElementFormat, FormatInfo
from ipu_isa.nx8 import Nx8Format

__all__ = ["MODE_KEYS", "VARIANT_NAMES", "Variant", "as_variant", "variant"]

#: The names a ``range`` or ``tolerance`` table may key a whole mode by.
MODE_KEYS = {fmt.value: fmt for fmt in ElementFormat}

#: The named NX8 variants, besides the general ``s<sign>e<fe>w<bits>`` form.
_NAMED: dict[str, Nx8Format] = {
    **{f"e{fe}": Nx8Format(sign=1, exp_format=fe, num_bits=8) for fe in range(1, 8)},
    "dynamic": Nx8Format(sign=1, exp_format=0, num_bits=8, dynamic=True),
}
_GENERAL = re.compile(r"s([01])e([1-7])w([1-8])")

#: Every variant name with a fixed spelling (the general form is not listed).
VARIANT_NAMES = ("fp32", *_NAMED)


class Variant(str):
    """One variant: the run mode and, in NX8, the element type.

    A ``str`` (its name, as an ``ElementFormat`` member is its value), so a
    variant compares equal to its name and keys a table by it.

    Attributes:
        name: Its name in ``kernel.toml`` (``"e4"``, ``"fp32"``).
        mode: The run mode (``ElementFormat.NX8`` or ``ElementFormat.FP32``).
        qf: The NX8 element type (a QF half's fields), or ``None`` in FP32.
    """

    mode: ElementFormat
    qf: Nx8Format | None

    def __new__(cls, name: str, mode: ElementFormat, qf: Nx8Format | None) -> Variant:
        obj = super().__new__(cls, name)
        obj.mode = mode
        obj.qf = qf
        return obj

    def __repr__(self) -> str:
        return f"Variant({str(self)!r})"

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return variant, (str(self),)

    @property
    def name(self) -> str:
        """Its name."""
        return str(self)

    @property
    def value(self) -> str:
        """The name (as an ``ElementFormat``'s ``value`` is)."""
        return str(self)

    @property
    def info(self) -> FormatInfo:
        """The run mode's sizes (``ipu_isa.formats.FORMATS``)."""
        return FORMATS[self.mode]

    @property
    def row_bytes(self) -> int:
        """Bytes of one XMEM row (metadata not included)."""
        return self.info.row_bytes

    @property
    def nx8(self) -> bool:
        """True in NX8 mode."""
        return self.mode is ElementFormat.NX8

    @property
    def qf_half(self) -> int:
        """The variant's QF half-CR value; 0 in FP32 (FP32 ignores the QF)."""
        return 0 if self.qf is None else qf(self.qf)


@cache
def variant(name: str) -> Variant:
    """The variant ``name`` names.

    Raises:
        ValueError: No variant has that name.
    """
    if name == "fp32":
        return Variant(name, ElementFormat.FP32, None)
    if name in _NAMED:
        return Variant(name, ElementFormat.NX8, _NAMED[name])
    match = _GENERAL.fullmatch(name)
    if match:
        sign, fe, bits = (int(g) for g in match.groups())
        try:
            fmt = Nx8Format(sign=sign, exp_format=fe, num_bits=bits)
        except IsaValueError as error:
            raise ValueError(f"variant {name!r}: {error}") from None
        return Variant(name, ElementFormat.NX8, fmt)
    raise ValueError(
        f"unknown variant {name!r}; valid: {', '.join(VARIANT_NAMES)}, or s<sign>e<fe>w<bits>"
    )


def as_variant(value: Variant | ElementFormat | str) -> Variant:
    """``value`` as a :class:`Variant`: a variant, its name, or
    ``ElementFormat.FP32`` (FP32 mode has one variant).

    Raises:
        ValueError: An unknown name, or ``ElementFormat.NX8``, which names no
            element type.
    """
    if isinstance(value, Variant):
        return value
    if isinstance(value, ElementFormat):
        if value is ElementFormat.FP32:
            return variant("fp32")
        raise ValueError("NX8 mode has several variants: name one (e4, e5, dynamic, ...)")
    return variant(str(value))
