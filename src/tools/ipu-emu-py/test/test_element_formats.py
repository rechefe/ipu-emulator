"""The emulator's ``DType`` corresponds to ``ipu_isa.formats`` (E03-13).

Pins the mapping E04-17 uses to replace ``DType`` with ``ElementFormat``
mechanically: ``DType(x)`` is the 8-bit format with ``x`` exponent bits.
"""

from __future__ import annotations

from ipu_emu.ipu import NARROW_MAX_ROW, XMEM_ADDRESSABLE_ROWS
from ipu_emu.ipu_math import DType
from ipu_isa.formats import FORMATS, ElementFormat, addressable_rows
from ipu_isa.tags import Tag

DTYPE_TO_FORMAT = {
    DType.INT8: ElementFormat.INT8,
    DType.E1: ElementFormat.E1,
    DType.E2: ElementFormat.E2,
    DType.E3: ElementFormat.E3,
    DType.E4: ElementFormat.E4,
    DType.E5: ElementFormat.E5,
    DType.E6: ElementFormat.E6,
    DType.E7: ElementFormat.E7,
}


def test_dtype_correspondence_is_complete():
    assert set(DTYPE_TO_FORMAT) == set(DType)
    hardware = {f for f, info in FORMATS.items() if info.tag is Tag.HARDWARE}
    assert set(DTYPE_TO_FORMAT.values()) == hardware


def test_dtype_value_is_exponent_bits():
    for dtype, fmt in DTYPE_TO_FORMAT.items():
        info = FORMATS[fmt]
        assert info.exp_bits == int(dtype)
        assert info.element_bits == 8
        assert info.man_bits == (7 - int(dtype) if info.is_float else 0)
        assert info.is_float == (dtype is not DType.INT8)
        assert fmt.name == dtype.name


def test_wide_rows_match_today():
    # Wide-vector mode (FP32/INT32) already addresses XMEM_BYTES // 512 rows.
    # 8-bit formats keep the emulator's 16384-row limit until E04 adopts
    # ``addressable_rows`` (IMPLEMENTATION-DECISIONS: rows = bytes / row width).
    assert addressable_rows(ElementFormat.FP32) == XMEM_ADDRESSABLE_ROWS
    assert addressable_rows(ElementFormat.INT32) == XMEM_ADDRESSABLE_ROWS
    assert addressable_rows(ElementFormat.INT8) >= NARROW_MAX_ROW
