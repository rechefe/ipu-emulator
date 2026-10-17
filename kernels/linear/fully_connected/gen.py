"""fully_connected's legacy data (E07-21): the ``ipu_apps`` kernel's ``.bin`` files.

``data/<dir>/`` holds, per data format, the inputs ``x.bin`` (10 x 128
elements), the weights ``w.bin`` (64 x 128 elements, output-major, not
transposed) and the golden ``y.bin`` (10 x 64 little-endian 32-bit
accumulator values: INT32 for INT8 data, FP32 for E4 and E5). They are the old
``test_data_format/{int8,fp8_e4m3,fp8_e5m2}`` files, renamed (Q34); nothing in
the repo generates them (E07-22 replaces them with seeded data and goldens
from the reference). INT32 runs the INT8 data: the format is the old
``wide_mode``, the elements widened to 32 bits as the old harness did by hand.

``legacy_bin`` is the case generator (``make(plan, rng)``) and
``legacy_golden`` the case's ``expected`` for ``y`` (``fn(plan)``). A missing
file is an error of the case, never a skip.
"""

from pathlib import Path

import numpy as np

from ipu_isa.formats import ElementFormat, element_values

DATA = Path(__file__).with_name("data")

#: The data folder each format reads.
FOLDERS = {
    ElementFormat.INT8: "int8",
    ElementFormat.INT32: "int8",
    ElementFormat.E4: "e4",
    ElementFormat.E5: "e5",
}


def _folder(fmt: ElementFormat) -> Path:
    if fmt not in FOLDERS:
        raise ValueError(f"fully_connected has no legacy data for {fmt.value}")
    return DATA / FOLDERS[fmt]


def _elements(path: Path, fmt: ElementFormat, shape: tuple[int, ...]) -> np.ndarray:
    """``path``'s 8-bit elements as values: signed bytes, or the FP8 codes of ``fmt``."""
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{path} holds {raw.size} elements, not {int(np.prod(shape))}")
    if fmt in (ElementFormat.INT8, ElementFormat.INT32):
        return raw.view(np.int8).astype(np.float64).reshape(shape)
    table = np.array(element_values(fmt), dtype=np.float64)
    return table[raw].reshape(shape)


def legacy_bin(plan, rng):
    """``x`` and ``w`` from the format's ``.bin`` files."""
    folder = _folder(plan.format)
    return {
        name: _elements(folder / f"{name}.bin", plan.format, plan.shapes[name])
        for name in ("x", "w")
    }


def legacy_golden(plan):
    """``y``, the format's golden ``.bin`` decoded: 32-bit integers for INT8 data,
    FP32 for E4 and E5."""
    path = _folder(plan.format) / "y.bin"
    dtype = "<i4" if plan.format in (ElementFormat.INT8, ElementFormat.INT32) else "<f4"
    values = np.frombuffer(path.read_bytes(), dtype=dtype)
    shape = plan.shapes["y"]
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{path} holds {values.size} values, not {int(np.prod(shape))}")
    return values.astype(np.float64).reshape(shape)
