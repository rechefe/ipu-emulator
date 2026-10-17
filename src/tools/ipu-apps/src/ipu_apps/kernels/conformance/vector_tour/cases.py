"""Vector tour inputs and an independent NumPy reference for every store.

The data is seeded and small: INT8 elements are integers in [-8, 8], float
elements multiples of 0.25 in [-2, 2], which E3, E4 and E5 all hold exactly.
Every product and sum is then exact in the int32 or float32 accumulator, so the
reference computes in int64 or float64 and casts once.
"""

import numpy as np

from ipu_apps.kernel_registry.case_support import check_close
from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_isa.formats import ElementFormat, decode_element, encode_element

from .app import A_ROWS, ACC_STORE_ROWS, ACC_STORES, ROW_BYTES, output_rows, parse_dtype

LANES = 128
#: R_MASK slots, 16 bytes each, bit i = lane i: all lanes, even lanes, lower half.
MASK_SLOTS = (b"\xff" * 16, b"\x55" * 16, b"\xff" * 8 + b"\x00" * 8)
#: The R0 element the program reads as a scalar (LR4).
SCALAR_LANE = 4


def _draw(rng, fmt, shape):
    steps = rng.integers(-8, 9, size=shape)
    return steps.astype(np.float64) * (1.0 if fmt is ElementFormat.INT8 else 0.25)


def _encode(values, fmt):
    codes = bytes(encode_element(float(v), fmt) for v in values.ravel())
    decoded = np.array([decode_element(code, fmt) for code in codes])
    if not np.array_equal(decoded, values.ravel()):
        raise ValueError(f"vector_tour data is not exact in {fmt}")
    return codes


def reference(a, w, c):
    """The three R_ACC stores, as the program's comments describe them."""
    a0 = a[0]
    lanes = np.arange(LANES)
    first = np.maximum(2 * a0 + w * a0 - c * c, c * a0[SCALAR_LANE])
    second = np.where(lanes % 2 == 0, w * a0, 0) + np.where(
        lanes < LANES // 2, 2 * a0[SCALAR_LANE], 0
    )
    third = second.copy()
    third[:A_ROWS] = (a * w).sum(axis=1)
    third[A_ROWS] = (a[A_ROWS - 1] * w).max()
    return np.stack((first, second, third))


def prepare(workspace, *, dtype, seed):
    fmt = parse_dtype(dtype)
    rng = np.random.default_rng(seed)
    a = _draw(rng, fmt, (A_ROWS, LANES))
    w = _draw(rng, fmt, LANES)
    c = _draw(rng, fmt, LANES)
    mask = b"".join(MASK_SLOTS).ljust(ROW_BYTES, b"\x00")
    inp, out = workspace / "input.bin", workspace / "output.bin"
    inp.write_bytes(_encode(a, fmt) + _encode(w, fmt) + _encode(c, fmt) + mask)
    expected = reference(a, w, c)

    def check():
        raw = out.read_bytes()
        if len(raw) != output_rows(fmt) * ROW_BYTES:
            raise ValueError(
                f"vector_tour stored {len(raw)} bytes, expected {output_rows(fmt) * ROW_BYTES}"
            )
        acc_bytes = ACC_STORES * ACC_STORE_ROWS * ROW_BYTES
        if fmt is ElementFormat.INT8:
            acc = np.frombuffer(raw[:acc_bytes], dtype="<i4").reshape(ACC_STORES, LANES)
            _check_equal("R_ACC stores", acc, expected.astype(np.int64))
            aaq = np.frombuffer(raw[acc_bytes:], dtype=np.int8)
            relu = np.clip(np.maximum(expected[-1], 0), -128, 127)
            want = np.concatenate((relu, np.zeros(aaq.size - LANES)))
            _check_equal("POST_AAQ_REG store", aaq, want.astype(np.int64))
        else:
            acc = np.frombuffer(raw[:acc_bytes], dtype="<f4").reshape(ACC_STORES, LANES)
            check_close(acc, expected.astype(np.float32))

    return PreparedCase({"dtype": fmt.value}, {"input_path": inp, "output_path": out}, check)


def _check_equal(what, actual, expected):
    bad = np.argwhere(actual != expected)
    if bad.size:
        index = tuple(bad[0])
        raise ValueError(
            f"vector_tour {what}: {len(bad)} elements differ; first at {index}: "
            f"got {actual[index]}, expected {expected[index]}"
        )


CASES = {
    "default": KernelCase(prepare, {"dtype": "int8", "seed": 16}),
    "int8": KernelCase(prepare, {"dtype": "int8", "seed": 16}),
    "e4": KernelCase(prepare, {"dtype": "e4", "seed": 16}),
    "e5": KernelCase(prepare, {"dtype": "e5", "seed": 16}),
    "e3": KernelCase(prepare, {"dtype": "e3", "seed": 16}),
}
