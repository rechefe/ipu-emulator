"""Every softmax app must write its output in the SAME layout as its input.

The apps use very different on-device layouts -- chunk-padded rows, pow2 width
padding, P logical rows packed per 128-lane vector -- and several of them
un-pack in the middle of the computation. None of that is allowed to leak into
the files: a caller hands in a row-major (rows x cols) float32 matrix and must
get a row-major (rows x cols) float32 matrix back, same size, same element
order, so that `np.frombuffer(out).reshape(x.shape)` is always correct.

This is a layout contract, deliberately separate from the per-app numerical
tests: it is checked by reshaping the raw output file with NO app-specific
un-packing knowledge. A regression that merely misplaces elements (rather than
computing them wrongly) would slip past a test that un-packs before comparing,
which is exactly how the packed-output discrepancy in softmax_rows_partial went
unnoticed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_columns import SoftmaxColumnsApp
from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp
from ipu_apps.softmax.softmax_rows import SoftmaxRowsApp
from ipu_apps.softmax.softmax_rows_long import SoftmaxRowsLongApp
from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp

_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/softmax"

# (id, app class, asm dir name, ctor kwargs, (rows, cols), softmax axis).
# Shapes are chosen so each app hits its padding-heavy regime -- that is where
# an input/output layout divergence can hide.
CASES = [
    ("rows",            SoftmaxRowsApp,          "softmax_rows",
     dict(rows=6), (6, 128), 1),
    # N < ps (20 of 32 lanes) AND rows % P != 0 (10 rows, P=4 -> padded to 12):
    # both padding kinds at once.
    ("partial_pad",     SoftmaxRowsPartialApp,   "softmax_rows_partial",
     dict(n=20, rows=10), (10, 20), 1),
    # N == ps and rows % P == 0: no padding at all, must still round-trip.
    ("partial_exact",   SoftmaxRowsPartialApp,   "softmax_rows_partial",
     dict(n=32, rows=8), (8, 32), 1),
    ("partial_p8",      SoftmaxRowsPartialApp,   "softmax_rows_partial",
     dict(n=5, rows=7), (7, 5), 1),
    # n % 128 != 0 -> a padded tail chunk on device.
    ("long_tail",       SoftmaxRowsLongApp,      "softmax_rows_long",
     dict(n=300, rows=5), (5, 300), 1),
    ("long_exact",      SoftmaxRowsLongApp,      "softmax_rows_long",
     dict(n=256, rows=3), (3, 256), 1),
    # width not a multiple of 128 -> padding lanes on device.
    ("columns_pad",     SoftmaxColumnsApp,       "softmax_columns",
     dict(rows=9, width=200), (9, 200), 0),
    ("columns_exact",   SoftmaxColumnsApp,       "softmax_columns",
     dict(rows=9, width=128), (9, 128), 0),
    # width not a pow2 and rows % rows_per_vec != 0.
    ("cols_packed_pad", SoftmaxColumnsPackedApp, "softmax_columns_packed",
     dict(rows=10, width=20), (10, 20), 0),
    ("cols_packed_exact", SoftmaxColumnsPackedApp, "softmax_columns_packed",
     dict(rows=8, width=32), (8, 32), 0),
]


def _reference(x: np.ndarray, axis: int) -> np.ndarray:
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


@pytest.mark.parametrize(
    "app_cls,asm_dir,kwargs,shape,axis",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_output_file_matches_input_layout(app_cls, asm_dir, kwargs, shape, axis):
    x = (np.random.RandomState(sum(shape)).randn(*shape) * 3.0).astype(np.float32)
    asm_path = next((_SRC / asm_dir).glob("*.asm"))

    with tempfile.TemporaryDirectory() as tmp:
        inst = Path(tmp) / "prog.bin"
        assemble_to_bin_file(asm_path.read_text(), str(inst))
        inp = Path(tmp) / "in.bin"
        outp = Path(tmp) / "out.bin"
        inp.write_bytes(x.tobytes())

        app_cls(inst_path=inst, input_path=inp, output_path=outp,
                **kwargs).run(max_cycles=20_000_000)

        in_bytes = inp.stat().st_size
        out_bytes = outp.stat().st_size
        raw = np.frombuffer(outp.read_bytes(), dtype=np.float32)

    # 1. Same file size -- no padding leaked out, nothing truncated.
    assert out_bytes == in_bytes, (
        f"output file is {out_bytes} B but input was {in_bytes} B: the "
        f"on-device padding is leaking into the output layout"
    )

    # 2. A naive reshape (no app-specific un-packing) must be correct, which
    #    pins element ORDER, not just size.
    out = raw.reshape(shape)
    assert np.abs(out - _reference(x, axis)).max() < 1e-4
