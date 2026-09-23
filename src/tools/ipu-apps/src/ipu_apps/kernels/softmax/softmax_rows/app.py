"""Row-softmax application harness (FP32 wide-vector mode).

Computes a numerically stable softmax over each of K rows of 128 FP32 elements
(any K >= 1), using the base-2 reformulation that matches the IPU's native
``exp2`` activation:

    softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)

so that ``2^(c * d) == e^d``. All scaling is done on the FP32 *vector* path
(``MULT.RC.VV`` / ``MULT.RC.VE``) because CR scalars are integer-only even in
wide mode (see docs/content/wide-vector-debug-mode.md). The constant ``c`` is
supplied as a resident 128-element vector ``C_VEC``.

Pass structure (see the .asm for the cycle-level layout):

    Pass 1  (reduction):  maxvec[r]  = max_j (c * x[r,j])         -> staged XMEM
    Pass 2  (trip):       num[r,j]   = 2^(c*x[r,j] - maxvec[r])   -> NUM region
    Pass 3  (reduction):  sumvec[r]  = SUM_j num[r,j];  rvec = 1/sumvec  -> staged
    Pass 4  (trip):       out[r,j]   = num[r,j] * rvec[r]         -> OUT region

Only Passes 2 and 4 write a full result matrix; Passes 1 and 3 produce one
128-element scalar vector each (maxvec / sumvec) that stays in R_ACC and is
staged once. See the project memory `softmax_rows_design` for the derivation
and the probe tests that validated each primitive.

Arbitrary row count: maxvec[r] / rvec[r] hold one scalar per row in a single
128-element vector, so the kernel can carry the per-row bookkeeping for at most 128
rows at once. All four passes therefore run once per group of up to 128 rows.
The group size is computed EXACTLY in the kernel as min(128, rows_remaining) --
there is no padding, so a 7-row input runs exactly 7 rows and a 130-row input
runs 128 then 2. The three big regions (input / num / output) are sized to the
row count and placed back-to-back per instance (see SoftmaxApp in softmax/app.py) so they don't
overlap for large inputs. Groups of 128 run at ~18 cyc/row.

Usage::

    from ipu_apps.kernels.softmax.softmax_rows.app import SoftmaxRowsApp

    app = SoftmaxRowsApp(
        inst_path="softmax_rows.bin",
        input_path="logits.bin",     # K * 512 bytes, FP32, row-major (any K)
        output_path="probs.bin",
        rows=500,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from ipu_apps.kernels.softmax.app import LANES, ROW_BYTES, SoftmaxApp, softmax_spec

__all__ = ["LANES", "ROW_BYTES", "SoftmaxRowsApp", "SPEC"]


class SoftmaxRowsApp(SoftmaxApp):
    """Row-softmax over ``rows`` x 128 FP32 logits (any row count >= 1).

    maxvec/rvec hold one scalar per row in a single 128-element vector, so the
    .asm processes rows in exact groups of at most 128 (no padding rows).
    """

    def __init__(self, *, rows: int = LANES, n: int = LANES, **kwargs) -> None:
        # The row length is fixed by the .asm, so both keep their defaults.
        super().__init__(rows=rows, n=n, **kwargs)

    def _crs(self):
        # CR9 = 128: group cap AND the R1 byte-index base for the maxvec select.
        return {9: LANES, 13: self.rows}


SPEC = softmax_spec(
    SoftmaxRowsApp, along_rows=True,
    supports=lambda q: (None if q.n == LANES else
                        f"handles exactly {LANES} elements per row; this row has {q.n}"),
    explain=lambda q: (f"n == {LANES} exactly: the full-width row kernel. Rows are processed "
                       f"in groups of at most {LANES}, so any row count works."),
    # Exact-width match: no padding, no chunking. Cheapest possible claim.
    cost=lambda q: 0.0,
)
