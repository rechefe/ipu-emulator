"""Column-softmax harness (FP32 wide-vector mode).

Softmax taken **down each column**: element ``x[r, col]`` is normalised against
all rows of the same column, ``{x[*, col]}``. This is the transpose of the
``softmax_rows`` family -- there the reduction collapses the 128 elements of one row
to a scalar; here every element is an *independent* column and the reduction runs
*across the rows* (across vectors), staying in-element the whole time. So there is
**no AGG and no fan-out**: the per-column max/sum are full 128-element vectors, and
ACC.MAX / ACC.ADD do the running reduce down the rows.

    softmax(x[r,col]) = 2^(c*(x[r,col]-cmax[col])) / SUM_r 2^(c*(x[r,col]-cmax[col]))

with ``c = log2(e)`` resident in a 128-element vector ``C_VEC`` (``2^(c*d)==e^d``,
matching the IPU's native ``exp2`` activation).

Layout (width >= 65, any value):
  Each row has width ``W`` padded up to the next multiple of 128, spanning
  ``cpr = ceil(W/128)`` consecutive 512 B chunks (no upper width bound). Row
  ``r``, chunk ``c`` lives at ``(r*cpr + c)*512``. A width that is not a multiple
  of 128 is padded with **0.0**; those filler elements are *separate columns* (never
  part of a real column's reduce), computed harmlessly and dropped on read-back.
  ``rows`` is an ordinary loop bound -- any count works (no group cap, unlike the
  row apps, because the per-column scalars are full vectors, not packed-per-row).

Per-chunk-column resident vectors ``cmax[c]`` / ``rvec[c]`` (one full 128-element
vector each) hold the column max / 1-over-sum for chunk-column ``c``. The four
passes loop **outer over chunk c, inner over rows r** so each pass's running ACC
reduce sweeps all rows for one chunk-column before moving on.

    Pass 1 (reduce):  cmax[c]  = max_r (c * x[r,c])           -> staged XMEM
    Pass 2 (trip):    num[r,c] = 2^(c*x[r,c] - cmax[c])       -> NUM region
    Pass 3 (reduce):  sum[c]   = SUM_r num[r,c]; rvec=1/sum    -> staged XMEM
    Pass 4 (trip):    out[r,c] = num[r,c] * rvec[c]           -> OUT region

Usage::

    from ipu_apps.kernels.softmax.softmax_columns.app import SoftmaxColumnsApp

    app = SoftmaxColumnsApp(
        inst_path="softmax_columns.bin",
        input_path="logits.bin",      # rows * width * 4 B, FP32, row-major
        output_path="probs.bin",
        rows=64,
        width=128,                    # pow2, 128..256 (padded up if needed)
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from ipu_apps.kernels.softmax.app import LANES, SoftmaxApp, softmax_spec

# Widths 1..64 are better served by softmax_columns_packed (several whole rows
# per vector). From 65 up, packing would round the group width to 128 -- one
# row per vector -- so this kernel is the right home for 65..127 even though
# those rows leave 128 - width elements idle.
MIN_WIDTH = 65


class SoftmaxColumnsApp(SoftmaxApp):
    """Column-softmax over ``rows`` x ``width`` FP32 logits (width >= 65).

    Width is padded to ``chunks_per_row`` whole 128-element chunks; padding
    elements are separate all-zero columns, dropped on read-back. The per-column
    max and 1/sum vectors hold one chunk per chunk-column.
    """

    dim = 0

    def _geometry(self):
        self.chunks_per_row = -(-self.width // LANES)
        return self.chunks_per_row * LANES

    def _layout_resident(self):
        self.max_addr = self._alloc(self.chunks_per_row)
        self.rvec_addr = self._alloc(self.chunks_per_row)

    def _crs(self):
        return {
            9: self.chunks_per_row,    # row stride in XMEM rows
            11: self.rows,             # row loop bound
            13: self.chunks_per_row,   # chunk-column loop bound
        }


def _explain(q):
    cpr = -(-q.width // LANES)
    return (f"width ({q.width}) >= {MIN_WIDTH}: per-element running ACC reduce down "
            f"the rows over {cpr} chunk-column(s), no AGG and no row-group cap.")


def _caveats(q):
    padded = -(-q.width // LANES) * LANES
    if padded == q.width:
        return ()
    return (f"width {q.width} pads to {padded} elements, so {padded - q.width} of "
            f"every {padded} elements sit idle ({q.width / padded:.0%} "
            f"utilisation). A chunk costs the same regardless of how many elements "
            f"carry real columns, so this runs at the cost of width {padded}.",)


SPEC = softmax_spec(
    SoftmaxColumnsApp, along_rows=False,
    supports=lambda q: (None if q.width >= MIN_WIDTH else
                        f"width ({q.width}) < {MIN_WIDTH}: at this width several whole "
                        f"rows fit in one vector, which the packed column kernel exploits"),
    explain=_explain,
    caveats=_caveats,
    cost=lambda q: 1.0,
)
