"""Long-row softmax harness (FP32 wide-vector mode).

Mix of ``softmax_rows`` (full 128-wide rows) and ``softmax_rows_partial``
(sub-128 rows): handles rows whose element count ``N > 128`` and is **not**
divisible by 128. Each row is ``full_chunks = N // 128`` full 128-element
chunks plus one tail of ``tail = N % 128`` elements (``tail > 0``).

    softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)

so the IPU's native ``exp2`` activation applies directly (``2^(c*d) == e^d``).
``c = log2(e)`` rides in a resident 128-element vector ``C_VEC``; the 1.0 scalar is
the read-only CR1.

Per-row scalars (``maxvec[r]`` / ``rvec[r]``) hold one value per row in a single
128-element vector, so this version processes up to 128 rows in one group (no group
loop yet -- see ``softmax_rows`` for the >128-row group machinery).

The novelty vs the two base apps is the **cross-chunk reduction**: a single
row's max (Pass 1) and sum (Pass 3) now span ``cpr = full_chunks + 1`` chunks,
so each pass keeps a *running* per-row scalar -- ``AGG.*.FIRST`` on chunk 0 then
``AGG.*`` (running) on chunks 1.. -- combining all of a row's chunks into one
slot before the row's normalisation.

Layout (row-contiguous, chunk-padded): row ``r`` occupies ``cpr`` consecutive
512 B chunks at ``BASE + r*cpr*512``; the tail chunk's unused elements
(``tail..127``) are zero-padded. The full chunks reduce with
``valid_elements=128`` (CR15); the tail chunk reduces with
``valid_elements=tail`` (CR8) -- the dual-dstructure-CR trick from
``softmax_rows_partial``.

Usage::

    from ipu_apps.kernels.softmax.softmax_rows_long.app import SoftmaxRowsLongApp

    app = SoftmaxRowsLongApp(
        inst_path="softmax_rows_long.bin",
        input_path="logits.bin",     # rows * N * 4 bytes, FP32, row-major
        output_path="probs.bin",
        rows=8,
        n=300,                        # N>128, N%128 != 0
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from ipu_apps.kernels.softmax.app import LANES, SoftmaxApp, softmax_spec


class SoftmaxRowsLongApp(SoftmaxApp):
    """Row-softmax over ``rows`` rows of ``n > 128`` FP32 logits.

    Each row spans ``chunks_per_row`` 128-element chunks, the last one holding
    a ``tail`` of ``n % 128`` elements. When ``tail == 0`` there is no tail
    chunk at all; Passes 1/3 then run their tail block with valid_elements=0
    (CR8), an exact no-op for the running AGG.MAX / AGG.SUM.
    """

    def _geometry(self):
        self.full_chunks, self.tail = divmod(self.n, LANES)
        self.chunks_per_row = self.full_chunks + (1 if self.tail else 0)
        return self.chunks_per_row * LANES

    def _crs(self):
        return {
            8: self.tail,              # tail dstructure: valid_elements = tail
            9: LANES,                  # maxvec R1 byte base
            11: self.rows,             # row loop bound
            12: self.full_chunks,      # full-chunk loop bound
            13: self.chunks_per_row,   # chunks per row
            14: self.chunks_per_row,   # row stride in XMEM rows
        }


def _explain(q):
    full, tail = divmod(q.n, LANES)
    shape = (f"{full} full chunks + {tail}-element tail" if tail
             else f"exactly {full} full chunks (no tail chunk)")
    return f"n ({q.n}) > {LANES}: {shape}, reduced with a running cross-chunk AGG."


SPEC = softmax_spec(
    SoftmaxRowsLongApp, along_rows=True,
    supports=lambda q: (None if q.n > LANES else
                        f"splits a row across several {LANES}-element chunks, so it needs a "
                        f"row longer than {LANES}; this row has {q.n}"),
    explain=_explain,
    cost=lambda q: 1.0,
)
