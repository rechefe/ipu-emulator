"""Long-row softmax harness (FP32 wide-vector mode).

Mix of ``softmax_rows`` (full 128-wide rows) and ``softmax_rows_partial``
(sub-128 rows): handles rows whose element count is ``N > 128``. Each row is
``full_chunks`` full 128-element chunks plus one tail chunk of ``tail``
elements: ``divmod(N, 128)``, except that when 128 divides ``N`` the last full
chunk is the tail (``tail = 128``), so there is never an empty tail chunk.

    softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)

so the IPU's native ``exp2`` activation applies directly (``2^(c*d) == e^d``).
``c = log2(e)`` rides in a resident 128-element vector ``C_VEC``; the 1.0 scalar is
the read-only CR1.

Per-row scalars (``maxvec[r]`` / ``rvec[r]``) hold one value per row in a single
128-element vector, so rows run in groups of up to 128, all four passes per
group (the same group loop as ``softmax_rows``).

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
        n=300,                        # N > 128
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from ipu_apps.kernel_registry import Table
from ipu_apps.kernels.softmax.app import LANES, SoftmaxApp, softmax_spec


class SoftmaxRowsLongApp(SoftmaxApp):
    """Row-softmax over ``rows`` rows of ``n > 128`` FP32 logits.

    Each row spans ``chunks_per_row`` 128-element chunks, the last one holding
    a ``tail`` of 1..128 elements (128 when 128 divides ``n``: the last full
    chunk is then the tail, reduced with valid_elements=128 through CR8).
    """

    def _geometry(self):
        self.full_chunks, self.tail = _split(self.n)
        self.chunks_per_row = self.full_chunks + 1
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


def _split(n):
    """``(full_chunks, tail)`` of an ``n``-element row; the tail is 1..128."""
    full, tail = divmod(n, LANES)
    return (full - 1, LANES) if tail == 0 else (full, tail)


def _explain(q):
    full, tail = _split(q.n)
    return (f"n ({q.n}) > {LANES}: {full} full chunks + {tail}-element tail, "
            f"reduced with a running cross-chunk AGG.")


def _tables(q):
    """XMEM tables as the Cache Unit sees them (see ``ipu_emu.table_trace``).

    A row spans ``chunks_per_row`` XMEM rows. Rows run in groups of up to 128,
    all four passes per group, each pass stepping back to the group's first
    row: one traversal with a group-sized jump_back (as in softmax_rows).
    """
    chunks_per_row = _split(q.n)[0] + 1
    device_rows = q.rows * chunks_per_row
    group = min(q.rows, LANES) * chunks_per_row
    return {
        "input": Table(cr=10, rows=device_rows, jump_back=group - 1),
        "num": Table(cr=4, rows=device_rows, jump_back=group - 1),
        "output": Table(cr=2, rows=device_rows),
        "c_vec": Table(cr=3, rows=1),
        "maxvec": Table(cr=5, rows=1),
        "rvec": Table(cr=6, rows=1),
    }


SPEC = softmax_spec(
    SoftmaxRowsLongApp, along_rows=True,
    supports=lambda q: (None if q.n > LANES else
                        f"splits a row across several {LANES}-element chunks, so it needs a "
                        f"row longer than {LANES}; this row has {q.n}"),
    explain=_explain,
    cost=lambda q: 1.0,
    tables=_tables,
)
