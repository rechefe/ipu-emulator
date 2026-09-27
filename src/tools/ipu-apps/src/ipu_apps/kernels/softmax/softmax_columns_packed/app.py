"""Packed sub-128 column-softmax harness (FP32 wide-vector mode).

Companion to ``softmax_columns`` for **narrow** rows: real width ``W <= 64``.
Softmax is still taken **down each column** (``x[r,col]`` normalised across all
rows of that column), but because a row is shorter than the 128-element datapath,
several whole rows are **packed side-by-side into one 128-element vector** to keep
the elements busy:

    W padded up to the next power of two in {16, 32, 64} (15->16, 17..31->32,
    33..63->64; exact 16/32/64 unchanged). rpv = 128 / W_pad in {2, 4, 8} rows
    share one vector: elements [0:W_pad)=row g, [W_pad:2*W_pad)=row g+1, ...

Each element is still an *independent* column, and the column reduce runs **down the
row dimension** (across vectors) exactly as in ``softmax_columns`` -- so the four
passes are the same per-element ACC.MAX / ACC.ADD kernel with **one chunk per packed
vector**. Packing just means each 128-element reduce now advances rpv input rows at
once. ``num_vectors = ceil(rows / rpv)``; the last vector zero-pads any missing
rows (a missing row is an all-zero column group, dropped on read-back).

Two kinds of padding element exist, both zero in the input and both kept zero in the
output:
  * intra-group width padding (W_real..W_pad within each group) when W isn't a
    clean power of two;
  * the tail of the last vector when rows isn't a multiple of rpv.
Because the MULT hardware mask is inert in wide FP32 mode, padding is zeroed on
chip by a **resident keep-mask vector** (1.0 in real-data elements, 0.0 elsewhere)
multiplied into the Pass 4 output. Teardown additionally emits a dense
``rows x W_real`` file.

    softmax(x[r,col]) = 2^(c*(x[r,col]-cmax[col])) / SUM_r 2^(c*(x[r,col]-cmax[col]))

with ``c = log2(e)`` resident in a 128-element vector ``C_VEC`` (``2^(c*d)==e^d``).

Usage::

    from ipu_apps.kernels.softmax.softmax_columns_packed.app import SoftmaxColumnsPackedApp

    app = SoftmaxColumnsPackedApp(
        inst_path="softmax_columns_packed.bin",
        input_path="logits.bin",      # rows * W_real * 4 B, FP32, row-major
        output_path="probs.bin",
        rows=100,
        width=16,                     # real width, 1..64
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import numpy as np

from ipu_apps.kernels.softmax.app import LANES, SoftmaxApp, partition_size, softmax_spec

MAX_WIDTH = 64               # real width <= 64 (>= 65 -> softmax_columns)
# Padding fill: loses every max and exp2 underflows to 0 in the sum, so the
# cross-group fold ignores missing-row groups and width padding for free.
NEG_PAD = -1.0e30


class SoftmaxColumnsPackedApp(SoftmaxApp):
    """Column-softmax over ``rows`` x ``width`` FP32 logits, width <= 64.

    Width is padded to ``group_width`` in {16, 32, 64}, so ``rows_per_vec``
    rows pack into each 128-element vector; a cross-group fold then combines
    the groups. All padding holds NEG_PAD.
    """

    dim = 0
    fill = NEG_PAD

    def _geometry(self):
        self.group_width = partition_size(self.width)
        self.rows_per_vec = LANES // self.group_width
        return self.group_width

    def _layout_resident(self):
        self.keep_addr = self._alloc()      # the .asm derives it as CVEC row + 1
        super()._layout_resident()
        self.scratch_addr = self._alloc()   # fold scratch

    def _write_resident(self, state):
        # 1.0 on each group's real columns, 0.0 on its width padding.
        keep = np.zeros((self.rows_per_vec, self.group_width), dtype="<f4")
        keep[:, : self.width] = 1.0
        state.xmem.write_address(self.keep_addr, keep.tobytes())

    def _crs(self):
        return {
            9: LANES,                          # fold's duplicate-load element index
            11: self.device_rows,              # vector loop bound
            12: self.scratch_addr // (LANES * 4),
            13: self.group_width,              # fold rc_idx step (elements)
            14: self.rows_per_vec,             # fold step count
        }


def _explain(q):
    w_pad = partition_size(q.width)
    return (f"width ({q.width}) <= {MAX_WIDTH}: packed column kernel, {LANES // w_pad} rows "
            f"packed per {LANES}-element vector (padded width {w_pad}).")


SPEC = softmax_spec(
    SoftmaxColumnsPackedApp, along_rows=False,
    supports=lambda q: (None if q.width <= MAX_WIDTH else
                        f"packs whole rows into one {LANES}-element vector, so it needs a "
                        f"width <= {MAX_WIDTH}; this input is {q.width} wide"),
    explain=_explain,
    # Fits several rows per vector, so it beats the general column kernel
    # wherever it applies.
    cost=lambda q: 0.0,
    tags=("packed",),
)
