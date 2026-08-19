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

    from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp

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

import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, no, yes
from ipu_apps.softmax._spec_support import (
    WIDE_VECTOR_ONLY,
    positive_dims,
    softmax_query,
)

if TYPE_CHECKING:
    pass

# -- Constants --------------------------------------------------------------

LANES = 128                  # elements per vector (fixed by the 128-element datapath)
CHUNK_BYTES = LANES * 4      # 512 bytes per FP32 vector
MAX_WIDTH = 64               # this app: real width <= 64 (>=65 -> softmax_columns)

INPUT_BASE_ADDR = 0x10000    # x   (input logits, FP32)

LOG2E = math.log2(math.e)    # c = 1.4426950408889634
NEG_PAD = -1.0e30            # padding-element fill: loses every max, exp2 -> 0 in the sum


def _pack_pow2(n: int) -> int:
    """Smallest power of two >= n (used for the packed group width, <=64)."""
    p = 1
    while p < n:
        p <<= 1
    return p


class SoftmaxColumnsPackedApp(IpuApp):
    """Column-softmax over ``rows`` x ``width`` FP32 logits, width <= 64 (packed).

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Path to the input logits binary (rows * width * 4 B, FP32,
                     row-major). ``width`` is the *real* (unpadded) width.
        output_path: Optional path to write the softmax output (dense rows x
                     width).
        rows:        Number of rows (>= 1; any count).
        width:       Real elements per row, 1..64. Padded up to the next power of
                     two W_pad in {16, 32, 64}; rpv = 128 / W_pad rows pack per
                     128-element vector.
    """

    def __init__(self, *, rows: int, width: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.rows = int(rows)
        self.width = int(width)

        # Delegate to the registry declaration rather than restating the bounds:
        # SPEC.supports is the single source of truth for this kernel's domain.
        # (The message this replaced still advised "use softmax_columns for
        # width >= 128" long after that boundary had moved to 65 -- exactly the
        # drift a single source prevents.)
        SPEC.guard(shape=(self.rows, self.width), dim=0)

        # Packed group width = next pow2, but at least 16 (rpv <= 8 keeps the
        # layout simple and matches the requested {16,32,64} bands).
        self.group_width = max(16, _pack_pow2(self.width))   # W_pad in {16,32,64}
        self.rows_per_vec = LANES // self.group_width        # rpv in {8,4,2}
        self.num_vectors = (self.rows + self.rows_per_vec - 1) // self.rows_per_vec
        self._layout()

    def _layout(self) -> None:
        """Three big regions (input / num / output), each num_vectors*512 B,
        back-to-back; resident vectors (C_VEC, keep-mask, cmax, rvec -- one chunk
        each) above. cmax/rvec are single full chunks (cpr == 1 here).
        """
        region = self.num_vectors * CHUNK_BYTES
        step = (region + 0xFFFF) & ~0xFFFF       # 64 KiB align
        self.input_base = INPUT_BASE_ADDR
        self.num_base = self.input_base + step
        self.output_base = self.num_base + step
        scalars = self.output_base + step
        self.cvec_addr = scalars                 # C_VEC      (1 chunk)
        self.keep_addr = scalars + CHUNK_BYTES   # keep-mask  (1 chunk)
        self.cmax_addr = scalars + 2 * CHUNK_BYTES
        self.rvec_addr = scalars + 3 * CHUNK_BYTES
        self.scratch_addr = scalars + 4 * CHUNK_BYTES  # fold scratch (1 chunk)

    # -- packing ------------------------------------------------------------

    def _lane_of(self, group_in_vec: int, col: int) -> int:
        """Lane index for column ``col`` of the ``group_in_vec``-th row in a
        vector."""
        return group_in_vec * self.group_width + col

    def _pack_input(self) -> bytes:
        """Pack row-major (rows x W_real) float32 into vectors of rpv rows each.
        Row r -> vector (r // rpv), group (r % rpv), element group_offset + col.

        All padding elements -- both intra-group width padding and the last vector's
        missing-row groups -- are initialised to a large negative (NEG_PAD) rather
        than zero. The cross-group fold then ignores them for free: NEG_PAD never
        wins the max, and exp2(c*NEG_PAD - cmax) underflows to 0 so it adds nothing
        to the sum. (Width-pad elements are separate junk columns dropped on read-back;
        missing-row groups must not pollute a real column's reduce -- this is the
        all-negative-column hazard, real here because the fold mixes groups.)
        """
        raw = self.input_path.read_bytes()
        flat = struct.unpack(f"<{self.rows * self.width}f", raw[: self.rows * self.width * 4])
        packed = bytearray(struct.pack("<f", NEG_PAD) * (self.num_vectors * LANES))
        for r in range(self.rows):
            v = r // self.rows_per_vec
            g = r % self.rows_per_vec
            base = v * CHUNK_BYTES + self._lane_of(g, 0) * 4
            row = flat[r * self.width:(r + 1) * self.width]
            struct.pack_into(f"<{self.width}f", packed, base, *row)
        return bytes(packed)

    def _keep_mask(self) -> bytes:
        """128-element FP32 keep-mask: 1.0 in real-column elements of every group,
        0.0 in the intra-group width-padding elements. (The last-vector missing-row
        groups are handled by the input being zero -> exp2(.)=finite but the
        teardown drops them; the keep-mask only needs the per-group width.)
        """
        mask = [0.0] * LANES
        for g in range(self.rows_per_vec):
            for col in range(self.width):
                mask[self._lane_of(g, col)] = 1.0
        return struct.pack("<128f", *mask)

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        state.dtype = DType.INT8  # mode byte; wide flag overrides element width
        return state

    def setup(self, state: "IpuState") -> None:
        state.xmem.write_address(self.input_base, self._pack_input())

        state.xmem.write_address(self.cvec_addr, struct.pack("<128f", *([LOG2E] * LANES)))
        state.xmem.write_address(self.keep_addr, self._keep_mask())

        # CR0 == 0, CR1 == 1 are READ-ONLY (CR1 = 1.0 scalar + loop increment).
        # .asm XMEM operands are ROW numbers (one row = CHUNK_BYTES), not byte
        # addresses -- see issue #179 -- so all base/stride CRs below are rows,
        # EXCEPT CR13/CR14 which feed MULT.RC's rc_idx, an ELEMENT offset into
        # r_cyclic (matches LDR_CYCLIC_MULT_REG's index -- issue #182/PR #196).
        state.regfile.set_cr(2, self.output_base // CHUNK_BYTES)
        state.regfile.set_cr(3, self.cvec_addr // CHUNK_BYTES)
        state.regfile.set_cr(4, self.num_base // CHUNK_BYTES)
        state.regfile.set_cr(5, self.cmax_addr // CHUNK_BYTES)
        state.regfile.set_cr(6, self.rvec_addr // CHUNK_BYTES)
        state.regfile.set_cr(7, 1)                        # vector stride: advance one row per vector
        # CR8 is free (the keep-mask row, used once in Pass 3, is computed in
        # .asm as CVEC row + 1 instead of a dedicated CR).
        # CR9 = LANES (128): the ELEMENT index for the fold's duplicate
        # LDR_CYCLIC_MULT_REG (loads the same partial into ring slot 1, right
        # after slot 0, so every fold rc_idx in [0, (rpv-1)*W] stays inside
        # real, loaded data -- see the .asm fold comment).
        state.regfile.set_cr(9, LANES)
        state.regfile.set_cr(10, self.input_base // CHUNK_BYTES)  # input base row
        state.regfile.set_cr(11, self.num_vectors)        # vector loop bound (rows reduced)
        state.regfile.set_cr(12, self.scratch_addr // CHUNK_BYTES)  # fold scratch chunk row
        # CR13/CR14: cross-group fold walk. The fold combines the rows_per_vec
        # (rpv) groups packed into one 128-element partial by duplicating it into
        # two adjacent ring slots then reading rc_idx = 0, W, ..., (rpv-1)*W
        # (rpv steps, W=group_width) via a running ACC.MAX/ACC.ADD -- see the
        # .asm comment for the full derivation.
        state.regfile.set_cr(13, self.group_width)        # W: per-step rc_idx increment (elements)
        state.regfile.set_cr(14, self.rows_per_vec)        # rpv: fold step count (loop bound)

        # CR15 = dstructure, valid_elements = 128, pad_mode = ZERO (default;
        # every element live, no masking except where a mask_offset says so).
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.num_vectors * CHUNK_BYTES)
        out = bytearray(self.rows * self.width * 4)
        for r in range(self.rows):
            v = r // self.rows_per_vec
            g = r % self.rows_per_vec
            src = v * CHUNK_BYTES + self._lane_of(g, 0) * 4
            dst = r * self.width * 4
            out[dst:dst + self.width * 4] = raw[src:src + self.width * 4]
        Path(self.output_path).write_bytes(bytes(out))

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain -- the constructor
# guard delegates to it rather than restating the bounds.


def _supports(**params):
    q = softmax_query(params["shape"], params["dim"])
    bad = positive_dims(q)
    if bad:
        return no(bad)
    if q.along_rows:
        return no("reduces along rows, not down columns")
    if q.width > MAX_WIDTH:
        return no(
            f"packs whole rows into one {LANES}-element vector, so it needs a "
            f"width <= {MAX_WIDTH}; this input is {q.width} wide"
        )
    return yes()


def _build(**params):
    q = softmax_query(params["shape"], params["dim"])
    return {"rows": q.rows, "width": q.width}


def _explain(**params):
    q = softmax_query(params["shape"], params["dim"])
    w_pad = 16
    while w_pad < q.width:
        w_pad *= 2
    rpv = LANES // w_pad
    return (
        f"width ({q.width}) <= {MAX_WIDTH}: packed column kernel, {rpv} rows "
        f"packed per {LANES}-element vector (padded width {w_pad})."
    )


SPEC = KernelSpec(
    name="softmax_columns_packed",
    op="softmax",
    variant="columns_packed",
    app_class=SoftmaxColumnsPackedApp,
    asm="softmax_columns_packed.asm",
    tags=("fp32-wide", "packed"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: softmax_query(params["shape"], params["dim"]).bundle,
    # Fits several rows per vector, so it is strictly better than the general
    # column kernel wherever it applies.
    cost=lambda **params: 0.0,
)
