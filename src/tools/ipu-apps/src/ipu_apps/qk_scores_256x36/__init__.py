"""QKᵀ scores harness (Agent C), one attention head.

Computes the query-major score matrix for a single attention head::

    S[i, s] = sum_{c=0..35} Q[i, c] * K[s, c]      for i, s in [0, 256)

Inputs Q, K are logically channel-major (head_dim D=36, N=256 tokens). K is
loaded channel-major verbatim; Q is staged query-major (a gather of its strided
channels) so one query's 36 head-channels load into r0 with a single
``LDR_MULT_REG`` — the matmul broadcast template (scalar = Q[i,c] from r0,
vector = K's channel-c column in r_cyclic, ``MULT.RC.VE``).

The score row is stored RAW (full-precision R_ACC, 512 B per 128-key group,
query-major) so softmax (Agent A) reads unquantized scores. No AGG.

Two modes:
  * wide-vector FP32 (``state.wide_vector_debug``): elements are 4-byte FP32.
  * INT8 / FP8 dtype: elements are 1 byte; output accumulators are still
    512-byte (FP32 for FP8, INT32 for INT8).

Usage::

    from ipu_apps.qk_scores_256x36 import QkScores256x36App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N    = 256          # tokens (queries = keys)
D    = 36           # head_dim (contraction width)
N_TG = 2            # key groups of 128 keys each
N_TPG = 128         # keys per group

K_BASE      = 0x00000
QROW_BASE   = 0x40000
S_BASE      = 0x80000

QROW_STRIDE      = 512   # bytes per staged query row (>= D*4, fits one r_cyclic)
OUTPUT_ROW_BYTES = 512   # R_ACC store width (always 512 B / 128 lanes)

_DTYPE_MAP = {
    "INT8":   DType.INT8,
    "int8":   DType.INT8,
    "E4":     DType.E4,
    "fp8_e4": DType.E4,
    "E5":     DType.E5,
    "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


class QkScores256x36App(IpuApp):
    """One-head QKᵀ → query-major scores application harness.

    ``dtype`` selects the narrow-mode element format (INT8 / FP8). When the
    supplied ``state`` has ``wide_vector_debug`` set, inputs are read as raw
    FP32 (4-byte elements) instead, for the primary correctness check.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.query_path = Path(self.query_path)
        self.key_path = Path(self.key_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    # -- staging -------------------------------------------------------------

    def _stage_inputs(self, state: "IpuState", elem: int) -> None:
        """Write K channel-major and Q query-major into XMEM.

        ``elem`` is the per-element byte width (1 for INT8/FP8, 4 for FP32).
        Input files are stored channel-major: element [token t, channel c] at
        (c*N + t)*elem.
        """
        q_raw = self.query_path.read_bytes()
        k_raw = self.key_path.read_bytes()

        # K: channel-major verbatim. Column c (256 keys) is contiguous already;
        #    write at K_BASE + c*(N*elem). The kernel loads two 128-key chunks.
        for c in range(D):
            col = k_raw[(c * N) * elem : (c * N + N) * elem]
            state.xmem.write_address(K_BASE + c * (N * elem), bytearray(col))

        # Q: gather the strided channels into contiguous query-major rows.
        #    QROW[i] = Q[i, 0..35] at QROW_BASE + i*QROW_STRIDE (rest zero-pad).
        for i in range(N):
            row = bytearray(QROW_STRIDE)
            for c in range(D):
                src = (c * N + i) * elem
                row[c * elem : c * elem + elem] = q_raw[src : src + elem]
            state.xmem.write_address(QROW_BASE + i * QROW_STRIDE, row)

    def setup(self, state: "IpuState") -> None:
        wide = bool(getattr(state, "wide_vector_debug", False))
        elem = 4 if wide else 1
        if not wide:
            state.dtype = self.dtype

        self._stage_inputs(state, elem)

        k_stride   = N * elem          # bytes per K channel column (256 * elem)
        g0_start   = -k_stride         # g=0 K-data startup: first live = 0
        g1_start   = -k_stride + N_TPG * elem  # g=1 startup: first live = +128 keys

        # CR1 (≡1) is read-only hardwired; QROW base lives on CR9.
        state.regfile.set_cr(0, K_BASE)                 # data base
        state.regfile.set_cr(9, QROW_BASE)              # staged query rows
        state.regfile.set_cr(3, S_BASE)                 # group 0 output base
        state.regfile.set_cr(4, S_BASE + 512)           # group 1 output base
        state.regfile.set_cr(5, g0_start)               # g=0 K-data startup
        state.regfile.set_cr(6, g1_start)               # g=1 K-data startup
        state.regfile.set_cr(7, -1)                      # channel fixed_idx startup
        state.regfile.set_cr(8, D - 2)                   # contraction bound (34)

        state.regfile.set_lr(0, 0)                       # r_cyclic write-index / mask_shift
        state.regfile.set_lr(2, k_stride)                # K data stride per channel
        state.regfile.set_lr(3, N_TG * 512)              # output stride per query (1024)
        state.regfile.set_lr(6, D - 2)                   # contraction BLT bound
        state.regfile.set_lr(7, 0)                       # output query byte offset
        state.regfile.set_lr(8, 0)                       # Q-row byte offset
        state.regfile.set_lr(9, 0)                       # query counter
        state.regfile.set_lr(10, N)                      # query-loop limit
        state.regfile.set_lr(12, QROW_STRIDE)            # Q-row stride per query

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # N queries × N_TG groups × 512 B, in query-major group order:
            #   row (i, g) at S_BASE + i*1024 + g*512.
            dump_xmem_to_binary(
                state, self.output_path,
                S_BASE, OUTPUT_ROW_BYTES, N * N_TG,
            )
