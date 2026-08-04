"""kQᵀ → key-major attention scores (Layer 4), one selected head.

Computes, for the selected attention head ``head`` and every one of the
``P = 4`` streams of Layer 4::

    S[p, i, s] = sum_c Q[p, i, c] * K[p, s, c]    i, s in [0, 64), c in [0, 48)

and stores S **key-major**: each key ``s`` owns a whole XMEM row holding the 64
query scores ``S[p, 0:64, s]`` in its leading lanes, so a key column is
contiguous (this is what the downstream broadcast attn@V consumes).

Layer 4 parameters: d = 192, N = 64 tokens per stream, P = 4 streams,
h = 4 heads, head_dim = 48.  ``head_dim = 48`` needs NO padding -- it is the
contraction LOOP COUNT (bound ``lr6 = D - 2 = 46``), not a lane count.  Lanes
here are queries, and N = 64 <= LANES = 128, so a query axis fits in ONE row:
the L3 two-group split (``N_TG = 2``) collapses to ``N_TG = 1`` and the trailing
64 lanes of every Q / S row are unused padding.

This is the L4 port of ``attn_scores_km_256x36``, and it keeps that kernel's
4-head-input / one-head-selected interface: the Q and K files are canonical
channel-major over all ``P * N_HEAD`` (stream, head) blocks, and the ``head``
kwarg picks which head is scored.  The kernel then runs all ``P = 4`` streams
of that head in one go, producing ``P * N = 256`` key rows.

Activation layout (canonical, channel-major, per (stream, head) block)::

    Q/K element [stream p, head h, token t, channel c] lives at
        BASE + (((p*N_HEAD + h)*D) + c) * N + t

The kernel needs ``K[p, s, 0:48]`` contiguous to load into R0 (the scalar
operand), so :func:`_load_k_keymajor` rearranges the selected head's K into a
key-major XMEM scratch (``K[p, s, :]`` on row ``p*N + s``).  Q stays
channel-major: each inner-loop step loads a channel column (64 queries, one
channel) straight into R_CYCLIC.

This is the KEY-MAJOR chain: it pairs ONLY with ``attn_v_bcast_48``.  It must
NOT be mixed with the query-major ``qk_scores_64x48`` / ``attn_v_64x48`` pair --
the two chains produce bit-different results by design.

Usage::

    from ipu_apps.attn_scores_km_64x48 import AttnScoresKM64x48App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N       = 64            # tokens per stream (queries == keys)
D       = 48            # head_dim (contraction width)
P       = 4             # streams / partitions
N_HEAD  = 4             # attention heads in the input file
N_BLOCK = P * N_HEAD    # 16 (stream, head) blocks in the input file
N_TG    = 1             # query groups: 64 queries fit in one 128-lane row
N_TPG   = N             # queries per group

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path. INT8 is not a mode this
# kernel is written against; it belongs at the XMEM write boundary.
#
# XMEM .asm operands are ROW numbers (issue #179). Region bases are DERIVED
# from row counts, not hardcoded bytes: a byte map sized for 1-byte elements
# overflows 4x at FP32 and silently corrupts results.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

# Row counts are LANE counts -- element-width independent by construction.
# N = 64 <= LANES, so one Q channel column and one key's K vector are each ONE
# whole row (trailing lanes unused).
Q_CHAN_ROWS   = 1                            # rows per Q channel column
K_STRIDE_ROWS = 1                            # one key-major K row per key
OUT_ROWS      = 1                            # one r_acc store = one row (wide)

Q_STREAM_ROWS = D * Q_CHAN_ROWS              # 48: Q rows per stream
K_STREAM_ROWS = N * K_STRIDE_ROWS            # 64: K rows per stream
S_STREAM_ROWS = N * N_TG * OUT_ROWS          # 64: score rows per stream

Q_ROWS = P * Q_STREAM_ROWS                   # 192
K_ROWS = P * K_STREAM_ROWS                   # 256
S_ROWS = P * S_STREAM_ROWS                   # 256

QBASE_ROW    = 0
KBASE_KM_ROW = QBASE_ROW + Q_ROWS
SBASE_ROW    = KBASE_KM_ROW + K_ROWS

QBASE    = QBASE_ROW * ROW_BYTES
KBASE_KM = KBASE_KM_ROW * ROW_BYTES
SBASE    = SBASE_ROW * ROW_BYTES

# One store = one whole row; only the first N lanes hold query scores.
OUTPUT_ROW_BYTES = N * ELEM_BYTES            # 256 valid bytes per stored row


def _read_head(path: Path, head: int) -> np.ndarray:
    """Return the selected head's channel-major block as ``[P, D, N]``.

    The file is canonical channel-major over all ``N_BLOCK`` (stream, head)
    blocks: element [p, h, t, c] at ``((p*N_HEAD + h)*D + c)*N + t``.
    """
    arr = np.frombuffer(Path(path).read_bytes(), dtype=np.float32)
    want = N_BLOCK * D * N
    if arr.size != want:
        raise ValueError(f"{path}: expected {want} FP32 elements, got {arr.size}")
    arr = arr.reshape(P, N_HEAD, D, N)
    return arr[:, head]                      # [stream, channel, token]


def _load_q_channel_major(state: "IpuState", q_path: Path, head: int) -> None:
    """Stage the selected head's Q channel columns, one whole row each.

    Row ``p*D + c`` holds ``Q[p, 0:64, c]`` in its leading 64 lanes -- exactly
    what one ``LDR_CYCLIC_MULT_REG`` pulls into R_CYCLIC as the vector operand.
    Rows are never shared: the trailing 64 lanes are zero padding.
    """
    q = _read_head(q_path, head)                       # [P, D, N]
    buf = np.zeros((P * D, LANES), dtype=np.float32)
    for p in range(P):
        buf[p * D:(p + 1) * D, :N] = q[p]
    state.xmem.write_address(QBASE, bytearray(buf.tobytes()))


def _load_k_keymajor(state: "IpuState", k_path: Path, head: int) -> None:
    """Rearrange the selected head's K from channel-major into key-major rows.

    Row ``p*N + s`` holds ``K[p, s, 0:48]`` contiguously, which is what lets one
    ``LDR_MULT_REG`` pull a key's whole head-channel vector into R0 (the scalar
    operand, indexed by the contraction channel c).
    """
    k = _read_head(k_path, head)                       # [P, D, N]
    buf = np.zeros((P * N, LANES), dtype=np.float32)
    for p in range(P):
        buf[p * N:(p + 1) * N, :D] = k[p].T            # [key, channel]
    state.xmem.write_address(KBASE_KM, bytearray(buf.tobytes()))


class AttnScoresKM64x48App(IpuApp):
    """Layer-4 kQᵀ → key-major scores, one selected head, all 4 streams."""

    def __init__(self, *, head: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        # Q is the "input"; K is the "weights" file slot.
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        if not 0 <= head < N_HEAD:
            raise ValueError(f"head must be in [0, {N_HEAD}), got {head}")
        self.head = head

    def setup(self, state: "IpuState") -> None:
        _load_q_channel_major(state, self.input_path, self.head)
        _load_k_keymajor(state, self.weights_path, self.head)

        # CR0 (==0) and CR1 (==1) are hardwired read-only, so every base lives
        # on a writable CR -- QBASE_ROW happening to be 0 must not be relied on.
        state.regfile.set_cr(2, SBASE_ROW)
        state.regfile.set_cr(3, QBASE_ROW)
        state.regfile.set_cr(9, KBASE_KM_ROW)
        state.regfile.set_cr(5, -Q_CHAN_ROWS)    # channel-column startup (rows)
        state.regfile.set_cr(7, -1)              # fixed_idx c startup
        state.regfile.set_cr(8, D - 2)           # c-loop bound: first=0, width=D
        state.regfile.set_cr(10, P)              # 4: stream-loop limit
        state.regfile.set_cr(11, Q_STREAM_ROWS)  # 48: Q rows per stream
        state.regfile.set_cr(12, K_STREAM_ROWS)  # 64: K rows per stream

        state.regfile.set_lr(0, 0)               # R_CYCLIC index 0
        state.regfile.set_lr(2, Q_CHAN_ROWS)     # channel stride in Q (rows)
        state.regfile.set_lr(3, OUT_ROWS)        # output store stride (rows)
        state.regfile.set_lr(6, D - 2)           # c-loop bound = 46
        state.regfile.set_lr(7, 0)               # output row pointer
        state.regfile.set_lr(8, 0)               # key row offset (walks all P*N)
        state.regfile.set_lr(9, 0)               # key counter within a stream
        state.regfile.set_lr(10, N)              # key-loop limit
        state.regfile.set_lr(12, K_STRIDE_ROWS)  # key stride into K scratch (rows)
        state.regfile.set_lr(13, 0)              # Q stream base offset (rows)
        state.regfile.set_lr(14, 0)              # stream counter p

    def teardown(self, state: "IpuState") -> None:
        """Crop each key row's valid N queries out of its whole 512 B row.

        Every store wrote a full row (one key per row -- rows are never shared),
        but only the leading ``N * ELEM_BYTES`` bytes hold scores. The output
        file is the densely packed crop: ``P * N`` rows of N FP32, key (p, s) at
        row index ``p*N + s``.
        """
        if self.output_path is None:
            return
        stride = OUT_ROWS * ROW_BYTES
        parts = [
            bytes(state.xmem.read_address(SBASE + r * stride, OUTPUT_ROW_BYTES))
            for r in range(P * N * N_TG)
        ]
        Path(self.output_path).write_bytes(b"".join(parts))
