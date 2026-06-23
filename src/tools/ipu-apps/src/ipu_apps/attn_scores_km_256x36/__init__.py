"""Agent D — kQᵀ → key-major attention scores (one head).

Computes, for a single attention head h (head_dim D=36, N=256 tokens)::

    S[i, s] = sum_c Q[i, c] * K[s, c]        i, s in [0, 256), c in [0, 36)

and stores S **key-major**: S[i, s] at SBASE + (s*256 + i)*4 (int32/fp32 words),
so that each key column S[:, s] is contiguous (feeds the downstream softmax chain).

Activation layout (canonical, channel-major, multi-head)::

    Q/K element [token t, head-channel c of head h] lives at
        BASE + (h*36 + c) * 256 + t

The kernel needs K[s, 0:35] contiguous to load into R0 (the scalar operand),
so :func:`_load_k_keymajor` rearranges this head's K into a key-major XMEM
scratch (K[s, :] at KBASE_KM + s*128).  Q stays channel-major: each inner-loop
step loads a channel column (128 queries, one channel) straight into R_CYCLIC.

Usage::

    from ipu_apps.attn_scores_km_256x36 import AttnScoresKM256x36App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_TOK    = 256          # queries = keys
D        = 36           # head_dim
N_TG     = 2            # query groups of 128
N_TPG    = 128
N_HEADS  = 4            # channels in the canonical input file = N_HEADS * D

QBASE    = 0x00000      # Q channel-major (verbatim head slice)
KBASE_KM = 0x10000      # K rearranged key-major (K[s,:] at +s*128)
SBASE    = 0x20000      # S key-major words

K_STRIDE     = 128                  # key-major K row stride (D padded to 128)
OUTPUT_ROW_BYTES = 512              # one (s,g) score row = 128 words

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


def _load_q_channel_major(state: "IpuState", q_path: str | Path, head: int) -> None:
    """Copy head `head`'s 36 channel columns of Q verbatim into XMEM at QBASE.

    Input file is canonical channel-major: Q[t, h*36+c] at (h*36+c)*256 + t.
    The kernel addresses this head as QBASE + c*256 + t, so we slice the head's
    contiguous channel block (channels h*36 .. h*36+35) to the front.
    """
    raw = Path(q_path).read_bytes()
    base = head * D * N_TOK
    state.xmem.write_address(QBASE, bytearray(raw[base : base + D * N_TOK]))


def _load_k_keymajor(state: "IpuState", k_path: str | Path, head: int) -> None:
    """Rearrange head `head`'s K from channel-major into key-major XMEM scratch.

    Source K[s, c] at (head*36 + c)*256 + s.  Destination K[s, :] contiguous at
    KBASE_KM + s*128 (36 channels, padded to 128 bytes).
    """
    raw = Path(k_path).read_bytes()
    head_base = head * D * N_TOK
    for s in range(N_TOK):
        row = bytearray(K_STRIDE)
        for c in range(D):
            row[c] = raw[head_base + c * N_TOK + s]
        state.xmem.write_address(KBASE_KM + s * K_STRIDE, row)


class AttnScoresKM256x36App(IpuApp):
    """kQᵀ → key-major scores, single head (D=36, N=256)."""

    def __init__(self, *, dtype: str | DType = "INT8", head: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        # Q is the "input"; K is the "weights" file slot.
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype
        self.head = head

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype
        _load_q_channel_major(state, self.input_path, self.head)
        _load_k_keymajor(state, self.weights_path, self.head)

        # CR1 (≡1) is read-only hardwired; cr0 (=QBASE=0x0) matches hardwired 0.
        state.regfile.set_cr(0, QBASE)
        state.regfile.set_cr(2, SBASE)
        state.regfile.set_cr(9, KBASE_KM)
        state.regfile.set_cr(5, -256)        # g=0 channel-column startup
        state.regfile.set_cr(6, -128)        # g=1 channel-column startup
        state.regfile.set_cr(7, -1)          # fixed_idx c startup
        state.regfile.set_cr(8, D - 2)       # c-loop bound: first=0, width=D → D-2 = 34

        state.regfile.set_lr(0, 0)           # R_CYCLIC index 0
        state.regfile.set_lr(2, 256)         # channel stride in Q
        state.regfile.set_lr(3, 512)         # output store stride
        state.regfile.set_lr(6, D - 2)       # c-loop bound = 34
        state.regfile.set_lr(7, 0)           # output byte pointer
        state.regfile.set_lr(8, -K_STRIDE)   # key byte offset startup (-128 → first live 0)
        state.regfile.set_lr(9, 0)           # key counter
        state.regfile.set_lr(10, N_TOK)      # key-loop limit
        state.regfile.set_lr(12, K_STRIDE)   # key stride into K scratch

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                SBASE, OUTPUT_ROW_BYTES, N_TOK * N_TG,
            )
