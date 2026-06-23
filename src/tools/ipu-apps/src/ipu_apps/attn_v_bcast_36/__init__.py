"""attn@V (broadcast kernel) harness — key-major scores, channel-major output.

Computes, per attention head h in [0,4):
    O[i, t] = sum_s P[i, s] * V[s, t]      i, s in [0, 256),  t in [0, 36)

Lanes = query tokens, scalar = V[s,t] indexed from R0++R1, ACC accumulates over
keys — the standard matmul broadcast (no AGG, no collision). The companion
query-major + AGG kernel is `attn_v_256x36`; this app is the key-major variant
and shares its V and O byte layouts so the two are directly comparable.

Inputs (1 byte/element):
  P key-major  : P[i, s] at PBASE + h*65536 + s*256 + i   (4 heads, head-major)
  V channel-major: V[s, chan] at VBASE + chan*256 + s,  chan = h*36 + t

Output (FP32 R_ACC, 512-byte group rows — transformer-matmul convention):
  O[i, t] at OBASE + chan*1024 + g*512 + local*4,  i = g*128 + local
  i.e. channel (h*36 + t) occupies 1024 bytes (2 groups of 128 FP32 lanes).

Usage::

    from ipu_apps.attn_v_bcast_36 import AttnVBcast36App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_TOK   = 256       # queries == keys
D       = 36        # head_dim
N_HEAD  = 4
N_CHAN  = N_HEAD * D  # 144 value channels total

# Byte layout in XMEM (mirrors attn_v_256x36).
PBASE = 0x00000     # P: 4 * 256 * 256        = 262144 B -> [0x00000, 0x40000)
VBASE = 0x40000     # V: 144 * 256            =  36864 B -> [0x40000, 0x49000)
OBASE = 0x50000     # O: 144 * 1024 (FP32)    = 147456 B -> [0x50000, 0x74000)

P_HEAD_STRIDE = 0x10000   # 256 keys * 256 queries
O_CHAN_BYTES  = 1024      # 2 groups * 512 B (FP32)

_DTYPE_MAP = {
    "INT8": DType.INT8, "int8": DType.INT8,
    "E4": DType.E4, "fp8_e4": DType.E4,
    "E5": DType.E5, "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


class AttnVBcast36App(IpuApp):
    """attn@V broadcast kernel harness (4 heads, N=256, head_dim=36)."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_path = Path(self.p_path)
        self.v_path = Path(self.v_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype
        # P and V are stored verbatim (already in the kernel's byte layout).
        state.xmem.write_address(PBASE, bytearray(self.p_path.read_bytes()))
        state.xmem.write_address(VBASE, bytearray(self.v_path.read_bytes()))

        # CR0 (==0) and CR1 (==1) are hardwired; the rest are writable.
        state.regfile.set_cr(2, PBASE)
        state.regfile.set_cr(3, VBASE)
        state.regfile.set_cr(4, OBASE)
        state.regfile.set_cr(5, 128)              # R1 source offset within V channel
        state.regfile.set_cr(6, -1)               # key-index startup
        state.regfile.set_cr(7, P_HEAD_STRIDE)    # P head stride (65536)
        # Loop bounds are count-1: the counter ADD and the BLT share one bundle,
        # so BLT reads the pre-ADD snapshot (branch taken while snapshot < bound).
        state.regfile.set_cr(8, N_TOK - 2)        # 254: key-loop bound (width 256, peeled+startup)
        state.regfile.set_cr(9, D - 1)            # 35: t-loop bound (36 channels)
        state.regfile.set_cr(10, N_HEAD - 1)      # 3: head-loop bound (4 heads)
        state.regfile.set_cr(11, 1)               # 1: g-loop bound (2 groups)
        # LRs
        state.regfile.set_lr(0, 0)                # r_cyclic index / mask_shift
        state.regfile.set_lr(1, N_TOK)            # 256: P key stride / V channel stride
        state.regfile.set_lr(2, 512)              # output-row stride
        state.regfile.set_lr(3, 128)              # group query offset

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # 144 channels, each 1024 B (two 512-B FP32 group rows).
            dump_xmem_to_binary(
                state, self.output_path,
                OBASE, O_CHAN_BYTES, N_CHAN,
            )
