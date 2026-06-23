"""attn@V (AGG kernel) harness — query-major scores, channel-major output.

Computes, per attention head h in [0,4):
    O[i, t] = sum_s P[i, s] * V[s, t]      i, s in [0, 256),  t in [0, 36)

Inputs (channel-major activation tensors, 1 byte/element):
  P query-major  : P[i, s] at PBASE + h*65536 + i*256 + s   (4 heads, head-major)
  V channel-major: V[s, t] at VBASE + (h*36 + t)*256 + s

Output (FP32 R_ACC, 512-byte group rows — same convention as the transformer
matmuls):
  O[i, t] at OBASE + (h*36 + t)*1024 + g*512 + local*4,  i = g*128 + local
  i.e. channel (h*36 + t) occupies 1024 bytes (2 groups of 128 FP32 lanes).

Usage::

    from ipu_apps.attn_v_256x36 import AttnV256x36App
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

# Byte layout in XMEM.
PBASE = 0x00000     # P: 4 * 256 * 256        = 262144 B -> [0x00000, 0x40000)
VBASE = 0x40000     # V: 144 * 256            =  36864 B -> [0x40000, 0x49000)
OBASE = 0x50000     # O: 144 * 1024 (FP32)    = 147456 B -> [0x50000, 0x74000)

P_HEAD_STRIDE = 0x10000   # 256 queries * 256 scores
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


class AttnV256x36App(IpuApp):
    """attn@V AGG kernel harness (4 heads, N=256, head_dim=36)."""

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

        # CR1 (==1) is read-only hardwired; cr0 (==0) is hardwired zero.
        state.regfile.set_cr(2, PBASE)
        state.regfile.set_cr(3, VBASE)
        state.regfile.set_cr(4, OBASE)
        state.regfile.set_cr(5, 256)            # P query / V channel stride (in)
        state.regfile.set_cr(6, 128)            # chunk offset (in)
        state.regfile.set_cr(7, 32768)          # P group-1 offset (128 * 256)
        state.regfile.set_cr(8, P_HEAD_STRIDE)  # P head stride (65536)
        state.regfile.set_cr(9, 127)            # inner-loop bound
        state.regfile.set_cr(10, D)             # t count (36)
        state.regfile.set_cr(11, N_HEAD)        # head count (4)
        state.regfile.set_cr(12, 512)           # O group stride (FP32)
        state.regfile.set_cr(13, O_CHAN_BYTES)  # O channel stride (1024)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # 144 channels, each 1024 B (two 512-B FP32 group rows).
            dump_xmem_to_binary(
                state, self.output_path,
                OBASE, O_CHAN_BYTES, N_CHAN,
            )
