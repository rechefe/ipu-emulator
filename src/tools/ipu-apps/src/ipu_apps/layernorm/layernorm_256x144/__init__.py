"""LayerNorm 256×144 harness.

Computes output[ch, tg, i] = γ[ch] × (x[ch,tg,i] − μ[tg,i]) / σ[tg,i] + β[ch]
for 144 channels × 2 token groups × 128 tokens/group, using wide-vector FP32.

Data layout in XMEM: DATA_BASE + (ch*N_TG + tg)*512  (channel-major, tg interleaved).
Output layout: OUTPUT_BASE + (ch*N_TG + tg)*512.
γ/β span two 512-byte rows (144 > 128 lanes).

Usage::

    from ipu_apps.layernorm.layernorm_256x144 import LayerNorm256x144App
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, no, yes
from ipu_apps.layernorm._spec_support import (
    WIDE_VECTOR_ONLY,
    layernorm_query,
    positive_dims,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_CH        = 144
N_TG        = 2
N_TPG       = 128    # tokens per group (SIMD width)

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path. INT8 is not a mode this
# kernel is written against; it belongs at the XMEM write boundary.
#
# XMEM .asm operands are ROW numbers (one row = LANES elements), not byte
# addresses (issue #179). Region bases are DERIVED from row counts rather than
# hardcoded as bytes: a hardcoded byte map silently goes wrong the moment a
# dimension changes, and regions overwrite each other with no crash.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

ROW_STRIDE_ROWS = 1              # one ROW_BYTES row = exactly 1 XMEM row

# Region sizes in rows. Each is a lane count, so no element-width factor.
DATA_ROWS      = N_CH * N_TG         # (ch, tg) interleaved, one row each
GAMMA_ROWS     = -(-N_CH // LANES)   # ceil: 144 channels span 2 rows
BETA_ROWS      = GAMMA_ROWS
CONST_ROWS     = 1               # ones / neg_inv_n / inv_n / neg_mean each
CENTERED_ROWS  = N_CH            # reused per token group
TEMP_ROWS      = 1
INVSTD_ROWS    = 1
OUTPUT_ROWS    = N_CH * N_TG

# Packed back to back, in rows.
DATA_BASE_ROW      = 0
GAMMA_BASE_ROW     = DATA_BASE_ROW      + DATA_ROWS
BETA_BASE_ROW      = GAMMA_BASE_ROW     + GAMMA_ROWS
ONES_BASE_ROW      = BETA_BASE_ROW      + BETA_ROWS
NEG_INV_N_BASE_ROW = ONES_BASE_ROW      + CONST_ROWS
INV_N_BASE_ROW     = NEG_INV_N_BASE_ROW + CONST_ROWS
NEG_MEAN_BASE_ROW  = INV_N_BASE_ROW     + CONST_ROWS
CENTERED_BASE_ROW  = NEG_MEAN_BASE_ROW  + CONST_ROWS
TEMP_BASE_ROW      = CENTERED_BASE_ROW  + CENTERED_ROWS
INVSTD_BASE_ROW    = TEMP_BASE_ROW      + TEMP_ROWS
OUTPUT_BASE_ROW    = INVSTD_BASE_ROW    + INVSTD_ROWS

# Byte addresses for this harness's direct xmem staging (which bypasses row
# translation); the CR values in setup() stay in rows.
DATA_BASE      = DATA_BASE_ROW      * ROW_BYTES
GAMMA_BASE     = GAMMA_BASE_ROW     * ROW_BYTES
BETA_BASE      = BETA_BASE_ROW      * ROW_BYTES
ONES_BASE      = ONES_BASE_ROW      * ROW_BYTES
NEG_INV_N_BASE = NEG_INV_N_BASE_ROW * ROW_BYTES
INV_N_BASE     = INV_N_BASE_ROW     * ROW_BYTES
NEG_MEAN_BASE  = NEG_MEAN_BASE_ROW  * ROW_BYTES
CENTERED_BASE  = CENTERED_BASE_ROW  * ROW_BYTES
TEMP_BASE      = TEMP_BASE_ROW      * ROW_BYTES
INVSTD_BASE    = INVSTD_BASE_ROW    * ROW_BYTES
OUTPUT_BASE    = OUTPUT_BASE_ROW    * ROW_BYTES


def _fp32_row(values: np.ndarray) -> bytes:
    """Pack a 1-D float32 array into 512 bytes (zero-padded to 128 lanes)."""
    assert values.ndim == 1 and len(values) <= 128
    padded = np.zeros(128, dtype=np.float32)
    padded[: len(values)] = values
    return padded.tobytes()


class LayerNorm256x144App(IpuApp):
    """256-token × 144-channel LayerNorm in wide-vector FP32 debug mode."""

    def __init__(
        self,
        *,
        input_path: str | Path,
        gamma_path: str | Path,
        beta_path: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(input_path)
        self.gamma_path = Path(gamma_path)
        self.beta_path  = Path(beta_path)

    def setup(self, state: "IpuState") -> None:
        # Data: N_CH × N_TG rows of N_TPG FP32 values
        # File layout: (ch*N_TG + tg) row order, each row 512 bytes
        state.xmem.write_address(DATA_BASE, bytearray(self.input_path.read_bytes()))

        # γ and β: 144 values each → two 512-byte rows
        gamma = np.frombuffer(self.gamma_path.read_bytes(), dtype=np.float32)
        beta  = np.frombuffer(self.beta_path.read_bytes(),  dtype=np.float32)
        assert len(gamma) == N_CH and len(beta) == N_CH

        state.xmem.write_address(GAMMA_BASE,              bytearray(_fp32_row(gamma[:128])))
        state.xmem.write_address(GAMMA_BASE + ROW_BYTES,  bytearray(_fp32_row(gamma[128:])))
        state.xmem.write_address(BETA_BASE,               bytearray(_fp32_row(beta[:128])))
        state.xmem.write_address(BETA_BASE + ROW_BYTES,   bytearray(_fp32_row(beta[128:])))

        # Constants
        ones = np.ones(N_TPG, dtype=np.float32).tobytes()
        state.xmem.write_address(ONES_BASE,     bytearray(ones))
        neg_inv_n = np.full(N_TPG, -1.0 / N_CH, dtype=np.float32).tobytes()
        state.xmem.write_address(NEG_INV_N_BASE, bytearray(neg_inv_n))
        inv_n = np.full(N_TPG,  1.0 / N_CH, dtype=np.float32).tobytes()
        state.xmem.write_address(INV_N_BASE,    bytearray(inv_n))

        # CR registers — must match ASM header.
        # NOTE: CR0 (=0) and CR1 (=1) are read-only hardwired constants in the new
        # architecture; writes are silently dropped. DATA_BASE is 0x0 so CR0 is fine,
        # and GAMMA_BASE moved off CR1 to CR11 (CR11's old const-zero role is served
        # by the hardwired CR0).
        state.regfile.set_cr(0,  DATA_BASE_ROW)
        state.regfile.set_cr(2,  BETA_BASE_ROW)
        state.regfile.set_cr(3,  ONES_BASE_ROW)
        state.regfile.set_cr(4,  NEG_INV_N_BASE_ROW)
        state.regfile.set_cr(5,  INV_N_BASE_ROW)
        state.regfile.set_cr(6,  NEG_MEAN_BASE_ROW)
        state.regfile.set_cr(7,  CENTERED_BASE_ROW)
        state.regfile.set_cr(8,  TEMP_BASE_ROW)
        state.regfile.set_cr(9,  INVSTD_BASE_ROW)
        state.regfile.set_cr(10, OUTPUT_BASE_ROW)
        state.regfile.set_cr(11, GAMMA_BASE_ROW)   # moved off read-only CR1
        state.regfile.set_cr(12, N_CH)      # 144
        state.regfile.set_cr(13, ROW_STRIDE_ROWS) # 1 row
        state.regfile.set_cr(14, N_TPG)     # 128

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, ROW_BYTES, N_CH * N_TG,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain.

TOTAL_TOKENS = N_TG * N_TPG  # 256: this kernel's total (not per-group) token count


def _supports(**params):
    q = layernorm_query(params["shape"])
    bad = positive_dims(q)
    if bad:
        return no(bad)
    if (q.channels, q.tokens) != (N_CH, TOTAL_TOKENS):
        return no(
            f"handles exactly (channels={N_CH}, tokens={TOTAL_TOKENS}); "
            f"this query is (channels={q.channels}, tokens={q.tokens})"
        )
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return f"exact match: (channels={N_CH}, tokens={TOTAL_TOKENS})."


SPEC = KernelSpec(
    name="layernorm_256x144",
    op="layernorm",
    variant="256x144",
    app_class=LayerNorm256x144App,
    asm="layernorm_256x144.asm",
    requires=("shape",),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: layernorm_query(params["shape"]).bundle,
    # Exact-shape match: no padding, no chunking. Cheapest possible claim.
    cost=lambda **params: 0.0,
)
