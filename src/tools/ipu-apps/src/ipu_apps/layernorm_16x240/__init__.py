"""LayerNorm 16x240 harness (Layer 5).

Computes output[ch, i] = gamma[ch] * (x[ch,i] - mu[i]) / sigma[i] + beta[ch]
for 240 channels x 1 token group (16 tokens/group), using wide-vector FP32.
mu and sigma are reduced over the CHANNEL axis, independently per token.

Layer 5 parameters: d=240, N=16 tokens/stream.

One channel per XMEM row: N_TG=1, so the data layout is simply
DATA_BASE + ch (in rows), with N_TOK=16 valid lanes per row and lanes
16..127 zero padding. Rows are NEVER shared between channels -- a channel
gets a whole 512-byte row even though only 64 bytes of it are meaningful.
The padding lanes ride through the arithmetic and are cropped in
``teardown``.

Output layout: OUTPUT_BASE + ch.
gamma/beta span two rows (240 > 128 lanes).

Padding lanes are arithmetically harmless: x is zero there, so mu and the
variance are zero, and the emulator's ``rsqrt`` activation maps a
non-positive input to 0.0 rather than to an infinity. The constant rows are
therefore filled uniformly across all LANES lanes, exactly as in
``layernorm_64x192``.

Unlike ``layernorm_256x144``, this kernel has NO checked-in test-data
dependency and no ``*_DATA_DIR`` env var -- inputs are generated inline by
``gen_debug_data.py``.

Usage::

    from ipu_apps.layernorm_16x240 import LayerNorm16x240App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_CH  = 240    # channels -> one XMEM row each; also the normalization axis
N_TG  = 1      # single token group
N_TOK = 16     # tokens per group (valid lanes per row)

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
# With N_TG=1 the per-channel data stride IS the row stride: consecutive
# channels are consecutive rows. (In the N_TG=2 ancestor this was 2 rows.)
DATA_STRIDE_ROWS = N_TG * ROW_STRIDE_ROWS    # 1

VALID_ROW_BYTES = N_TOK * ELEM_BYTES         # 64 B of meaningful data per row

# Sub-loop split for step 6: gamma/beta span ceil(240/128) = 2 rows.
SUB_A_CHANNELS = LANES                       # 128 channels from gamma/beta row 0
SUB_B_CHANNELS = N_CH - LANES                # 112 channels from gamma/beta row 1

# Region sizes in rows. Each is a row count, so no element-width factor.
DATA_ROWS      = N_CH * N_TG         # one row per (ch, tg); N_TG=1 -> 240
GAMMA_ROWS     = -(-N_CH // LANES)   # ceil: 240 channels span 2 rows
BETA_ROWS      = GAMMA_ROWS
CONST_ROWS     = 1               # ones / neg_inv_n / inv_n / neg_mean each
CENTERED_ROWS  = N_CH            # centered/normalized scratch, one row per ch
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
    assert values.ndim == 1 and len(values) <= LANES
    padded = np.zeros(LANES, dtype=np.float32)
    padded[: len(values)] = values
    return padded.tobytes()


class LayerNorm16x240App(IpuApp):
    """240-channel x 16-token LayerNorm in wide-vector FP32 debug mode."""

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
        # Data: N_CH rows of LANES FP32 values, one channel per row, tokens in
        # lanes 0..N_TOK-1 and zero padding after. A channel occupies a WHOLE
        # row: rows are never shared, so the row count is N_CH regardless of
        # how few of the 128 lanes carry tokens.
        raw = self.input_path.read_bytes()
        assert len(raw) == N_CH * N_TG * ROW_BYTES, (
            f"input is {len(raw)} bytes, expected {N_CH * N_TG * ROW_BYTES} "
            "(one full 512-byte row per channel)"
        )
        state.xmem.write_address(DATA_BASE, bytearray(raw))

        # gamma and beta: 240 values each -> two 512-byte rows (128 + 112).
        gamma = np.frombuffer(self.gamma_path.read_bytes(), dtype=np.float32)
        beta  = np.frombuffer(self.beta_path.read_bytes(),  dtype=np.float32)
        assert len(gamma) == N_CH and len(beta) == N_CH

        state.xmem.write_address(GAMMA_BASE,             bytearray(_fp32_row(gamma[:LANES])))
        state.xmem.write_address(GAMMA_BASE + ROW_BYTES, bytearray(_fp32_row(gamma[LANES:])))
        state.xmem.write_address(BETA_BASE,              bytearray(_fp32_row(beta[:LANES])))
        state.xmem.write_address(BETA_BASE + ROW_BYTES,  bytearray(_fp32_row(beta[LANES:])))

        # Constants. The reduction is over N_CH channels, so the mean/variance
        # normalizer is 1/N_CH -- NOT 1/N_TOK. Filling all LANES lanes keeps the
        # padding lanes arithmetically consistent (they stay 0).
        ones = np.ones(LANES, dtype=np.float32).tobytes()
        state.xmem.write_address(ONES_BASE,     bytearray(ones))
        neg_inv_n = np.full(LANES, -1.0 / N_CH, dtype=np.float32).tobytes()
        state.xmem.write_address(NEG_INV_N_BASE, bytearray(neg_inv_n))
        inv_n = np.full(LANES,  1.0 / N_CH, dtype=np.float32).tobytes()
        state.xmem.write_address(INV_N_BASE,    bytearray(inv_n))

        # CR registers -- must match the ASM header.
        # NOTE: CR0 (=0) and CR1 (=1) are read-only hardwired constants; writes
        # are silently dropped. DATA_BASE_ROW is 0 so CR0 is fine, and
        # GAMMA_BASE lives on CR11 (CR11's const-zero role is served by CR0).
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
        state.regfile.set_cr(12, N_CH)             # 240 = full channel loop bound
        state.regfile.set_cr(13, ROW_STRIDE_ROWS)  # 1 row (also the data stride, N_TG=1)
        state.regfile.set_cr(14, LANES)            # 128 = sub-loop A bound; r1 base offset
        # CR15 is the reserved dstructure register and must not be overwritten,
        # so step 6's sub-loop B bound (112) is built in the ASM from CR1 and
        # CR14 rather than from a CR of its own.

    def teardown(self, state: "IpuState") -> None:
        """Dump the output, cropping each row to its N_TOK valid tokens."""
        if self.output_path is None:
            return
        raw_path = Path(self.output_path).with_suffix(".rows.bin")
        dump_xmem_to_binary(
            state, raw_path,
            OUTPUT_BASE, ROW_BYTES, N_CH * N_TG,
        )
        rows = np.frombuffer(raw_path.read_bytes(), dtype=np.float32).reshape(
            N_CH * N_TG, LANES
        )
        Path(self.output_path).write_bytes(
            np.ascontiguousarray(rows[:, :N_TOK]).tobytes()
        )
