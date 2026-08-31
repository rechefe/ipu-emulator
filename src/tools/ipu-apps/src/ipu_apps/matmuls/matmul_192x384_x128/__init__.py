"""Transformer matmul 192×384×128 harness (Layer 4 FFN2, 64 tokens).

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 192), t in [0, 64).

  D: channel-major [384, 64] input  — K channels × 64 tokens
  W: output-major  [192, 384] weights — N_OUT rows × K cols, stored verbatim (no transpose)
  C: channel-major [192, 64] output  — N_OUT channels × 64 tokens (FP32 accumulators)

Single token group: 64 tokens fit in one 128-lane SIMD vector, so each output
channel needs only one accumulate-and-store pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, no, yes
from ipu_apps.matmuls._spec_support import OP, matmul_query, positive_dims

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

K      = 384   # input channels
N_OUT  = 192   # output channels
N_TOK  = 64    # tokens (single group, padded to LANES in XMEM)

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path here. INT8 is not a mode
# this kernel is written against; it belongs at the XMEM write boundary
# (ACTIVATE.QUANTIZE), which is what makes it invisible to the kernel.
#
# XMEM .asm operands are ROW numbers, not byte addresses (issue #179), and a
# row is LANES *elements*. Region bases are DERIVED from row counts rather
# than hardcoded as bytes: a hardcoded byte map sized for 1-byte elements
# overflows at 4 bytes/element, which silently corrupted wide runs (D ran
# through WEIGHTS_BASE and weight staging overwrote it, so the kernel read
# zeros for high k and dropped most of the contraction).
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

W_STRIDE_ROWS    = -(-K // LANES)            # rows per output channel (ceil) = 3
DATA_STRIDE_ROWS = 1                         # one row per input channel (N_TOK padded to LANES)
W_STRIDE         = W_STRIDE_ROWS * LANES     # elements per output channel (padded)

# One accumulator store writes all 512 B of r_acc. In wide mode a row is also
# 512 B, so a store is exactly one row and one output channel owns one row.
# The narrow harness used to *pack* the output by overlapping successive 512 B
# stores at a 256 B stride; at 4 bytes/element a store fills a whole row
# exactly, so there is nothing to overlap -- each channel gets a full row of
# LANES lanes with the first N_TOK valid and the rest ignored.
OUTPUT_ROW_BYTES   = 512
OUTPUT_STRIDE_ROWS = 1

DATA_ROWS   = K * DATA_STRIDE_ROWS
WEIGHT_ROWS = N_OUT * W_STRIDE_ROWS

# D/W/C packed back to back, in rows.
DATA_BASE_ROW    = 0
WEIGHTS_BASE_ROW = DATA_BASE_ROW + DATA_ROWS
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

# Byte addresses for this harness's direct xmem staging (which bypasses row
# translation); the CR/LR values below stay in rows.
DATA_BASE    = DATA_BASE_ROW * ROW_BYTES
WEIGHTS_BASE = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE  = OUTPUT_BASE_ROW * ROW_BYTES


def _load_data(state: "IpuState", data_path: str | Path) -> None:
    """Stage D, padding each channel's N_TOK elements out to a whole row.

    File layout: K channels × N_TOK elements (D[k][tok] at k*N_TOK + tok).
    XMEM layout:  channel k at row DATA_BASE_ROW + k (N_TOK valid + zero pad).
    """
    raw = Path(data_path).read_bytes()
    expected = K * N_TOK * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{data_path}: expected >= {expected} B, got {len(raw)}")
    for k in range(K):
        row = raw[k * N_TOK * ELEM_BYTES : (k * N_TOK + N_TOK) * ELEM_BYTES]
        padded = bytearray(ROW_BYTES)
        padded[: len(row)] = row
        state.xmem.write_address(DATA_BASE + k * DATA_STRIDE_ROWS * ROW_BYTES, padded)


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Stage W, padding each output channel's K elements out to whole rows.

    File layout: W[j][k] at element j*K + k  (N_OUT rows × K cols).
    K=384 is exactly 3 whole rows, so no padding is needed here -- the generic
    chunk loop covers it anyway.
    """
    raw = Path(weights_path).read_bytes()
    row_elems = LANES                      # elements per XMEM row
    stride = W_STRIDE_ROWS * row_elems     # elements per output channel (padded)
    for j in range(N_OUT):
        row = raw[j * K * ELEM_BYTES : (j * K + K) * ELEM_BYTES]
        for chunk in range(W_STRIDE_ROWS):
            lo = chunk * row_elems
            hi = min(lo + row_elems, K)
            buf = bytearray(row_elems * ELEM_BYTES)
            if hi > lo:
                buf[: (hi - lo) * ELEM_BYTES] = row[lo * ELEM_BYTES : hi * ELEM_BYTES]
            state.xmem.write_address(
                WEIGHTS_BASE + (j * stride + lo) * ELEM_BYTES, buf
            )


class MatMul192x384x128App(IpuApp):
    """192x384_x128 transformer matmul harness (Layer 4 FFN2)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)

    def setup(self, state: "IpuState") -> None:
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)
        # CR1 (≡1) is a read-only hardwired constant; WEIGHTS_BASE lives in CR9.
        state.regfile.set_cr(0, DATA_BASE_ROW)
        state.regfile.set_cr(9, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, WEIGHTS_BASE_ROW + 1)          # +1 weight row
        state.regfile.set_cr(3, WEIGHTS_BASE_ROW + 2)          # +2 weight rows
        state.regfile.set_cr(5, OUTPUT_BASE_ROW)
        state.regfile.set_cr(6, -DATA_STRIDE_ROWS)             # data startup: -1 row
        state.regfile.set_cr(8, -1)                            # per-chunk fixed_idx startup
        state.regfile.set_lr(0, 0)                             # r_cyclic write-index 0
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)              # data stride (rows)
        state.regfile.set_lr(3, OUTPUT_STRIDE_ROWS)            # output stride (rows)
        state.regfile.set_lr(6, 126)                           # width-128 chunk bound
        state.regfile.set_lr(7, 0)                             # output pointer
        state.regfile.set_lr(8, 0)                             # weight byte offset
        state.regfile.set_lr(9, 0)                             # j counter
        state.regfile.set_lr(10, N_OUT)                        # j-loop limit
        state.regfile.set_lr(11, 126)                          # tail-chunk bound: width=128
        state.regfile.set_lr(12, W_STRIDE_ROWS)                # weight stride per j (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUT,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. Every
# matmul kernel is a single fixed (M, K, N) triple with no padding tolerance,
# so `supports` is an exact-shape match. Single token group: M is N_TOK.

M = N_TOK
N = N_OUT


def _supports(**params):
    q = matmul_query(params["shape_a"], params["shape_b_t"])
    bad = positive_dims(q)
    if bad:
        return no(bad)
    if q.shape != (M, K, N):
        return no(
            f"handles exactly (M, K, N) == {(M, K, N)}; got {q.shape}"
        )
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return f"(M, K, N) == {(M, K, N)} exactly: the fixed-shape 192x384 transformer matmul kernel (Layer 4 FFN2)."


SPEC = KernelSpec(
    name="matmul_192x384_x128",
    op=OP,
    variant="192x384_x128",
    app_class=MatMul192x384x128App,
    asm="matmul_192x384_x128.asm",
    requires=("shape_a", "shape_b_t"),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: matmul_query(params["shape_a"], params["shape_b_t"]).bundle,
    # Exact-shape match: no two of the sixteen matmul kernels share a triple.
    cost=lambda **params: 0.0,
)
