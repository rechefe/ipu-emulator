"""Transformer matmul 432×144 harness.

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 432), t in [0, 256).

  D: grouped channel-major [2, 144, 128] input  — K channels × 256 tokens (2 token groups × 128)
  W: output-major  [432, 144] weights — N_OUT rows × K cols, stored verbatim (no transpose)
  C: channel-major [432, 256] output  — N_OUT channels × 256 tokens (FP32 accumulators)

Usage::

    from ipu_apps.matmuls.matmul_432x144_x128 import MatMul432x144x128App

    app = MatMul432x144x128App(
        inst_path="matmul_432x144_x128.bin",
        input_path="input.bin",
        weights_path="weights.bin",
        output_path="output.bin",
    )
    state, cycles = app.run()
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

K     = 144   # input channels
N_OUT = 432   # output channels
N_TG  = 2     # token groups
N_TOK = 128   # tokens per group

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

W_STRIDE_ROWS    = -(-K // LANES)            # rows per output channel (ceil)
DATA_STRIDE_ROWS = (N_TG * N_TOK) // LANES   # rows per input channel
W_STRIDE         = W_STRIDE_ROWS * LANES     # elements per output channel (padded)

# One accumulator store writes all 512 B of r_acc. In wide mode a row is also
# 512 B, so a store is exactly one row and one output channel owns one row.
OUTPUT_ROW_BYTES   = 512
OUTPUT_STRIDE_ROWS = 1

DATA_ROWS   = K * N_TG                       # one row per (k, tg)
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
    """Stage D. Channel-major and already contiguous, so a straight copy works.

    File layout: 2 tg blocks × K channels × N_TOK elements each.
    """
    raw = Path(data_path).read_bytes()
    expected = K * N_TG * N_TOK * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{data_path}: expected >= {expected} B, got {len(raw)}")
    state.xmem.write_address(DATA_BASE, bytearray(raw[:expected]))


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Stage W, padding each output channel's K elements out to whole rows.

    File layout: W[j][k] at element j*K + k  (N_OUT rows × K cols).
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


class MatMul432x144x128App(IpuApp):
    """432×144 transformer matmul application harness.

    Args:
        inst_path:    Path to assembled instruction binary.
        input_path:   Path to channel-major FP32 input D.
        weights_path: Path to output-major FP32 weights W (W[j][k]).
        output_path:  Optional path to write output C.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)

    def setup(self, state: "IpuState") -> None:
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)

        # CR1 (≡1) is a read-only hardwired constant on the new architecture —
        # writes are silently dropped. WEIGHTS_BASE is moved to CR9 (free).
        # cr0=DATA_BASE is 0x0 (harmless no-op, matches hardwired 0); cr2 is a
        # writable CR and stays. See MIGRATION_CHECKLIST.md Bug #2.
        state.regfile.set_cr(0, DATA_BASE_ROW)
        state.regfile.set_cr(9, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, WEIGHTS_BASE_ROW + 1)           # W[j,128..143]: next row
        state.regfile.set_cr(3, OUTPUT_BASE_ROW)                                    # tg=0 output
        state.regfile.set_cr(4, OUTPUT_BASE_ROW + N_OUT * OUTPUT_STRIDE_ROWS)       # tg=1 output
        state.regfile.set_cr(5, -DATA_STRIDE_ROWS)              # tg=0 data startup (rows)
        state.regfile.set_cr(6, -(DATA_STRIDE_ROWS // N_TG))    # tg=1 data startup (rows)
        state.regfile.set_cr(7, -1)                             # k-loop1 fixed_idx startup
        state.regfile.set_cr(8, 127)                            # k-loop2 fixed_idx startup
        state.regfile.set_lr(0, 0)                              # r_cyclic write-index 0
        state.regfile.set_lr(2, DATA_STRIDE_ROWS)               # data stride (rows)
        state.regfile.set_lr(3, OUTPUT_STRIDE_ROWS)             # output stride (rows)
        state.regfile.set_lr(6, 126)                            # k-loop1 bound: first_index=0, width=128 → 126
        state.regfile.set_lr(7, 0)                              # output pointer
        state.regfile.set_lr(8, 0)                              # weight byte offset
        state.regfile.set_lr(9, 0)                              # j counter
        state.regfile.set_lr(10, N_OUT)                         # j-loop limit
        state.regfile.set_lr(11, 142)                           # k-loop2 bound: first_index=128, width=16 → 142
        state.regfile.set_lr(12, W_STRIDE_ROWS)                 # weight stride per j (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUT * N_TG,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. Every
# matmul kernel is a single fixed (M, K, N) triple with no padding tolerance,
# so `supports` is an exact-shape match. M is the total token count (N_TG
# groups of N_TOK); N is the output-channel count N_OUT.

M = N_TG * N_TOK
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
    return f"(M, K, N) == {(M, K, N)} exactly: the fixed-shape 432x144 transformer matmul kernel."


SPEC = KernelSpec(
    name="matmul_432x144_x128",
    op=OP,
    variant="432x144_x128",
    app_class=MatMul432x144x128App,
    asm="matmul_432x144_x128.asm",
    requires=("shape_a", "shape_b_t"),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: matmul_query(params["shape_a"], params["shape_b_t"]).bundle,
    # Exact-shape match: no two of the sixteen matmul kernels share a triple.
    cost=lambda **params: 0.0,
)
