"""Matrix-multiplication 128×64×64 test harness.

Computes C = A × W^T where:
  A: (M=128, K=64) input   — K elements per row in file; zero-padded to LANES in XMEM
  W: (N=64,  K=64) weights — output-major, W[n][k] (row n = all K inputs for output n)
  C: (M=128, N=64) output  — N accumulators per matrix row

Both K=64 and N=64 are < SIMD width (128):
  - A rows zero-padded out to a whole XMEM row.
  - T[k] (transposed weight row) zero-padded out to a whole row; acc[64..127] = 0.

Output: one accumulator store writes all 512 B of r_acc, which in wide mode is
exactly one XMEM row, so each matrix row of C owns one row and only its first
N*4 bytes are valid. The teardown crops those out, so the output file stays
densely packed at N*4 bytes per row.

Usage::

    from ipu_apps.matmuls.matmul_128x64x64 import MatMul128x64x64App

    app = MatMul128x64x64App(
        inst_path="matmul_128x64x64.bin",
        input_path="input.bin",
        weights_path="weights.bin",
        output_path="output.bin",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, no, yes
from ipu_apps.matmuls._spec_support import OP, matmul_query, positive_dims

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

M = 128   # rows of A / rows of C
K = 64    # cols of A / cols of W per row  (< SIMD width; A rows zero-padded)
N = 64    # rows of W (output neurons) / cols of C  (< SIMD width; T rows zero-padded)

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path here. INT8 is not a mode
# this kernel is written against; it belongs at the XMEM write boundary
# (ACTIVATE.QUANTIZE), which is what makes it invisible to the kernel.
#
# XMEM .asm operands are ROW numbers, not byte addresses (issue #179), and a
# row is LANES *elements*. Region bases are DERIVED from row counts rather
# than hardcoded as bytes: a hardcoded byte map sized for 1-byte elements
# overflows at 4 bytes/element and silently corrupts the run.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                    # FP32
LANES      = 128                  # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES   # 512

INPUT_ROWS  = M   # one padded row per matrix row of A
WEIGHT_ROWS = K   # T[k] is one row per k

# A 512 B r_acc store is exactly one 512 B row in wide mode, so the output
# stride is one row per matrix row of C -- N < LANES only means the row's tail
# is unused, not that the stride is sub-row.
OUTPUT_ROW_BYTES   = N * ELEM_BYTES   # 256 valid bytes at the start of each row
OUTPUT_STRIDE_ROWS = 1
VEC_STRIDE_ROWS    = 1                # one LANES-element vector = 1 row

# A / T / C packed back to back, in rows.
INPUT_BASE_ROW   = 0
WEIGHTS_BASE_ROW = INPUT_BASE_ROW + INPUT_ROWS
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

# Byte addresses for this harness's direct xmem staging (which bypasses row
# translation); the CR/LR values below stay in rows.
INPUT_BASE_ADDR   = INPUT_BASE_ROW * ROW_BYTES
WEIGHTS_BASE_ADDR = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE_ADDR  = OUTPUT_BASE_ROW * ROW_BYTES


def _load_input_padded(state: "IpuState", input_path: str | Path) -> None:
    """Stage A with each row zero-padded from K elements out to a whole row."""
    raw = Path(input_path).read_bytes()
    expected = M * K * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{input_path}: expected >= {expected} B, got {len(raw)}")
    for m in range(M):
        buf = bytearray(ROW_BYTES)
        buf[: K * ELEM_BYTES] = raw[m * K * ELEM_BYTES : (m * K + K) * ELEM_BYTES]
        state.xmem.write_address(INPUT_BASE_ADDR + m * ROW_BYTES, buf)


def _load_and_transpose_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load W[n][k] from file and write T[k] (column k of W) into XMEM.

    File layout: W[n][k] at element n*K + k  (N rows × K cols, output-major).
    XMEM layout: T[k] at row WEIGHTS_BASE_ROW + k.
    T[k][n] = W[n][k]; lanes N..LANES-1 stay zero (N=64 < 128).
    """
    raw = Path(weights_path).read_bytes()
    expected = N * K * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{weights_path}: expected >= {expected} B, got {len(raw)}")
    for k in range(K):
        t_row = bytearray(ROW_BYTES)
        for n in range(N):
            src = (n * K + k) * ELEM_BYTES
            t_row[n * ELEM_BYTES : (n + 1) * ELEM_BYTES] = raw[src : src + ELEM_BYTES]
        state.xmem.write_address(WEIGHTS_BASE_ADDR + k * ROW_BYTES, t_row)


class MatMul128x64x64App(IpuApp):
    """128×64×64 matrix-multiplication application harness.

    Args:
        inst_path:    Path to assembled instruction binary.
        input_path:   Path to input matrix A binary (M×K FP32, row-major).
        weights_path: Path to weight matrix W binary (N×K FP32, output-major W[n][k]).
        output_path:  Optional path to write output C.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)

    def setup(self, state: "IpuState") -> None:
        _load_input_padded(state, self.input_path)
        _load_and_transpose_weights(state, self.weights_path)
        # CR0 (≡0) and CR1 (≡1) are read-only hardwired constants on the new
        # architecture — writes are silently dropped. INPUT_BASE_ROW is 0, so
        # cr0 still reads the correct input base; the weights base is moved to
        # CR11 (a free CR) instead of CR1. See MIGRATION_CHECKLIST.md Bug #2.
        state.regfile.set_cr(0, INPUT_BASE_ROW)
        state.regfile.set_cr(11, WEIGHTS_BASE_ROW)
        state.regfile.set_cr(2, OUTPUT_BASE_ROW)
        state.regfile.set_cr(3, 1)
        state.regfile.set_cr(4, VEC_STRIDE_ROWS)                 # input/weight stride (1 row)
        state.regfile.set_cr(5, OUTPUT_STRIDE_ROWS)              # output stride (rows)
        state.regfile.set_cr(6, M * VEC_STRIDE_ROWS)             # outer-loop limit, compared vs row ptr
        state.regfile.set_cr(7, 0)
        state.regfile.set_cr(8, -VEC_STRIDE_ROWS)                # weight-offset startup (-1 row)
        state.regfile.set_cr(9, -1)
        state.regfile.set_cr(10, K - 1)

    def teardown(self, state: "IpuState") -> None:
        """Crop each output row's valid N accumulators out of its 512 B row."""
        if self.output_path is None:
            return
        parts = [
            bytes(state.xmem.read_address(
                OUTPUT_BASE_ADDR + m * OUTPUT_STRIDE_ROWS * ROW_BYTES,
                OUTPUT_ROW_BYTES,
            ))
            for m in range(M)
        ]
        Path(self.output_path).write_bytes(b"".join(parts))


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. Every
# matmul kernel is a single fixed (M, K, N) triple with no padding tolerance,
# so `supports` is an exact-shape match.


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
    return f"(M, K, N) == {(M, K, N)} exactly: the fixed-shape 128x64x64 matmul kernel."


SPEC = KernelSpec(
    name="matmul_128x64x64",
    op=OP,
    variant="128x64x64",
    app_class=MatMul128x64x64App,
    asm="matmul_128x64x64.asm",
    requires=("shape_a", "shape_b_t"),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: matmul_query(params["shape_a"], params["shape_b_t"]).bundle,
    # Exact-shape match: no two of the sixteen matmul kernels share a triple.
    cost=lambda **params: 0.0,
)
