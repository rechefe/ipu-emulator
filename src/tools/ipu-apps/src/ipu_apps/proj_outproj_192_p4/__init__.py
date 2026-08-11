"""Multi-stream transformer projection matmul harness (Layer 4 OutProj, P=4).

Computes C[p, j, t] = sum_k W[j, k] * D[p, k, t]
  for all p in [0, 4), j in [0, N_OUT=576), t in [0, N_TOK=64).

  D[p]: channel-major [192, 64] input per stream — K channels x 64 tokens
  W:    output-major  [576, 192] weights, SHARED across all 4 streams —
        N_OUT rows x K cols, stored verbatim (no transpose)
  C[p]: channel-major [576, 64] output per stream — N_OUT channels x 64
        tokens (FP32 accumulators)

One set of learned weights applied independently to 4 pixel-streams in a
single invocation (real transformer-layer property), instead of 4 host
round-trips through the single-stream matmul_576x192_x128 kernel.

The K-dimension contraction runs through a RUNTIME chunk loop (one .asm
control-flow body, not per-shape hand-unrolled labels) -- see the .asm
header for the full design rationale. This harness supplies the two
registers that generalize it: CHUNK_COUNT (= ceil(K/128)) and TAIL_BOUND
(the last chunk's width-2 inner-loop bound; every non-last chunk is a fixed
width-128 -> bound 126, K is only ever partial on the FINAL chunk).

Usage::

    from ipu_apps.proj_outproj_192_p4 import ProjOutProj192P4App

    app = ProjOutProj192P4App(
        inst_path="proj_outproj_192_p4.bin",
        input_paths=[d0, d1, d2, d3],
        weights_path="weights.bin",
        output_paths=[o0, o1, o2, o3],
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

K       = 192   # input channels
N_OUT   = 192   # output channels
N_TOK   = 64    # tokens (single group, padded to LANES in XMEM)
N_STREAM = 4    # pixel-streams processed per invocation

# ---------------------------------------------------------------------------
# Wide-vector FP32 only -- see matmul_576x192_x128/__init__.py for the full
# rationale (elements are 4 B, a row is LANES*4 = 512 B, XMEM .asm operands
# are ROW numbers per issue #179, region bases are derived from row counts).
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

W_STRIDE_ROWS    = -(-K // LANES)            # rows per output channel (ceil) = 2
DATA_STRIDE_ROWS = 1                         # one row per input channel (N_TOK padded to LANES)

CHUNK_COUNT = W_STRIDE_ROWS                  # ceil(K/128) chunks in the runtime chunk loop
_TAIL_WIDTH = K - LANES * (CHUNK_COUNT - 1)  # width of the last (possibly partial) chunk
TAIL_BOUND  = _TAIL_WIDTH - 2                # do-while bound (see .asm header): width - 2
FULL_BOUND  = LANES - 2                      # 126: bound for any full width-128 chunk

OUTPUT_ROW_BYTES   = 512    # one store = one row = one output channel
OUTPUT_STRIDE_ROWS = 1

DATA_ROWS_PER_STREAM   = K * DATA_STRIDE_ROWS
WEIGHT_ROWS            = N_OUT * W_STRIDE_ROWS
OUTPUT_ROWS_PER_STREAM = N_OUT * OUTPUT_STRIDE_ROWS

# D (4 streams) / W (shared) / C (4 streams), packed back to back, in rows.
DATA_BASE_ROW    = 0
WEIGHTS_BASE_ROW = DATA_BASE_ROW + N_STREAM * DATA_ROWS_PER_STREAM
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

DATA_BASE    = DATA_BASE_ROW * ROW_BYTES
WEIGHTS_BASE = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE  = OUTPUT_BASE_ROW * ROW_BYTES


def _load_stream_data(state: "IpuState", data_path: str | Path, stream: int) -> None:
    """Stage one stream's D block, padding each channel to a whole row.

    File layout: K channels x N_TOK elements (D[k][tok] at k*N_TOK + tok).
    XMEM layout: stream p, channel k at row
      DATA_BASE_ROW + p*DATA_ROWS_PER_STREAM + k  (N_TOK valid + zero pad).
    """
    raw = Path(data_path).read_bytes()
    expected = K * N_TOK * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{data_path}: expected >= {expected} B, got {len(raw)}")
    stream_base = DATA_BASE + stream * DATA_ROWS_PER_STREAM * ROW_BYTES
    for k in range(K):
        row = raw[k * N_TOK * ELEM_BYTES : (k * N_TOK + N_TOK) * ELEM_BYTES]
        padded = bytearray(ROW_BYTES)
        padded[: len(row)] = row
        state.xmem.write_address(stream_base + k * DATA_STRIDE_ROWS * ROW_BYTES, padded)


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Stage the shared W, padding each output channel's K elements to whole rows.

    File layout: W[j][k] at element j*K + k  (N_OUT rows x K cols).
    XMEM layout per output channel j (W_STRIDE_ROWS rows):
      row c: W[j, c*128 .. c*128+127], zero-padded on the last (partial) chunk.
    """
    raw = Path(weights_path).read_bytes()
    row_elems = LANES
    stride = W_STRIDE_ROWS * row_elems
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


class ProjOutProj192P4App(IpuApp):
    """576x192x128 multi-stream (P=4) transformer projection harness (Layer 4 QKV)."""

    def __init__(self, **kwargs) -> None:
        self.output_paths: list[Path] | None = None
        super().__init__(**kwargs)
        self.input_paths = [Path(p) for p in self.input_paths]
        if len(self.input_paths) != N_STREAM:
            raise ValueError(f"expected {N_STREAM} input_paths, got {len(self.input_paths)}")
        self.weights_path = Path(self.weights_path)
        if self.output_paths is not None:
            if len(self.output_paths) != N_STREAM:
                raise ValueError(f"expected {N_STREAM} output_paths, got {len(self.output_paths)}")
            self.output_paths = [Path(p) for p in self.output_paths]

    def setup(self, state: "IpuState") -> None:
        for p, path in enumerate(self.input_paths):
            _load_stream_data(state, path, p)
        _load_weights(state, self.weights_path)

        # CR1 (=1) is a read-only hardwired constant on the new architecture.
        state.regfile.set_cr(0, 0)                              # ZERO
        state.regfile.set_cr(2, DATA_BASE_ROW)                  # DATA_BASE (stream 0)
        state.regfile.set_cr(3, WEIGHTS_BASE_ROW)                # WEIGHTS_BASE (shared)
        state.regfile.set_cr(4, OUTPUT_BASE_ROW)                 # OUTPUT_BASE (stream 0)
        state.regfile.set_cr(5, -1)                              # NEG_ONE
        state.regfile.set_cr(6, FULL_BOUND)                      # FULL_BOUND (126)
        state.regfile.set_cr(7, TAIL_BOUND)                      # TAIL_BOUND (shape-specific)
        state.regfile.set_cr(8, W_STRIDE_ROWS)                   # W_STRIDE (rows per output channel)
        state.regfile.set_cr(9, N_OUT)                           # N_OUT_CR (j-loop limit)
        state.regfile.set_cr(10, CHUNK_COUNT)                    # CHUNK_COUNT
        state.regfile.set_cr(11, CHUNK_COUNT - 1)                # LAST_CHUNK_IDX
        state.regfile.set_cr(12, DATA_ROWS_PER_STREAM)           # DATA_STREAM_STR
        state.regfile.set_cr(13, OUTPUT_ROWS_PER_STREAM)         # OUT_STREAM_STR
        state.regfile.set_cr(14, N_STREAM)                       # STREAM_COUNT

    def teardown(self, state: "IpuState") -> None:
        if self.output_paths is None:
            return
        for p, path in enumerate(self.output_paths):
            stream_base = OUTPUT_BASE + p * OUTPUT_ROWS_PER_STREAM * ROW_BYTES
            parts = [
                bytes(state.xmem.read_address(
                    stream_base + j * OUTPUT_STRIDE_ROWS * ROW_BYTES, OUTPUT_ROW_BYTES
                ))
                for j in range(N_OUT)
            ]
            Path(path).write_bytes(b"".join(parts))
