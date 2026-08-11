"""Multi-stream transformer projection matmul harness (Layer 3 FFN1, P=4).

Computes C[p, tg, j, t] = sum_k W[j, k] * D[p, k, tg, t]
  for all p in [0, 4), j in [0, N_OUT    = 288), tg in [0, N_TG=2), t in [0, 128).

  D[p]: interleaved channel-major [144, 2, 128] input per stream -- K
        channels x N_TG token groups x 128 tokens, row (k, tg) at
        p*DATA_STREAM_STRIDE_ROWS + k*N_TG + tg
  W:    output-major  [144, 288] weights, SHARED across all 4 streams --
        N_OUT rows x K cols, stored verbatim (no transpose)
  C[p]: grouped channel-major [2, 144, 128] output per stream -- row (j, tg)
        at p*OUT_STREAM_STRIDE_ROWS + tg*N_OUT + j (FP32 accumulators)

One set of learned weights applied independently to 4 pixel-streams in a
single invocation (real transformer-layer property), instead of 4 host
round-trips through the single-stream matmul_144x288_x128 kernel.

L3's structural difference from the L4/L5 proj_*_p4 family: N=256 tokens per
stream means N_TG=2 (two 128-token groups), so the .asm hand-duplicates a
tg=0/tg=1 block inside the j-loop (mirroring the single-stream ancestor's own
hand-duplication), each running the SAME runtime chunk loop over K that the
L4/L5 kernels use -- see the .asm header for the full design rationale. This
harness supplies CHUNK_COUNT (= ceil(K/128)) and TAIL_BOUND (the last
chunk's width-2 inner-loop bound; every non-last chunk is a fixed width-128
-> bound 126).

Usage::

    from ipu_apps.proj_ffn1_144_p4 import ProjFfn1144P4App

    app = ProjFfn1144P4App(
        inst_path="proj_ffn1_144_p4.bin",
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

K        = 144   # input channels
N_OUT    = 288   # output channels
N_TG     = 2      # token groups (256 tokens = 2 x 128)
N_TOK    = 128    # tokens per group
N_STREAM = 4      # pixel-streams processed per invocation

# ---------------------------------------------------------------------------
# Wide-vector FP32 only -- see matmul_144x288_x128/__init__.py for the full
# rationale (elements are 4 B, a row is LANES*4 = 512 B, XMEM .asm operands
# are ROW numbers per issue #179, region bases are derived from row counts).
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

W_STRIDE_ROWS    = -(-K // LANES)            # rows per output channel (ceil) = 2
DATA_STRIDE_ROWS = N_TG                      # rows per input channel (tg-interleaved)

CHUNK_COUNT = W_STRIDE_ROWS                  # ceil(K/128) chunks in the runtime chunk loop
_TAIL_WIDTH = K - LANES * (CHUNK_COUNT - 1)  # width of the last (possibly partial) chunk
TAIL_BOUND  = _TAIL_WIDTH - 2                # do-while bound (see .asm header): width - 2
FULL_BOUND  = LANES - 2                      # 126: bound for any full width-128 chunk

OUTPUT_ROW_BYTES   = 512    # one store = one row = one output channel
OUTPUT_STRIDE_ROWS = 1

DATA_ROWS_PER_STREAM   = K * N_TG * DATA_STRIDE_ROWS // N_TG  # = K * N_TG (one row per (k, tg))
WEIGHT_ROWS            = N_OUT * W_STRIDE_ROWS
OUTPUT_ROWS_PER_STREAM = N_TG * N_OUT * OUTPUT_STRIDE_ROWS

# D (4 streams) / W (shared) / C (4 streams), packed back to back, in rows.
DATA_BASE_ROW    = 0
WEIGHTS_BASE_ROW = DATA_BASE_ROW + N_STREAM * DATA_ROWS_PER_STREAM
OUTPUT_BASE_ROW  = WEIGHTS_BASE_ROW + WEIGHT_ROWS

DATA_BASE    = DATA_BASE_ROW * ROW_BYTES
WEIGHTS_BASE = WEIGHTS_BASE_ROW * ROW_BYTES
OUTPUT_BASE  = OUTPUT_BASE_ROW * ROW_BYTES


def _load_stream_data(state: "IpuState", data_path: str | Path, stream: int) -> None:
    """Stage one stream's D block, tg-interleaved per channel (matches the
    single-stream ancestor's row (k, tg) layout).

    File layout: N_TG tg blocks x K channels x N_TOK elements each (matches
    matmul_144x288_x128's own _load_data contiguous layout, just staged per
    stream instead of once).
    XMEM layout: stream p, channel k, group tg at row
      DATA_BASE_ROW + p*DATA_ROWS_PER_STREAM + k*N_TG + tg.
    """
    raw = Path(data_path).read_bytes()
    expected = K * N_TG * N_TOK * ELEM_BYTES
    if len(raw) < expected:
        raise ValueError(f"{data_path}: expected >= {expected} B, got {len(raw)}")
    stream_base = DATA_BASE + stream * DATA_ROWS_PER_STREAM * ROW_BYTES
    for tg in range(N_TG):
        tg_off = tg * K * N_TOK * ELEM_BYTES
        for k in range(K):
            row = raw[tg_off + k * N_TOK * ELEM_BYTES : tg_off + (k * N_TOK + N_TOK) * ELEM_BYTES]
            padded = bytearray(ROW_BYTES)
            padded[: len(row)] = row
            row_idx = k * N_TG + tg
            state.xmem.write_address(stream_base + row_idx * ROW_BYTES, padded)


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


class ProjFfn1144P4App(IpuApp):
    """144x288 multi-stream (P=4) transformer projection harness (Layer 3 FFN1)."""

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
        state.regfile.set_cr(3, WEIGHTS_BASE_ROW)               # WEIGHTS_BASE (shared)
        state.regfile.set_cr(4, OUTPUT_BASE_ROW)                # OUTPUT_BASE (stream 0, tg=0 sub-block)
        state.regfile.set_cr(5, -1)                             # NEG_ONE
        state.regfile.set_cr(6, FULL_BOUND)                     # FULL_BOUND (126)
        state.regfile.set_cr(7, TAIL_BOUND)                     # TAIL_BOUND (shape-specific)
        state.regfile.set_cr(8, W_STRIDE_ROWS)                  # W_STRIDE (rows per output channel)
        state.regfile.set_cr(9, N_OUT)                          # N_OUT_CR (j-loop limit, tg1 output offset)
        state.regfile.set_cr(10, CHUNK_COUNT)                   # CHUNK_COUNT
        state.regfile.set_cr(11, CHUNK_COUNT - 1)                # LAST_CHUNK_IDX
        state.regfile.set_cr(12, DATA_ROWS_PER_STREAM)          # DATA_STREAM_STR
        state.regfile.set_cr(13, OUTPUT_ROWS_PER_STREAM)        # OUT_STREAM_STR
        state.regfile.set_cr(14, N_STREAM)                      # STREAM_COUNT

    def teardown(self, state: "IpuState") -> None:
        if self.output_paths is None:
            return
        for p, path in enumerate(self.output_paths):
            stream_base = OUTPUT_BASE + p * OUTPUT_ROWS_PER_STREAM * ROW_BYTES
            parts = []
            for tg in range(N_TG):
                for j in range(N_OUT):
                    row_idx = tg * N_OUT + j
                    parts.append(bytes(state.xmem.read_address(
                        stream_base + row_idx * OUTPUT_STRIDE_ROWS * ROW_BYTES, OUTPUT_ROW_BYTES
                    )))
            Path(path).write_bytes(b"".join(parts))
