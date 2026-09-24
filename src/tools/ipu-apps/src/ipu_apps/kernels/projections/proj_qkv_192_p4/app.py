"""Multi-stream transformer projection matmul harness (Layer 4 QKV, P=4).

Computes C[p, j, t] = sum_k W[j, k] * D[p, k, t]
  for all p in [0, 4), j in [0, N_OUT=576), t in [0, N_TOK=64).

  D[p]: channel-major [192, 64] input per stream -- K channels x 64 tokens
  W:    output-major  [576, 192] weights, SHARED across all 4 streams --
        N_OUT rows x K cols, stored verbatim (no transpose)
  C[p]: channel-major [576, 64] output per stream -- N_OUT channels x 64
        tokens (FP32 accumulators)

One set of learned weights applied independently to 4 pixel-streams in a
single invocation (real transformer-layer property), instead of 4 host
round-trips through the single-stream matmul_576x192_x128 kernel.

The K-dimension contraction runs through a RUNTIME chunk loop (one .asm
control-flow body, not per-shape hand-unrolled labels) -- see the .asm
header for the full design rationale. The harness supplies the two
registers that generalize it: CHUNK_COUNT (= ceil(K/128)) and TAIL_BOUND
(the last chunk's width-2 inner-loop bound; every non-last chunk is a fixed
width-128 -> bound 126, K is only ever partial on the FINAL chunk).

The harness, file layouts, XMEM map and CRs are shared by the whole family:
see :mod:`ipu_apps.kernels.projections.app`.

Usage::

    from ipu_apps.kernels.projections.proj_qkv_192_p4.app import ProjQkv192P4App

    app = ProjQkv192P4App(
        inst_path="proj_qkv_192_p4.bin",
        input_path="input.bin",      # (4, 1, 192, 64) FP32: all 4 streams' D
        weights_path="weights.bin",  # (576, 192) FP32
        output_path="output.bin",    # (4, 1, 576, 128) FP32 XMEM rows
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from ipu_apps.kernels.projections.app import (  # noqa: F401  (LANES, N_STREAM re-exported)
    LANES, N_STREAM, ProjectionLayout, ProjectionP4App, projection_spec,
)

# -- Dimensions -------------------------------------------------------------

K     = 192   # input channels
N_OUT = 576   # output channels
N_TG  = 1     # single token group
N_TOK = 64    # tokens (padded to LANES in XMEM)

ACTIVATION = "none"   # fused into the store by the .asm


class ProjQkv192P4App(ProjectionP4App):
    """192->576 multi-stream (P=4) transformer projection harness (Layer 4 QKV)."""

    layout = ProjectionLayout(k=K, n_out=N_OUT, n_tok=N_TOK, n_tg=N_TG)


SPEC = projection_spec(ProjQkv192P4App, activation=ACTIVATION)
