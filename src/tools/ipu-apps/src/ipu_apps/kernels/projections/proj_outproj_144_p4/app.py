"""Multi-stream transformer projection matmul harness (Layer 3 OutProj, P=4).

Computes C[p, tg, j, t] = sum_k W[j, k] * D[p, k, tg, t]
  for all p in [0, 4), j in [0, N_OUT=144), tg in [0, N_TG=2), t in [0, 128).

  D[p]: interleaved channel-major [144, 2, 128] input per stream -- K
        channels x N_TG token groups x 128 tokens, row (k, tg) at
        p*DATA_STREAM_STRIDE_ROWS + k*N_TG + tg
  W:    output-major  [144, 144] weights, SHARED across all 4 streams --
        N_OUT rows x K cols, stored verbatim (no transpose)
  C[p]: grouped channel-major [2, 144, 128] output per stream -- row (j, tg)
        at p*OUT_STREAM_STRIDE_ROWS + tg*N_OUT + j (FP32 accumulators)

One set of learned weights applied independently to 4 pixel-streams in a
single invocation (real transformer-layer property), instead of 4 host
round-trips through the single-stream matmul_144x144_x128 kernel.

L3's structural difference from the L4/L5 proj_*_p4 family: N=256 tokens per
stream means N_TG=2 (two 128-token groups), so the .asm hand-duplicates a
tg=0/tg=1 block inside the j-loop (mirroring the single-stream ancestor's own
hand-duplication), each running the SAME runtime chunk loop over K that the
L4/L5 kernels use -- see the .asm header for the full design rationale. The
harness supplies CHUNK_COUNT (= ceil(K/128)) and TAIL_BOUND (the last
chunk's width-2 inner-loop bound; every non-last chunk is a fixed width-128
-> bound 126).

The harness, file layouts, XMEM map and CRs are shared by the whole family:
see :mod:`ipu_apps.kernels.projections.app`.

Usage::

    from ipu_apps.kernels.projections.proj_outproj_144_p4.app import ProjOutproj144P4App

    app = ProjOutproj144P4App(
        inst_path="proj_outproj_144_p4.bin",
        input_path="input.bin",      # (4, 2, 144, 128) FP32: all 4 streams' D
        weights_path="weights.bin",  # (144, 144) FP32
        output_path="output.bin",    # (4, 2, 144, 128) FP32 XMEM rows
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from ipu_apps.kernels.projections.app import (  # noqa: F401  (LANES, N_STREAM re-exported)
    LANES, N_STREAM, ProjectionLayout, ProjectionP4App, projection_spec,
)

# -- Dimensions -------------------------------------------------------------

K     = 144   # input channels
N_OUT = 144   # output channels
N_TG  = 2     # token groups (256 tokens = 2 x 128)
N_TOK = 128   # tokens per group

ACTIVATION = "none"   # fused into the store by the .asm


class ProjOutproj144P4App(ProjectionP4App):
    """144->144 multi-stream (P=4) transformer projection harness (Layer 3 OutProj)."""

    layout = ProjectionLayout(k=K, n_out=N_OUT, n_tok=N_TOK, n_tg=N_TG)


SPEC = projection_spec(ProjOutproj144P4App, activation=ACTIVATION)
