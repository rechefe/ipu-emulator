"""Full transformer-layer end-to-end test, Layer 4 (d=192, N_TOK=64/stream,
P=4 streams, h=4 heads, head_dim=48, L=4 layers).

Nothing before this test has run a WHOLE layer, let alone four of them back
to back with distinct weights per layer. Pairwise seam tests (already in this
directory) cannot see: error accumulating across a layer, state carrying over
between repeated layers, or two kernels agreeing by coincidence on a single
seam while the composed chain is wrong. This test runs the real dataflow:

    LayerNorm -> QKV projection -> repack -> scores -> softmax stub ->
    attn@V -> head concat -> out_proj -> residual -> LayerNorm -> FFN1 ->
    silu -> FFN2 -> residual

for BOTH attention mappings (query-major: qk_scores_64x48 + attn_v_64x48,
using AGG; key-major: attn_scores_km_64x48 + attn_v_bcast_48, using
ACC.ADD), run separately and never mixed, across all 4 layers with distinct
per-layer weights.

Standing method for this round: poison destination regions before every
stage (done inside fixture_full_layer_chain.py's run_* helpers), assert on
raw stored bytes, and compare against a numpy reference of the WHOLE layer
built independently -- not assembled from the individual kernels' own
per-kernel golden functions (those encode each kernel's specific rounding
path and would agree with a wrong composition for the same wrong reason).

Softmax and fold are host-side NUMPY TEST FIXTURES (fixture_host_softmax_fold_stubs.py),
not real kernels -- the real ones are owned elsewhere (see
kernel_docs/kernel_layer_map.md; the `pr-softmax-apps` branch has WIP softmax
apps not merged to this branch). Head concat needs no repack -- it is pure
addressing, confirmed in kernel_layer_map.md's QKV->scores repack section.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Make this file's own directory importable regardless of whether the test
# runner treats test/ as a package (bazel auto-generates test/__init__.py,
# so this module loads as `test.test_full_layer_l4`) or as a flat rootdir
# (plain `pytest`/`uv run pytest`, no __init__.py, module loads as
# `test_full_layer_l4`) -- the fixture modules below are siblings in this
# same directory either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ipu_apps.layernorm_64x192 import LayerNorm64x192App
from ipu_apps.proj_qkv_192_p4 import ProjQkv192P4App, K as QKV_K, N_OUT as QKV_N_OUT
from ipu_apps.qk_scores_64x48 import QkScores64x48App
from ipu_apps.attn_scores_km_64x48 import AttnScoresKM64x48App
from ipu_apps.attn_v_64x48 import AttnV64x48App
from ipu_apps.attn_v_bcast_48 import AttnVBcast48App
import ipu_apps.attn_v_bcast_48 as attn_v_bcast_48_mod
from ipu_apps.proj_outproj_192_p4 import ProjOutProj192P4App, K as OUTPROJ_K, N_OUT as OUTPROJ_N_OUT
from ipu_apps.residual_add_64x192 import ResidualAdd64x192App
from ipu_apps.proj_ffn1_192_p4 import ProjFFN1192P4App, K as FFN1_K, N_OUT as FFN1_N_OUT
from ipu_apps.proj_ffn2_192_p4 import ProjFFN2192P4App, K as FFN2_K, N_OUT as FFN2_N_OUT

from fixture_host_softmax_fold_stubs import softmax_query_major, softmax_key_major
from fixture_full_layer_chain import (
    run_layernorm, run_proj_p4, run_qk_scores_query_major_all_blocks,
    run_attn_v_query_major_all_blocks, run_attn_scores_km_one_head, run_attn_v_bcast,
    run_residual_add, relative_error_stats, format_relative_error_stats,
)

_LN_INST = Path(os.environ["LAYERNORM_64X192_INST_BIN"])
_QKV_INST = Path(os.environ["PROJ_QKV_192_P4_INST_BIN"])
_QK_INST = Path(os.environ["QK_SCORES_64X48_INST_BIN"])
_KM_INST = Path(os.environ["ATTN_SCORES_KM_64X48_INST_BIN"])
_AV_INST = Path(os.environ["ATTN_V_64X48_INST_BIN"])
_AVBC_INST = Path(os.environ["ATTN_V_BCAST_48_INST_BIN"])
_OUTPROJ_INST = Path(os.environ["PROJ_OUTPROJ_192_P4_INST_BIN"])
_RESID_INST = Path(os.environ["RESIDUAL_ADD_64X192_INST_BIN"])
_FFN1_INST = Path(os.environ["PROJ_FFN1_192_P4_INST_BIN"])
_FFN2_INST = Path(os.environ["PROJ_FFN2_192_P4_INST_BIN"])

D_MODEL = 192
N_TOK = 64
P_STREAM = 4
N_HEAD = 4
HEAD_DIM = 48
N_LAYERS = 4
ROW_BYTES = 512
SCALE = 1.0 / np.sqrt(HEAD_DIM)  # 0.144338, per kernel_docs/kernel_layer_map.md
EXPECTED_SCALE_RATIO = 1.0 / SCALE  # unscaled / scaled = sqrt(head_dim) = 6.928...

assert QKV_K == D_MODEL and QKV_N_OUT == 3 * N_HEAD * HEAD_DIM == 576
assert OUTPROJ_K == D_MODEL and OUTPROJ_N_OUT == D_MODEL
assert FFN1_K == D_MODEL and FFN1_N_OUT == 384
assert FFN2_K == 384 and FFN2_N_OUT == D_MODEL


def _make_layer_weights(rng: np.random.RandomState) -> dict:
    return dict(
        ln1_gamma=rng.uniform(0.5, 1.5, size=D_MODEL).astype(np.float32),
        ln1_beta=rng.uniform(-0.5, 0.5, size=D_MODEL).astype(np.float32),
        w_qkv=rng.uniform(-0.3, 0.3, size=(QKV_N_OUT, D_MODEL)).astype(np.float32),
        w_outproj=rng.uniform(-0.3, 0.3, size=(D_MODEL, D_MODEL)).astype(np.float32),
        ln2_gamma=rng.uniform(0.5, 1.5, size=D_MODEL).astype(np.float32),
        ln2_beta=rng.uniform(-0.5, 0.5, size=D_MODEL).astype(np.float32),
        w_ffn1=rng.uniform(-0.3, 0.3, size=(384, D_MODEL)).astype(np.float32),
        w_ffn2=rng.uniform(-0.3, 0.3, size=(D_MODEL, 384)).astype(np.float32),
    )


def _numpy_layer_reference(x: np.ndarray, w: dict, *, apply_scale: bool) -> np.ndarray:
    """Independent full-layer numpy reference, built from scratch (never from
    a per-kernel golden), for ONE stream, ONE layer. x: [D_MODEL, N_TOK].

    apply_scale=False matches the kernels' current unscaled behaviour (the
    pass/fail comparison for the raw-bytes test below); apply_scale=True
    matches the real model and is used only by the scale-visibility check.
    """
    def layernorm(v, gamma, beta):
        mu = v.mean(axis=0, keepdims=True)
        var = v.var(axis=0, keepdims=True)
        return gamma[:, None] * (v - mu) / np.sqrt(var + 1e-5) + beta[:, None]

    def silu(v):
        return v / (1.0 + np.exp(-v))

    h = layernorm(x.astype(np.float64), w["ln1_gamma"].astype(np.float64), w["ln1_beta"].astype(np.float64))
    qkv = w["w_qkv"].astype(np.float64) @ h  # [576, N_TOK]
    q_all = qkv[0 * N_HEAD * HEAD_DIM: 1 * N_HEAD * HEAD_DIM]  # [192, N_TOK]
    k_all = qkv[1 * N_HEAD * HEAD_DIM: 2 * N_HEAD * HEAD_DIM]
    v_all = qkv[2 * N_HEAD * HEAD_DIM: 3 * N_HEAD * HEAD_DIM]

    if apply_scale:
        q_all = q_all * SCALE

    attn_out = np.zeros((N_HEAD * HEAD_DIM, N_TOK), dtype=np.float64)
    for head in range(N_HEAD):
        qh = q_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]  # [D, N_TOK]
        kh = k_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        vh = v_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        scores = qh.T @ kh  # [query, key]
        probs = softmax_query_major(scores)  # row = query, reduce over key
        attn_out[head * HEAD_DIM:(head + 1) * HEAD_DIM] = vh @ probs.T  # [D, query]

    proj = w["w_outproj"].astype(np.float64) @ attn_out
    resid1 = proj + x.astype(np.float64)

    h2 = layernorm(resid1, w["ln2_gamma"].astype(np.float64), w["ln2_beta"].astype(np.float64))
    ffn1 = silu(w["w_ffn1"].astype(np.float64) @ h2)
    ffn2 = w["w_ffn2"].astype(np.float64) @ ffn1
    resid2 = ffn2 + resid1
    return resid2


def _numpy_layer_reference_staged(x: np.ndarray, w: dict, *, apply_scale: bool) -> dict:
    """Same computation as _numpy_layer_reference, but returns EVERY stage
    boundary in a dict instead of only the final output -- used by the
    per-stage error-instrumentation test (step 2 of the scale experiment) to
    localize where error multiplies rather than merely carries forward.
    Stream 0's Q/K/V only (matches what the real-kernel per-stage runners
    below report for stream 0), so keys are unprefixed by head/stream.
    """
    def layernorm(v, gamma, beta):
        mu = v.mean(axis=0, keepdims=True)
        var = v.var(axis=0, keepdims=True)
        return gamma[:, None] * (v - mu) / np.sqrt(var + 1e-5) + beta[:, None]

    def silu(v):
        return v / (1.0 + np.exp(-v))

    stages: dict = {}
    h = layernorm(x.astype(np.float64), w["ln1_gamma"].astype(np.float64), w["ln1_beta"].astype(np.float64))
    stages["ln1"] = h
    qkv = w["w_qkv"].astype(np.float64) @ h
    q_all = qkv[0 * N_HEAD * HEAD_DIM: 1 * N_HEAD * HEAD_DIM]
    k_all = qkv[1 * N_HEAD * HEAD_DIM: 2 * N_HEAD * HEAD_DIM]
    v_all = qkv[2 * N_HEAD * HEAD_DIM: 3 * N_HEAD * HEAD_DIM]
    stages["qkv"] = qkv

    if apply_scale:
        q_all = q_all * SCALE

    scores_all = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float64)
    probs_all = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float64)
    attn_out = np.zeros((N_HEAD * HEAD_DIM, N_TOK), dtype=np.float64)
    for head in range(N_HEAD):
        qh = q_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        kh = k_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        vh = v_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        scores = qh.T @ kh
        scores_all[head] = scores
        probs = softmax_query_major(scores)
        probs_all[head] = probs
        attn_out[head * HEAD_DIM:(head + 1) * HEAD_DIM] = vh @ probs.T
    stages["scores"] = scores_all
    stages["softmax"] = probs_all
    stages["attn_v"] = attn_out

    proj = w["w_outproj"].astype(np.float64) @ attn_out
    stages["out_proj"] = proj
    resid1 = proj + x.astype(np.float64)
    stages["resid1"] = resid1

    h2 = layernorm(resid1, w["ln2_gamma"].astype(np.float64), w["ln2_beta"].astype(np.float64))
    stages["ln2"] = h2
    ffn1 = silu(w["w_ffn1"].astype(np.float64) @ h2)
    stages["ffn1"] = ffn1
    ffn2 = w["w_ffn2"].astype(np.float64) @ ffn1
    stages["ffn2"] = ffn2
    resid2 = ffn2 + resid1
    stages["resid2"] = resid2
    return stages


# ---------------------------------------------------------------------------
# Real-kernel per-stage runners for one stream, one layer
# ---------------------------------------------------------------------------

def _run_layer_query_major(x: np.ndarray, w: dict, tmp_path: Path, tag: str,
                            profile: list, *, scale_q: bool = False,
                            stages: dict | None = None) -> np.ndarray:
    """Run one full layer for ONE stream through the REAL kernels, query-major
    attention chain (qk_scores_64x48 + attn_v_64x48, AGG). x: [D_MODEL, N_TOK].
    Since qk_scores_64x48/attn_v_64x48 process all 16 (stream,head) blocks
    per call, this stages the OTHER 3 streams' Q/K/V as independent random
    filler so the real kernel call shape matches production (all 4 streams,
    4 heads) while this function reports only stream 0's result.

    scale_q: when True, applies the 1/sqrt(head_dim) attention scale to Q
    only, host-side, right before staging into XMEM -- mirrors folding the
    scale into W_q. Must be paired with apply_scale=True on the numpy
    reference for a fair comparison (see the *_scaled test entries below).

    stages: when given a dict, populated in place with stream 0's real
    (float64-cast) intermediate value at every stage boundary (ln1, qkv,
    scores, softmax, attn_v, out_proj, resid1, ln2, ffn1, ffn2, resid2) --
    used by the per-stage error-instrumentation test to compare against
    _numpy_layer_reference_staged's matching keys.
    """
    rng = np.random.RandomState(hash((tag, "filler")) & 0xFFFFFFFF)

    ln1, c, _ = run_layernorm(LayerNorm64x192App, inst_bin=_LN_INST, x=x,
                               gamma=w["ln1_gamma"], beta=w["ln1_beta"], n_ch=D_MODEL,
                               n_tok=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                               tag=f"{tag}_ln1", full_row_output=True)
    profile.append(("layernorm_64x192", c))
    if stages is not None:
        stages["ln1"] = ln1.astype(np.float64)

    d_streams = [ln1 if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                 for p in range(P_STREAM)]
    qkv_outs, c, _ = run_proj_p4(ProjQkv192P4App, inst_bin=_QKV_INST, d_streams=d_streams,
                                  w=w["w_qkv"], k=D_MODEL, n_out=QKV_N_OUT, n_tok=N_TOK,
                                  n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                  tag=f"{tag}_qkv")
    profile.append(("proj_qkv_192_p4", c))
    if stages is not None:
        stages["qkv"] = qkv_outs[0][:, :N_TOK].astype(np.float64)

    # Repack: strip padding lanes only (no permutation), per
    # kernel_layer_map.md's QKV->scores repack bridge. Build [N_BLOCK, D, N]
    # channel-major Q/K blocks for qk_scores_64x48's single call covering all
    # 16 (stream, head) blocks; N_BLOCK order is (p*N_HEAD + h).
    q_blocks = np.zeros((P_STREAM * N_HEAD, HEAD_DIM, N_TOK), dtype=np.float32)
    k_blocks = np.zeros((P_STREAM * N_HEAD, HEAD_DIM, N_TOK), dtype=np.float32)
    v_blocks_all = np.zeros((P_STREAM * N_HEAD, HEAD_DIM, N_TOK), dtype=np.float32)
    for p in range(P_STREAM):
        out = qkv_outs[p][:, :N_TOK]  # [576, N_TOK] cropped padding lanes
        q_all = out[0 * N_HEAD * HEAD_DIM: 1 * N_HEAD * HEAD_DIM]
        k_all = out[1 * N_HEAD * HEAD_DIM: 2 * N_HEAD * HEAD_DIM]
        v_all = out[2 * N_HEAD * HEAD_DIM: 3 * N_HEAD * HEAD_DIM]
        for h in range(N_HEAD):
            b = p * N_HEAD + h
            # scale_q=False (default) matches the kernels' CURRENT unscaled
            # behaviour (apply_scale=False in the reference below). The
            # scale-visibility test builds its own scaled reference
            # separately -- see test_full_l4_attention_scale_is_visible. The
            # scaled-fixture chain tests below pass scale_q=True to probe
            # whether the missing scale explains the error growth.
            q_block = q_all[h * HEAD_DIM:(h + 1) * HEAD_DIM]
            if scale_q:
                q_block = q_block * SCALE
            q_blocks[b] = q_block
            k_blocks[b] = k_all[h * HEAD_DIM:(h + 1) * HEAD_DIM]
            v_blocks_all[b] = v_all[h * HEAD_DIM:(h + 1) * HEAD_DIM]

    scores, c = run_qk_scores_query_major_all_blocks(
        QkScores64x48App, inst_bin=_QK_INST, q_blocks=q_blocks, k_blocks=k_blocks,
        n_block=P_STREAM * N_HEAD, d=HEAD_DIM, n=N_TOK, row_bytes=ROW_BYTES,
        tmp_path=tmp_path, tag=f"{tag}_qk")
    profile.append(("qk_scores_64x48", c))
    if stages is not None:
        stages["scores"] = scores[0:N_HEAD].astype(np.float64)

    probs = np.stack([softmax_query_major(scores[b]) for b in range(P_STREAM * N_HEAD)])
    if stages is not None:
        stages["softmax"] = probs[0:N_HEAD].astype(np.float64)

    attn_out, c = run_attn_v_query_major_all_blocks(
        AttnV64x48App, inst_bin=_AV_INST, p_blocks=probs, v_blocks=v_blocks_all,
        n_block=P_STREAM * N_HEAD, d=HEAD_DIM, n=N_TOK, row_bytes=ROW_BYTES,
        tmp_path=tmp_path, tag=f"{tag}_av")
    profile.append(("attn_v_64x48", c))

    # Head concat: pure addressing (no repack) -- stream 0's heads are blocks
    # 0..3, already contiguous in head order.
    attn_concat = attn_out[0:N_HEAD].reshape(N_HEAD * HEAD_DIM, N_TOK)
    if stages is not None:
        stages["attn_v"] = attn_concat.astype(np.float64)

    d_streams_op = [attn_concat if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    op_outs, c, _ = run_proj_p4(ProjOutProj192P4App, inst_bin=_OUTPROJ_INST, d_streams=d_streams_op,
                                 w=w["w_outproj"], k=D_MODEL, n_out=D_MODEL, n_tok=N_TOK,
                                 n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                 tag=f"{tag}_op")
    profile.append(("proj_outproj_192_p4", c))
    proj = op_outs[0][:, :N_TOK]
    if stages is not None:
        stages["out_proj"] = proj.astype(np.float64)

    resid1, c = run_residual_add(ResidualAdd64x192App, inst_bin=_RESID_INST, a=proj, b=x,
                                  n_ch=D_MODEL, n_tok=N_TOK, row_bytes=ROW_BYTES,
                                  tmp_path=tmp_path, tag=f"{tag}_res1", full_row_output=True)
    profile.append(("residual_add_64x192", c))
    if stages is not None:
        stages["resid1"] = resid1.astype(np.float64)

    ln2, c, _ = run_layernorm(LayerNorm64x192App, inst_bin=_LN_INST, x=resid1,
                               gamma=w["ln2_gamma"], beta=w["ln2_beta"], n_ch=D_MODEL,
                               n_tok=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                               tag=f"{tag}_ln2", full_row_output=True)
    profile.append(("layernorm_64x192", c))
    if stages is not None:
        stages["ln2"] = ln2.astype(np.float64)

    d_streams_f1 = [ln2 if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    ffn1_outs, c, _ = run_proj_p4(ProjFFN1192P4App, inst_bin=_FFN1_INST, d_streams=d_streams_f1,
                                   w=w["w_ffn1"], k=D_MODEL, n_out=384, n_tok=N_TOK,
                                   n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                   tag=f"{tag}_ffn1")
    profile.append(("proj_ffn1_192_p4", c))
    ffn1_act = ffn1_outs[0][:, :N_TOK]  # silu already applied at the store boundary
    if stages is not None:
        stages["ffn1"] = ffn1_act.astype(np.float64)

    d_streams_f2 = [ffn1_act if p == 0 else rng.uniform(-1, 1, size=(384, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    ffn2_outs, c, _ = run_proj_p4(ProjFFN2192P4App, inst_bin=_FFN2_INST, d_streams=d_streams_f2,
                                   w=w["w_ffn2"], k=384, n_out=D_MODEL, n_tok=N_TOK,
                                   n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                   tag=f"{tag}_ffn2")
    profile.append(("proj_ffn2_192_p4", c))
    ffn2 = ffn2_outs[0][:, :N_TOK]
    if stages is not None:
        stages["ffn2"] = ffn2.astype(np.float64)

    resid2, c = run_residual_add(ResidualAdd64x192App, inst_bin=_RESID_INST, a=ffn2, b=resid1,
                                  n_ch=D_MODEL, n_tok=N_TOK, row_bytes=ROW_BYTES,
                                  tmp_path=tmp_path, tag=f"{tag}_res2", full_row_output=True)
    profile.append(("residual_add_64x192", c))
    if stages is not None:
        stages["resid2"] = resid2.astype(np.float64)

    return resid2


def _run_layer_key_major(x: np.ndarray, w: dict, tmp_path: Path, tag: str,
                          profile: list, *, scale_q: bool = False) -> np.ndarray:
    """Same layer, KEY-MAJOR attention chain (attn_scores_km_64x48 +
    attn_v_bcast_48, ACC.ADD, no AGG). Bit-different from the query-major
    chain by design -- must produce a separate reference comparison, never
    mixed with the query-major numbers.

    scale_q: see _run_layer_query_major's matching parameter.
    """
    rng = np.random.RandomState(hash((tag, "filler_km")) & 0xFFFFFFFF)

    ln1, c, _ = run_layernorm(LayerNorm64x192App, inst_bin=_LN_INST, x=x,
                               gamma=w["ln1_gamma"], beta=w["ln1_beta"], n_ch=D_MODEL,
                               n_tok=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                               tag=f"{tag}_ln1", full_row_output=True)
    profile.append(("layernorm_64x192", c))

    d_streams = [ln1 if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                 for p in range(P_STREAM)]
    qkv_outs, c, _ = run_proj_p4(ProjQkv192P4App, inst_bin=_QKV_INST, d_streams=d_streams,
                                  w=w["w_qkv"], k=D_MODEL, n_out=QKV_N_OUT, n_tok=N_TOK,
                                  n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                  tag=f"{tag}_qkv")
    profile.append(("proj_qkv_192_p4", c))

    # attn_scores_km_64x48's canonical input file is [P, N_HEAD, D, N]
    # channel-major over ALL 4 streams for a chosen head -- build that
    # directly from all 4 streams' QKV output (real stream 0 + filler 1-3).
    q_canonical = np.zeros((P_STREAM, N_HEAD, HEAD_DIM, N_TOK), dtype=np.float32)
    k_canonical = np.zeros((P_STREAM, N_HEAD, HEAD_DIM, N_TOK), dtype=np.float32)
    v_blocks_all = np.zeros((P_STREAM * N_HEAD, HEAD_DIM, N_TOK), dtype=np.float32)
    for p in range(P_STREAM):
        out = qkv_outs[p][:, :N_TOK]
        q_all = out[0 * N_HEAD * HEAD_DIM: 1 * N_HEAD * HEAD_DIM]
        k_all = out[1 * N_HEAD * HEAD_DIM: 2 * N_HEAD * HEAD_DIM]
        v_all = out[2 * N_HEAD * HEAD_DIM: 3 * N_HEAD * HEAD_DIM]
        for h in range(N_HEAD):
            # scale_q: see the matching note in _run_layer_query_major above.
            q_head = q_all[h * HEAD_DIM:(h + 1) * HEAD_DIM]
            if scale_q:
                q_head = q_head * SCALE
            q_canonical[p, h] = q_head
            k_canonical[p, h] = k_all[h * HEAD_DIM:(h + 1) * HEAD_DIM]
            v_blocks_all[p * N_HEAD + h] = v_all[h * HEAD_DIM:(h + 1) * HEAD_DIM]

    # One attn_scores_km_64x48 call per head (all 4 streams at once); build
    # attn_v_bcast_48's p_path by interleaving the 4 per-head runs at
    # block b = p*N_HEAD + h (see test_seam_km_bcast_48.py).
    head_outputs = {}
    for h in range(N_HEAD):
        raw, c = run_attn_scores_km_one_head(
            AttnScoresKM64x48App, inst_bin=_KM_INST, q_all_heads=q_canonical,
            k_all_heads=k_canonical, head=h, n_block_or_stream=P_STREAM, d=HEAD_DIM,
            n=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path, tag=f"{tag}_km")
        profile.append(("attn_scores_km_64x48", c))
        head_outputs[h] = raw

    n_block = P_STREAM * N_HEAD
    p_bytes = bytearray(n_block * N_TOK * (ROW_BYTES // 4) * 4)
    for h, raw in head_outputs.items():
        rows = np.frombuffer(raw, dtype=np.float32).reshape(P_STREAM, N_TOK, ROW_BYTES // 4)
        for p in range(P_STREAM):
            b = p * N_HEAD + h
            key_rows = rows[p].copy()  # [N_TOK keys, LANES] key-major, row s = key s
            # Softmax stub, KEY-MAJOR: reduce over the key axis, which here is
            # the ROW axis (axis=-2) since row s = key s, column q = query q.
            key_cols = key_rows[:, :N_TOK].T  # [query, key] -> transpose view for stub call
            probs_km = softmax_key_major(key_cols.T)  # [key, query], row=key
            out_rows = key_rows.copy()
            out_rows[:, :N_TOK] = probs_km
            lo = b * N_TOK * (ROW_BYTES // 4) * 4
            hi = lo + N_TOK * (ROW_BYTES // 4) * 4
            p_bytes[lo:hi] = out_rows.astype(np.float32).tobytes()

    attn_out, c = run_attn_v_bcast(
        AttnVBcast48App, inst_bin=_AVBC_INST, p_bytes=bytes(p_bytes), v_blocks=v_blocks_all,
        n_block=n_block, d=HEAD_DIM, n=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
        tag=f"{tag}_avbc", obase_module=attn_v_bcast_48_mod, full_row_output=False)
    profile.append(("attn_v_bcast_48", c))

    attn_concat = attn_out[0:N_HEAD].reshape(N_HEAD * HEAD_DIM, N_TOK)

    d_streams_op = [attn_concat if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    op_outs, c, _ = run_proj_p4(ProjOutProj192P4App, inst_bin=_OUTPROJ_INST, d_streams=d_streams_op,
                                 w=w["w_outproj"], k=D_MODEL, n_out=D_MODEL, n_tok=N_TOK,
                                 n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                 tag=f"{tag}_op")
    profile.append(("proj_outproj_192_p4", c))
    proj = op_outs[0][:, :N_TOK]

    resid1, c = run_residual_add(ResidualAdd64x192App, inst_bin=_RESID_INST, a=proj, b=x,
                                  n_ch=D_MODEL, n_tok=N_TOK, row_bytes=ROW_BYTES,
                                  tmp_path=tmp_path, tag=f"{tag}_res1", full_row_output=True)
    profile.append(("residual_add_64x192", c))

    ln2, c, _ = run_layernorm(LayerNorm64x192App, inst_bin=_LN_INST, x=resid1,
                               gamma=w["ln2_gamma"], beta=w["ln2_beta"], n_ch=D_MODEL,
                               n_tok=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                               tag=f"{tag}_ln2", full_row_output=True)
    profile.append(("layernorm_64x192", c))

    d_streams_f1 = [ln2 if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    ffn1_outs, c, _ = run_proj_p4(ProjFFN1192P4App, inst_bin=_FFN1_INST, d_streams=d_streams_f1,
                                   w=w["w_ffn1"], k=D_MODEL, n_out=384, n_tok=N_TOK,
                                   n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                   tag=f"{tag}_ffn1")
    profile.append(("proj_ffn1_192_p4", c))
    ffn1_act = ffn1_outs[0][:, :N_TOK]

    d_streams_f2 = [ffn1_act if p == 0 else rng.uniform(-1, 1, size=(384, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    ffn2_outs, c, _ = run_proj_p4(ProjFFN2192P4App, inst_bin=_FFN2_INST, d_streams=d_streams_f2,
                                   w=w["w_ffn2"], k=384, n_out=D_MODEL, n_tok=N_TOK,
                                   n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                   tag=f"{tag}_ffn2")
    profile.append(("proj_ffn2_192_p4", c))
    ffn2 = ffn2_outs[0][:, :N_TOK]

    resid2, c = run_residual_add(ResidualAdd64x192App, inst_bin=_RESID_INST, a=ffn2, b=resid1,
                                  n_ch=D_MODEL, n_tok=N_TOK, row_bytes=ROW_BYTES,
                                  tmp_path=tmp_path, tag=f"{tag}_res2", full_row_output=True)
    profile.append(("residual_add_64x192", c))

    return resid2


def _run_full_l4_layer_stack_agrees(tmp_path: Path, chain_name: str, run_layer_fn, *,
                                     scale_q: bool = False) -> None:
    """Run all 4 L4 layers back to back with distinct per-layer weights on
    ONE stream, real kernels throughout, and compare per-layer output against
    an independently-built numpy reference of the WHOLE layer (unscaled Q,
    matching the kernels' current behaviour -- see the scale-visibility test
    below for the folded-scale comparison). Reports a per-layer error trace,
    not just the final figure, so a chain that degrades steadily and one that
    fails at a specific layer are distinguishable.

    scale_q: when True, runs the SCALED-Q diagnostic variant instead (see
    module docstring's scale experiment) -- both the real-kernel chain and
    the numpy reference apply 1/sqrt(head_dim) to Q, so the comparison stays
    apples-to-apples while probing whether the missing scale explains the
    observed ~4x-per-layer error growth in the unscaled chain.

    Split into two separately-named test functions below (not
    @pytest.mark.parametrize) so each attention mapping gets its own bazel
    test timeout budget -- both mappings together exceeded even a 3600s
    "enormous" combined budget in the sandboxed bazel environment.
    """
    seed_base = 0x1_4002 if scale_q else 0x1_4000
    master_rng = np.random.RandomState(seed_base + {"query_major": 0, "key_major": 1}[chain_name])
    x = master_rng.uniform(-1.0, 1.0, size=(D_MODEL, N_TOK)).astype(np.float32)
    x_ref = x.copy()

    profile: list[tuple[str, int]] = []
    per_layer_errors = []
    per_layer_stats = []

    for layer_idx in range(N_LAYERS):
        w = _make_layer_weights(master_rng)

        got = run_layer_fn(x, w, tmp_path, tag=f"{chain_name}_l{layer_idx}", profile=profile,
                            scale_q=scale_q)
        expected = _numpy_layer_reference(x_ref, w, apply_scale=scale_q)

        max_err = float(np.max(np.abs(got.astype(np.float64) - expected)))
        stats = relative_error_stats(got, expected)
        per_layer_errors.append(max_err)
        per_layer_stats.append(stats)
        label = "L4 scaled" if scale_q else "L4"
        print(f"{label} {chain_name} layer {layer_idx}: max abs error = {max_err:.3e}  "
              f"{format_relative_error_stats(stats)}")

        x = got.astype(np.float32)
        x_ref = expected

    label = "L4 scaled" if scale_q else "L4"
    print(f"{label} {chain_name} per-layer error trace: "
          f"{['%.3e' % e for e in per_layer_errors]}")
    print(f"{label} {chain_name} per-layer max_rel trace: "
          f"{['%.3e' % s['max_rel'] for s in per_layer_stats]}")

    for layer_idx, err in enumerate(per_layer_errors):
        assert err < 5e-1, (
            f"{label} {chain_name} layer {layer_idx}: max abs error {err:.3e} exceeds "
            f"tolerance -- see per-layer trace above for where error entered"
        )

    cycles_by_kernel: dict[str, int] = {}
    for name, c in profile:
        cycles_by_kernel[name] = cycles_by_kernel.get(name, 0) + c
    total_cycles = sum(cycles_by_kernel.values())
    print(f"{label} {chain_name} total cycles across the full 4-layer stack (one stream): {total_cycles}")
    for name, c in sorted(cycles_by_kernel.items()):
        print(f"  {name:<28}{c:>10} cycles")


def test_full_l4_layer_stack_agrees_query_major(tmp_path: Path) -> None:
    _run_full_l4_layer_stack_agrees(tmp_path, "query_major", _run_layer_query_major)


def test_full_l4_layer_stack_agrees_key_major(tmp_path: Path) -> None:
    _run_full_l4_layer_stack_agrees(tmp_path, "key_major", _run_layer_key_major)


def test_full_l4_layer_stack_agrees_query_major_scaled(tmp_path: Path) -> None:
    """Scale experiment (see module docstring / kernel_docs): same chain as
    test_full_l4_layer_stack_agrees_query_major, but with 1/sqrt(head_dim)
    applied to Q on both sides. If the missing scale is the amplifier behind
    the ~4x-per-layer error growth, this trace should flatten toward linear.
    """
    _run_full_l4_layer_stack_agrees(tmp_path, "query_major", _run_layer_query_major, scale_q=True)


def test_full_l4_layer_stack_agrees_key_major_scaled(tmp_path: Path) -> None:
    """Scale experiment, key-major counterpart -- see
    test_full_l4_layer_stack_agrees_query_major_scaled."""
    _run_full_l4_layer_stack_agrees(tmp_path, "key_major", _run_layer_key_major, scale_q=True)


def _run_l4_per_stage_error(tmp_path: Path, *, scale_q: bool) -> list[tuple[str, float, dict]]:
    """Scale experiment step 2: run ONE L4 layer (query-major chain) through
    the real kernels while capturing every stage boundary, and diff each
    stage's real value against an independently-built numpy reference of
    that SAME stage (not a reference forward-computed from the previous
    stage's real, already-diverged value). This localizes where error enters
    and multiplies rather than only reporting the cumulative figure at the
    end of the layer. Returns (stage_name, max_abs_error, relative_error_stats).
    """
    rng = np.random.RandomState(0x1_4010 if scale_q else 0x1_4011)
    x = rng.uniform(-1.0, 1.0, size=(D_MODEL, N_TOK)).astype(np.float32)
    w = _make_layer_weights(rng)

    stages_got: dict = {}
    profile: list = []
    _run_layer_query_major(x, w, tmp_path, tag="l4_stage_probe", profile=profile,
                            scale_q=scale_q, stages=stages_got)
    stages_expected = _numpy_layer_reference_staged(x, w, apply_scale=scale_q)

    order = ["ln1", "qkv", "scores", "softmax", "attn_v", "out_proj", "resid1",
             "ln2", "ffn1", "ffn2", "resid2"]
    errors = []
    for name in order:
        got = stages_got[name]
        expected = stages_expected[name]
        err = float(np.max(np.abs(got - expected)))
        stats = relative_error_stats(got, expected)
        errors.append((name, err, stats))
    return errors


def _print_l4_per_stage_error(errors: list[tuple[str, float, dict]], label: str) -> None:
    print(f"L4 per-stage error ({label}, one layer):")
    prev = None
    for name, err, stats in errors:
        ratio_str = f"  (x{err / prev:.2f})" if prev not in (None, 0.0) else ""
        print(f"  {name:<10} max abs error = {err:.3e}{ratio_str}  "
              f"{format_relative_error_stats(stats)}")
        prev = err


def test_full_l4_per_stage_error_unscaled(tmp_path: Path) -> None:
    """Scale experiment step 2, unscaled configuration (matches the current
    kernel behaviour / the pass-fail chain). Prints the per-stage error so a
    jump concentrated in scores->softmax->attn_v (softmax hypothesis) can be
    told apart from a jump elsewhere (a different bug).
    """
    errors = _run_l4_per_stage_error(tmp_path, scale_q=False)
    _print_l4_per_stage_error(errors, "unscaled")


def test_full_l4_per_stage_error_scaled(tmp_path: Path) -> None:
    """Scale experiment step 2, scaled configuration (1/sqrt(head_dim)
    applied to Q on both sides). Compare this trace against
    test_full_l4_per_stage_error_unscaled's to see whether scaling flattens
    the scores->softmax->attn_v jump specifically.
    """
    errors = _run_l4_per_stage_error(tmp_path, scale_q=True)
    _print_l4_per_stage_error(errors, "scaled")


def test_full_l4_attention_scale_is_visible() -> None:
    """The attention-scale check specified for the seam audit and never
    actually performed: build TWO independent full-layer references, one
    WITHOUT 1/sqrt(head_dim) (matching what the kernels currently do -- the
    pass/fail comparison above) and one WITH it (matching the real model),
    and assert they differ by exactly sqrt(head_dim) = 6.928..., not some
    other ratio. A reference built to mirror the kernels' unscaled behaviour
    can never expose this gap by construction; this test builds the SCALED
    reference specifically to make the gap visible and documented.
    """
    rng = np.random.RandomState(0x5CA1E)
    x = rng.uniform(-1.0, 1.0, size=(D_MODEL, N_TOK)).astype(np.float32)
    w = _make_layer_weights(rng)

    unscaled = _numpy_layer_reference(x, w, apply_scale=False)
    scaled = _numpy_layer_reference(x, w, apply_scale=True)

    diff = unscaled - scaled
    assert np.max(np.abs(diff)) > 1e-6, (
        "unscaled and scaled L4 references are numerically identical -- the "
        "scale is not reaching the computation, so this check has no teeth"
    )

    # The scale enters Q, then Q@K, then softmax (nonlinear) -- so the two
    # OUTPUTS do not differ by a clean constant ratio (softmax saturates
    # differently at the two scales). What IS exactly checkable is the PRE-
    # SOFTMAX score ratio, which is linear in the scale: score_unscaled /
    # score_scaled == 1/SCALE == sqrt(head_dim) exactly, for any nonzero
    # score. Recompute just the scores at both settings to pin that down.
    def scores_only(apply_scale: bool) -> np.ndarray:
        mu = x.astype(np.float64).mean(axis=0, keepdims=True)
        var = x.astype(np.float64).var(axis=0, keepdims=True)
        h = w["ln1_gamma"][:, None] * (x.astype(np.float64) - mu) / np.sqrt(var + 1e-5) + w["ln1_beta"][:, None]
        qkv = w["w_qkv"].astype(np.float64) @ h
        q_all = qkv[0:N_HEAD * HEAD_DIM]
        k_all = qkv[N_HEAD * HEAD_DIM:2 * N_HEAD * HEAD_DIM]
        if apply_scale:
            q_all = q_all * SCALE
        qh = q_all[0:HEAD_DIM]
        kh = k_all[0:HEAD_DIM]
        return qh.T @ kh

    s_unscaled = scores_only(apply_scale=False)
    s_scaled = scores_only(apply_scale=True)

    nonzero = np.abs(s_scaled) > 1e-9
    ratio = s_unscaled[nonzero] / s_scaled[nonzero]

    print(f"L4 scale-visibility: measured score ratio = {ratio.mean():.6f}, "
          f"expected sqrt(head_dim)=sqrt(48) = {EXPECTED_SCALE_RATIO:.6f}")

    np.testing.assert_allclose(
        ratio, EXPECTED_SCALE_RATIO, rtol=1e-6,
        err_msg=(
            f"L4 unscaled/scaled score ratio does not match sqrt(head_dim)="
            f"{EXPECTED_SCALE_RATIO:.6f} -- the missing attention scale is not "
            "the documented factor"
        ),
    )
