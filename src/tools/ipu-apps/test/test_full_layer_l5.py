"""Full transformer-layer end-to-end test, Layer 5 (d=240, N_TOK=16/stream,
P=4 streams, h=4 heads, head_dim=60, L=3 layers).

Same standing method as test_full_layer_l4.py: poison destination regions
before every stage, assert on raw stored bytes, compare against an
independently-built numpy reference of the whole layer, and never mix the
query-major and key-major attention chains.

L5's attention kernels have a DIFFERENT call granularity than L4's:
  * qk_scores_16x60 / attn_scores_km_16x60 score exactly ONE (stream, head)
    pair per call (no P, no N_HEAD axis in the kernel itself) -- a full L5
    layer needs P*N_HEAD=16 calls, not 1.
  * attn_v_16x60 / attn_v_bcast_60 cover all 4 heads of ONE stream per call
    (no P axis) -- a full L5 layer needs P=4 calls, not 1.
This asymmetry is real (see kernel_docs/kernel_layer_map.md and the L4/L5
constructor-signature research for this task), not a simplification made
here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# See test_full_layer_l4.py's matching comment: makes this file's own
# directory importable whether test/ is a package (bazel) or a flat rootdir
# (plain pytest/uv).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ipu_apps.layernorm_16x240 import LayerNorm16x240App
from ipu_apps.proj_qkv_240_p4 import ProjQKV240P4App, K as QKV_K, N_OUT as QKV_N_OUT
from ipu_apps.qk_scores_16x60 import QkScores16x60App
from ipu_apps.attn_scores_km_16x60 import AttnScoresKM16x60App
from ipu_apps.attn_v_16x60 import AttnV16x60App
from ipu_apps.attn_v_bcast_60 import AttnVBcast60App
import ipu_apps.attn_v_bcast_60 as attn_v_bcast_60_mod
from ipu_apps.proj_outproj_240_p4 import ProjOutProj240P4App, K as OUTPROJ_K, N_OUT as OUTPROJ_N_OUT
from ipu_apps.residual_add_16x240 import ResidualAdd16x240App
from ipu_apps.proj_ffn1_240_p4 import ProjFFN1240P4App, K as FFN1_K, N_OUT as FFN1_N_OUT
from ipu_apps.proj_ffn2_240_p4 import ProjFFN2240P4App, K as FFN2_K, N_OUT as FFN2_N_OUT

from fixture_host_softmax_fold_stubs import softmax_query_major, softmax_key_major
from fixture_full_layer_chain import (
    run_layernorm, run_proj_p4, run_qk_scores_query_major_one_head,
    run_attn_v_query_major_one_stream, run_attn_scores_km_one_head, run_attn_v_bcast,
    run_residual_add, relative_error_stats, format_relative_error_stats,
    real_softmax_query_major, real_softmax_key_major,
)

# See test_full_layer_l4.py's matching flag for what this does.
USE_REAL_SOFTMAX = os.environ.get("USE_REAL_SOFTMAX", "0") == "1"

_LN_INST = Path(os.environ["LAYERNORM_16X240_INST_BIN"])
_QKV_INST = Path(os.environ["PROJ_QKV_240_P4_INST_BIN"])
_QK_INST = Path(os.environ["QK_SCORES_16X60_INST_BIN"])
_KM_INST = Path(os.environ["ATTN_SCORES_KM_16X60_INST_BIN"])
_AV_INST = Path(os.environ["ATTN_V_16X60_INST_BIN"])
_AVBC_INST = Path(os.environ["ATTN_V_BCAST_60_INST_BIN"])
_OUTPROJ_INST = Path(os.environ["PROJ_OUTPROJ_240_P4_INST_BIN"])
_RESID_INST = Path(os.environ["RESIDUAL_ADD_16X240_INST_BIN"])
_FFN1_INST = Path(os.environ["PROJ_FFN1_240_P4_INST_BIN"])
_FFN2_INST = Path(os.environ["PROJ_FFN2_240_P4_INST_BIN"])

D_MODEL = 240
N_TOK = 16
P_STREAM = 4
N_HEAD = 4
HEAD_DIM = 60
N_LAYERS = 3
ROW_BYTES = 512
SCALE = 1.0 / np.sqrt(HEAD_DIM)  # 0.129099, per kernel_docs/kernel_layer_map.md
EXPECTED_SCALE_RATIO = 1.0 / SCALE  # sqrt(60) = 7.745...

assert QKV_K == D_MODEL and QKV_N_OUT == 3 * N_HEAD * HEAD_DIM == 720
assert OUTPROJ_K == D_MODEL and OUTPROJ_N_OUT == D_MODEL
assert FFN1_K == D_MODEL and FFN1_N_OUT == 480
assert FFN2_K == 480 and FFN2_N_OUT == D_MODEL


def _make_layer_weights(rng: np.random.RandomState) -> dict:
    return dict(
        ln1_gamma=rng.uniform(0.5, 1.5, size=D_MODEL).astype(np.float32),
        ln1_beta=rng.uniform(-0.5, 0.5, size=D_MODEL).astype(np.float32),
        w_qkv=rng.uniform(-0.3, 0.3, size=(QKV_N_OUT, D_MODEL)).astype(np.float32),
        w_outproj=rng.uniform(-0.3, 0.3, size=(D_MODEL, D_MODEL)).astype(np.float32),
        ln2_gamma=rng.uniform(0.5, 1.5, size=D_MODEL).astype(np.float32),
        ln2_beta=rng.uniform(-0.5, 0.5, size=D_MODEL).astype(np.float32),
        w_ffn1=rng.uniform(-0.3, 0.3, size=(480, D_MODEL)).astype(np.float32),
        w_ffn2=rng.uniform(-0.3, 0.3, size=(D_MODEL, 480)).astype(np.float32),
    )


def _numpy_layer_reference(x: np.ndarray, w: dict, *, apply_scale: bool) -> np.ndarray:
    """Independent full-layer numpy reference (built from scratch, not from
    any per-kernel golden), for ONE stream, ONE layer. x: [D_MODEL, N_TOK]."""
    def layernorm(v, gamma, beta):
        mu = v.mean(axis=0, keepdims=True)
        var = v.var(axis=0, keepdims=True)
        return gamma[:, None] * (v - mu) / np.sqrt(var + 1e-5) + beta[:, None]

    def silu(v):
        return v / (1.0 + np.exp(-v))

    h = layernorm(x.astype(np.float64), w["ln1_gamma"].astype(np.float64), w["ln1_beta"].astype(np.float64))
    qkv = w["w_qkv"].astype(np.float64) @ h  # [720, N_TOK]
    q_all = qkv[0 * N_HEAD * HEAD_DIM: 1 * N_HEAD * HEAD_DIM]
    k_all = qkv[1 * N_HEAD * HEAD_DIM: 2 * N_HEAD * HEAD_DIM]
    v_all = qkv[2 * N_HEAD * HEAD_DIM: 3 * N_HEAD * HEAD_DIM]

    if apply_scale:
        q_all = q_all * SCALE

    attn_out = np.zeros((N_HEAD * HEAD_DIM, N_TOK), dtype=np.float64)
    for head in range(N_HEAD):
        qh = q_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        kh = k_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        vh = v_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        scores = qh.T @ kh  # [query, key]
        probs = softmax_query_major(scores)
        attn_out[head * HEAD_DIM:(head + 1) * HEAD_DIM] = vh @ probs.T

    proj = w["w_outproj"].astype(np.float64) @ attn_out
    resid1 = proj + x.astype(np.float64)

    h2 = layernorm(resid1, w["ln2_gamma"].astype(np.float64), w["ln2_beta"].astype(np.float64))
    ffn1 = silu(w["w_ffn1"].astype(np.float64) @ h2)
    ffn2 = w["w_ffn2"].astype(np.float64) @ ffn1
    resid2 = ffn2 + resid1
    return resid2


def _numpy_layer_reference_staged(x: np.ndarray, w: dict, *, apply_scale: bool) -> dict:
    """L5 counterpart of test_full_layer_l4.py's matching function -- same
    computation as _numpy_layer_reference, but returns every stage boundary
    for stream 0's Q/K/V, used by the per-stage error-instrumentation test.
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


def _run_layer_query_major(x: np.ndarray, w: dict, tmp_path: Path, tag: str,
                            profile: list, *, scale_q: bool = False,
                            stages: dict | None = None) -> np.ndarray:
    """One full L5 layer, ONE stream, query-major chain (qk_scores_16x60 +
    attn_v_16x60, AGG). qk_scores_16x60 scores exactly one (stream, head)
    pair per call, so this stream's 4 heads need 4 real kernel calls;
    attn_v_16x60 covers all 4 heads of one stream in a single call.

    scale_q: when True, applies the 1/sqrt(head_dim) attention scale to Q
    only, host-side, right before staging into XMEM. See
    test_full_layer_l4.py's matching parameter.

    stages: see test_full_layer_l4.py's matching parameter -- populated in
    place with stream 0's real (float64-cast) intermediate value at every
    stage boundary.
    """
    rng = np.random.RandomState(hash((tag, "filler")) & 0xFFFFFFFF)

    ln1, c, _ = run_layernorm(LayerNorm16x240App, inst_bin=_LN_INST, x=x,
                               gamma=w["ln1_gamma"], beta=w["ln1_beta"], n_ch=D_MODEL,
                               n_tok=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                               tag=f"{tag}_ln1", full_row_output=False)
    profile.append(("layernorm_16x240", c))
    if stages is not None:
        stages["ln1"] = ln1.astype(np.float64)

    d_streams = [ln1 if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                 for p in range(P_STREAM)]
    qkv_outs, c, _ = run_proj_p4(ProjQKV240P4App, inst_bin=_QKV_INST, d_streams=d_streams,
                                  w=w["w_qkv"], k=D_MODEL, n_out=QKV_N_OUT, n_tok=N_TOK,
                                  n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                  tag=f"{tag}_qkv")
    profile.append(("proj_qkv_240_p4", c))
    if stages is not None:
        stages["qkv"] = qkv_outs[0][:, :N_TOK].astype(np.float64)

    out0 = qkv_outs[0][:, :N_TOK]
    # scale_q=False (default) matches the kernels' current unscaled behaviour
    # (apply_scale=False in the reference). The scaled-fixture chain tests
    # pass scale_q=True to probe the missing-scale hypothesis.
    q_all = out0[0 * N_HEAD * HEAD_DIM: 1 * N_HEAD * HEAD_DIM]
    k_all = out0[1 * N_HEAD * HEAD_DIM: 2 * N_HEAD * HEAD_DIM]
    v_all = out0[2 * N_HEAD * HEAD_DIM: 3 * N_HEAD * HEAD_DIM]

    p_bytes_per_head = []
    scores_by_head = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float64)
    probs_by_head = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float64)
    for head in range(N_HEAD):
        q_head = q_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]  # [D, N_TOK]
        if scale_q:
            q_head = q_head * SCALE
        k_head = k_all[head * HEAD_DIM:(head + 1) * HEAD_DIM]

        raw_s, c = run_qk_scores_query_major_one_head(
            QkScores16x60App, inst_bin=_QK_INST, q_head=q_head, k_head=k_head,
            d=HEAD_DIM, n=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
            tag=f"{tag}_qk_h{head}")
        profile.append(("qk_scores_16x60", c))

        # Softmax stub over the RAW scores (query-major: row i = query i),
        # then re-stage the post-softmax probabilities at the same full-row
        # pitch qk_scores_16x60 emitted, so attn_v_16x60 sees exactly its
        # expected layout.
        lanes = ROW_BYTES // 4
        s_rows = np.frombuffer(raw_s, dtype=np.float32).reshape(N_TOK, lanes)
        scores = s_rows[:, :N_TOK]
        scores_by_head[head] = scores.astype(np.float64)
        if USE_REAL_SOFTMAX:
            probs = real_softmax_query_major(scores, tmp_path=tmp_path, tag=f"{tag}_sm_h{head}")
        else:
            probs = softmax_query_major(scores)
        probs_by_head[head] = probs
        out_rows = s_rows.copy()
        out_rows[:, :N_TOK] = probs
        p_bytes_per_head.append(out_rows.astype(np.float32).tobytes())
    if stages is not None:
        stages["scores"] = scores_by_head
        stages["softmax"] = probs_by_head

    v_stream = v_all.reshape(N_HEAD, HEAD_DIM, N_TOK)
    attn_out, c = run_attn_v_query_major_one_stream(
        AttnV16x60App, inst_bin=_AV_INST, p_bytes_per_head=p_bytes_per_head,
        v_stream=v_stream, n_head=N_HEAD, d=HEAD_DIM, n=N_TOK, row_bytes=ROW_BYTES,
        tmp_path=tmp_path, tag=f"{tag}_av")
    profile.append(("attn_v_16x60", c))

    attn_concat = attn_out.reshape(N_HEAD * HEAD_DIM, N_TOK)  # head concat = pure addressing
    if stages is not None:
        stages["attn_v"] = attn_concat.astype(np.float64)

    d_streams_op = [attn_concat if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    op_outs, c, _ = run_proj_p4(ProjOutProj240P4App, inst_bin=_OUTPROJ_INST, d_streams=d_streams_op,
                                 w=w["w_outproj"], k=D_MODEL, n_out=D_MODEL, n_tok=N_TOK,
                                 n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                 tag=f"{tag}_op")
    profile.append(("proj_outproj_240_p4", c))
    proj = op_outs[0][:, :N_TOK]
    if stages is not None:
        stages["out_proj"] = proj.astype(np.float64)

    resid1, c = run_residual_add(ResidualAdd16x240App, inst_bin=_RESID_INST, a=proj, b=x,
                                  n_ch=D_MODEL, n_tok=N_TOK, row_bytes=ROW_BYTES,
                                  tmp_path=tmp_path, tag=f"{tag}_res1", full_row_output=False)
    profile.append(("residual_add_16x240", c))
    if stages is not None:
        stages["resid1"] = resid1.astype(np.float64)

    ln2, c, _ = run_layernorm(LayerNorm16x240App, inst_bin=_LN_INST, x=resid1,
                               gamma=w["ln2_gamma"], beta=w["ln2_beta"], n_ch=D_MODEL,
                               n_tok=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                               tag=f"{tag}_ln2", full_row_output=False)
    profile.append(("layernorm_16x240", c))
    if stages is not None:
        stages["ln2"] = ln2.astype(np.float64)

    d_streams_f1 = [ln2 if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    ffn1_outs, c, _ = run_proj_p4(ProjFFN1240P4App, inst_bin=_FFN1_INST, d_streams=d_streams_f1,
                                   w=w["w_ffn1"], k=D_MODEL, n_out=480, n_tok=N_TOK,
                                   n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                   tag=f"{tag}_ffn1")
    profile.append(("proj_ffn1_240_p4", c))
    ffn1_act = ffn1_outs[0][:, :N_TOK]
    if stages is not None:
        stages["ffn1"] = ffn1_act.astype(np.float64)

    d_streams_f2 = [ffn1_act if p == 0 else rng.uniform(-1, 1, size=(480, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    ffn2_outs, c, _ = run_proj_p4(ProjFFN2240P4App, inst_bin=_FFN2_INST, d_streams=d_streams_f2,
                                   w=w["w_ffn2"], k=480, n_out=D_MODEL, n_tok=N_TOK,
                                   n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                   tag=f"{tag}_ffn2")
    profile.append(("proj_ffn2_240_p4", c))
    ffn2 = ffn2_outs[0][:, :N_TOK]
    if stages is not None:
        stages["ffn2"] = ffn2.astype(np.float64)

    resid2, c = run_residual_add(ResidualAdd16x240App, inst_bin=_RESID_INST, a=ffn2, b=resid1,
                                  n_ch=D_MODEL, n_tok=N_TOK, row_bytes=ROW_BYTES,
                                  tmp_path=tmp_path, tag=f"{tag}_res2", full_row_output=False)
    profile.append(("residual_add_16x240", c))
    if stages is not None:
        stages["resid2"] = resid2.astype(np.float64)

    return resid2


def _run_layer_key_major(x: np.ndarray, w: dict, tmp_path: Path, tag: str,
                          profile: list, *, scale_q: bool = False) -> np.ndarray:
    """One full L5 layer, ONE stream, key-major chain (attn_scores_km_16x60 +
    attn_v_bcast_60, ACC.ADD). attn_scores_km_16x60 scores one (stream, head)
    pair per call (4 calls for this stream's 4 heads); attn_v_bcast_60 covers
    all 4 heads of one stream in a single call.

    scale_q: see _run_layer_query_major's matching parameter.
    """
    rng = np.random.RandomState(hash((tag, "filler_km")) & 0xFFFFFFFF)

    ln1, c, _ = run_layernorm(LayerNorm16x240App, inst_bin=_LN_INST, x=x,
                               gamma=w["ln1_gamma"], beta=w["ln1_beta"], n_ch=D_MODEL,
                               n_tok=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                               tag=f"{tag}_ln1", full_row_output=False)
    profile.append(("layernorm_16x240", c))

    d_streams = [ln1 if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                 for p in range(P_STREAM)]
    qkv_outs, c, _ = run_proj_p4(ProjQKV240P4App, inst_bin=_QKV_INST, d_streams=d_streams,
                                  w=w["w_qkv"], k=D_MODEL, n_out=QKV_N_OUT, n_tok=N_TOK,
                                  n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                  tag=f"{tag}_qkv")
    profile.append(("proj_qkv_240_p4", c))

    out0 = qkv_outs[0][:, :N_TOK]
    # attn_scores_km_16x60's canonical file is [N_HEADS*D, N_TOK] channel-major
    # for ONE stream -- this stream's Q/K slice. scale_q: see the matching
    # note in _run_layer_query_major above.
    q_canonical = out0[0 * N_HEAD * HEAD_DIM: 1 * N_HEAD * HEAD_DIM].astype(np.float32)
    if scale_q:
        q_canonical = q_canonical * SCALE
    k_canonical = out0[1 * N_HEAD * HEAD_DIM: 2 * N_HEAD * HEAD_DIM].astype(np.float32)
    v_all = out0[2 * N_HEAD * HEAD_DIM: 3 * N_HEAD * HEAD_DIM]
    v_stream = v_all.reshape(N_HEAD, HEAD_DIM, N_TOK)

    lanes = ROW_BYTES // 4
    p_bytes_per_head = []
    for head in range(N_HEAD):
        raw, c = run_attn_scores_km_one_head(
            AttnScoresKM16x60App, inst_bin=_KM_INST, q_all_heads=q_canonical,
            k_all_heads=k_canonical, head=head, n_block_or_stream=1, d=HEAD_DIM,
            n=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path, tag=f"{tag}_km")
        profile.append(("attn_scores_km_16x60", c))

        # Row s = key s, columns = queries -- softmax reduces over the KEY
        # axis, which is the ROW axis here (axis=-2 in [key, query] shape).
        key_rows = np.frombuffer(raw, dtype=np.float32).reshape(N_TOK, lanes).copy()
        key_cols = key_rows[:, :N_TOK]  # [key, query]
        if USE_REAL_SOFTMAX:
            probs_km = real_softmax_key_major(key_cols, tmp_path=tmp_path, tag=f"{tag}_sm_h{head}")
        else:
            probs_km = softmax_key_major(key_cols)  # still [key, query], row=key
        out_rows = key_rows.copy()
        out_rows[:, :N_TOK] = probs_km
        p_bytes_per_head.append(out_rows.astype(np.float32).tobytes())

    p_bytes = b"".join(p_bytes_per_head)

    attn_out, c = run_attn_v_bcast(
        AttnVBcast60App, inst_bin=_AVBC_INST, p_bytes=p_bytes, v_blocks=v_stream,
        n_block=N_HEAD, d=HEAD_DIM, n=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
        tag=f"{tag}_avbc", obase_module=attn_v_bcast_60_mod, full_row_output=True)
    profile.append(("attn_v_bcast_60", c))

    attn_concat = attn_out.reshape(N_HEAD * HEAD_DIM, N_TOK)

    d_streams_op = [attn_concat if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    op_outs, c, _ = run_proj_p4(ProjOutProj240P4App, inst_bin=_OUTPROJ_INST, d_streams=d_streams_op,
                                 w=w["w_outproj"], k=D_MODEL, n_out=D_MODEL, n_tok=N_TOK,
                                 n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                 tag=f"{tag}_op")
    profile.append(("proj_outproj_240_p4", c))
    proj = op_outs[0][:, :N_TOK]

    resid1, c = run_residual_add(ResidualAdd16x240App, inst_bin=_RESID_INST, a=proj, b=x,
                                  n_ch=D_MODEL, n_tok=N_TOK, row_bytes=ROW_BYTES,
                                  tmp_path=tmp_path, tag=f"{tag}_res1", full_row_output=False)
    profile.append(("residual_add_16x240", c))

    ln2, c, _ = run_layernorm(LayerNorm16x240App, inst_bin=_LN_INST, x=resid1,
                               gamma=w["ln2_gamma"], beta=w["ln2_beta"], n_ch=D_MODEL,
                               n_tok=N_TOK, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                               tag=f"{tag}_ln2", full_row_output=False)
    profile.append(("layernorm_16x240", c))

    d_streams_f1 = [ln2 if p == 0 else rng.uniform(-1, 1, size=(D_MODEL, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    ffn1_outs, c, _ = run_proj_p4(ProjFFN1240P4App, inst_bin=_FFN1_INST, d_streams=d_streams_f1,
                                   w=w["w_ffn1"], k=D_MODEL, n_out=480, n_tok=N_TOK,
                                   n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                   tag=f"{tag}_ffn1")
    profile.append(("proj_ffn1_240_p4", c))
    ffn1_act = ffn1_outs[0][:, :N_TOK]

    d_streams_f2 = [ffn1_act if p == 0 else rng.uniform(-1, 1, size=(480, N_TOK)).astype(np.float32)
                    for p in range(P_STREAM)]
    ffn2_outs, c, _ = run_proj_p4(ProjFFN2240P4App, inst_bin=_FFN2_INST, d_streams=d_streams_f2,
                                   w=w["w_ffn2"], k=480, n_out=D_MODEL, n_tok=N_TOK,
                                   n_stream=P_STREAM, row_bytes=ROW_BYTES, tmp_path=tmp_path,
                                   tag=f"{tag}_ffn2")
    profile.append(("proj_ffn2_240_p4", c))
    ffn2 = ffn2_outs[0][:, :N_TOK]

    resid2, c = run_residual_add(ResidualAdd16x240App, inst_bin=_RESID_INST, a=ffn2, b=resid1,
                                  n_ch=D_MODEL, n_tok=N_TOK, row_bytes=ROW_BYTES,
                                  tmp_path=tmp_path, tag=f"{tag}_res2", full_row_output=False)
    profile.append(("residual_add_16x240", c))

    return resid2


def _run_full_l5_layer_stack_agrees(tmp_path: Path, chain_name: str, run_layer_fn, *,
                                     scale_q: bool = False) -> None:
    """Run all 3 L5 layers back to back with distinct per-layer weights on
    ONE stream, real kernels throughout, compare per-layer output against an
    independently-built numpy reference, and report a per-layer error trace.

    scale_q: see test_full_layer_l4.py's matching parameter -- runs the
    scaled-Q diagnostic variant with the scale applied on both sides.

    Split into two separately-named test functions below (see
    test_full_layer_l4.py's matching comment) so each attention mapping gets
    its own bazel test timeout budget.
    """
    seed_base = 0x1_5002 if scale_q else 0x1_5000
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
        label = "L5 scaled" if scale_q else "L5"
        print(f"{label} {chain_name} layer {layer_idx}: max abs error = {max_err:.3e}  "
              f"{format_relative_error_stats(stats)}")

        x = got.astype(np.float32)
        x_ref = expected

    label = "L5 scaled" if scale_q else "L5"
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
    print(f"{label} {chain_name} total cycles across the full 3-layer stack (one stream): {total_cycles}")
    for name, c in sorted(cycles_by_kernel.items()):
        print(f"  {name:<28}{c:>10} cycles")


def test_full_l5_layer_stack_agrees_query_major(tmp_path: Path) -> None:
    _run_full_l5_layer_stack_agrees(tmp_path, "query_major", _run_layer_query_major)


def test_full_l5_layer_stack_agrees_key_major(tmp_path: Path) -> None:
    _run_full_l5_layer_stack_agrees(tmp_path, "key_major", _run_layer_key_major)


def test_full_l5_layer_stack_agrees_query_major_scaled(tmp_path: Path) -> None:
    """Scale experiment (see module docstring / kernel_docs): same chain as
    test_full_l5_layer_stack_agrees_query_major, but with 1/sqrt(head_dim)
    applied to Q on both sides.
    """
    _run_full_l5_layer_stack_agrees(tmp_path, "query_major", _run_layer_query_major, scale_q=True)


def test_full_l5_layer_stack_agrees_key_major_scaled(tmp_path: Path) -> None:
    """Scale experiment, key-major counterpart -- see
    test_full_l5_layer_stack_agrees_query_major_scaled."""
    _run_full_l5_layer_stack_agrees(tmp_path, "key_major", _run_layer_key_major, scale_q=True)


def _run_l5_per_stage_error(tmp_path: Path, *, scale_q: bool) -> list[tuple[str, float, dict]]:
    """L5 counterpart of test_full_layer_l4.py's _run_l4_per_stage_error --
    see its docstring for the method. Returns (stage_name, max_abs_error,
    relative_error_stats).
    """
    rng = np.random.RandomState(0x1_5010 if scale_q else 0x1_5011)
    x = rng.uniform(-1.0, 1.0, size=(D_MODEL, N_TOK)).astype(np.float32)
    w = _make_layer_weights(rng)

    stages_got: dict = {}
    profile: list = []
    _run_layer_query_major(x, w, tmp_path, tag="l5_stage_probe", profile=profile,
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


def _print_l5_per_stage_error(errors: list[tuple[str, float, dict]], label: str) -> None:
    print(f"L5 per-stage error ({label}, one layer):")
    prev = None
    for name, err, stats in errors:
        ratio_str = f"  (x{err / prev:.2f})" if prev not in (None, 0.0) else ""
        print(f"  {name:<10} max abs error = {err:.3e}{ratio_str}  "
              f"{format_relative_error_stats(stats)}")
        prev = err


def test_full_l5_per_stage_error_unscaled(tmp_path: Path) -> None:
    """Scale experiment step 2, L5, unscaled configuration. See
    test_full_layer_l4.py's test_full_l4_per_stage_error_unscaled.
    """
    errors = _run_l5_per_stage_error(tmp_path, scale_q=False)
    _print_l5_per_stage_error(errors, "unscaled")


def test_full_l5_per_stage_error_scaled(tmp_path: Path) -> None:
    """Scale experiment step 2, L5, scaled configuration. See
    test_full_layer_l4.py's test_full_l4_per_stage_error_scaled.
    """
    errors = _run_l5_per_stage_error(tmp_path, scale_q=True)
    _print_l5_per_stage_error(errors, "scaled")


def test_full_l5_attention_scale_is_visible() -> None:
    """L5 counterpart of the L4 scale-visibility check: assert the
    unscaled/scaled pre-softmax score ratio equals sqrt(head_dim)=sqrt(60) =
    7.745..., turning the documented-but-unasserted missing scale into a
    tested fact.
    """
    rng = np.random.RandomState(0x5CA15)
    x = rng.uniform(-1.0, 1.0, size=(D_MODEL, N_TOK)).astype(np.float32)
    w = _make_layer_weights(rng)

    unscaled = _numpy_layer_reference(x, w, apply_scale=False)
    scaled = _numpy_layer_reference(x, w, apply_scale=True)
    assert np.max(np.abs(unscaled - scaled)) > 1e-6, (
        "unscaled and scaled L5 references are numerically identical -- the "
        "scale is not reaching the computation"
    )

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

    print(f"L5 scale-visibility: measured score ratio = {ratio.mean():.6f}, "
          f"expected sqrt(head_dim)=sqrt(60) = {EXPECTED_SCALE_RATIO:.6f}")

    np.testing.assert_allclose(
        ratio, EXPECTED_SCALE_RATIO, rtol=1e-6,
        err_msg=(
            f"L5 unscaled/scaled score ratio does not match sqrt(head_dim)="
            f"{EXPECTED_SCALE_RATIO:.6f}"
        ),
    )
