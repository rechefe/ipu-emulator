"""Conformance suite for the 6 families ported from ZDlinear (registry-driven).

Mirrors ``test_kernel_registry.py``'s softmax conformance test
(``test_resolved_kernel_computes_the_operation``): for each op, ``resolve()``
is asked for the exact shape one of the family's own apps declares, and the
kernel the registry hands back must actually compute the operation -- not
just claim to. Unlike softmax's row-count sweep, every kernel in these six
families is an exact-shape-only match (no padding, no chunking across apps),
so there is exactly one query worth asking per app: its own declared shape.

Each op's staging convention (channel-major vs query-major vs key-major,
tg-interleaving, one-channel-per-row padding) is copied verbatim from that
family's own existing per-app test -- see e.g. test_attn_v_16x60_wide.py and
test_attn_v_bcast_36_wide.py, whose P layouts are deliberately NOT
interchangeable (query-major vs key-major) despite sharing ctor kwarg names.
Getting a layout wrong here would silently mis-stage inputs and produce a
false failure, so nothing is re-derived -- only reused.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.kernel_registry import resolve

_APP_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps"


def _assemble(spec) -> Path:
    asm = next(_APP_SRC.rglob(spec.asm))
    tmp = tempfile.mkdtemp()
    inst = Path(tmp) / "prog.bin"
    # Explicit UTF-8: some .asm headers contain non-ASCII characters (e.g. the
    # multiplication sign in "128x16"), and this must not depend on the
    # runtime's default codepage (cp1255 on some Windows locales cannot
    # decode them, while Bazel's hermetic Python always defaults to UTF-8).
    assemble_to_bin_file(asm.read_text(encoding="utf-8"), str(inst))
    return inst


def _wide_state() -> IpuState:
    return IpuState(
        wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32
    )


# -- matmul -------------------------------------------------------------


def test_resolved_matmul_computes_the_operation(tmp_path):
    from ipu_apps.matmuls.matmul_128x128 import M, K, N

    verdict = resolve("matmul", shape_a=(M, K), shape_b_t=(N, K))
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "matmul_128x128"

    rng = np.random.RandomState(0x128128)
    A = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)

    input_path = tmp_path / "a.bin"
    weights_path = tmp_path / "w.bin"
    output_path = tmp_path / "out.bin"
    input_path.write_bytes(A.tobytes())
    weights_path.write_bytes(W.tobytes())

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        input_path=input_path,
        weights_path=weights_path,
        output_path=output_path,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=_wide_state())
    assert cycles > 0

    expected = A @ W.T
    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


# -- projection -----------------------------------------------------------


def test_resolved_projection_computes_the_operation(tmp_path):
    from ipu_apps.projections.proj_qkv_144_p4 import K, N_OUT, N_TG, N_TOK, N_STREAM

    verdict = resolve("projection", k=K, n_out=N_OUT)
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "proj_qkv_144_p4"

    rng = np.random.RandomState(0xC0FFEE)
    D = [
        rng.uniform(-1.0, 1.0, size=(N_TG, K, N_TOK)).astype(np.float32)
        for _ in range(N_STREAM)
    ]
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)

    input_paths = []
    for p in range(N_STREAM):
        path = tmp_path / f"input_p{p}.bin"
        path.write_bytes(D[p].tobytes())
        input_paths.append(path)
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(W.tobytes())
    output_paths = [tmp_path / f"output_p{p}.bin" for p in range(N_STREAM)]

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        input_paths=input_paths,
        weights_path=weights_path,
        output_paths=output_paths,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=_wide_state())
    assert cycles > 0

    for p in range(N_STREAM):
        raw = np.frombuffer(output_paths[p].read_bytes(), dtype=np.float32)
        got = raw.reshape(N_TG, N_OUT, 128)
        for tg in range(N_TG):
            expected = W @ D[p][tg]
            np.testing.assert_allclose(
                got[tg][:, :N_TOK], expected, rtol=1e-4, atol=1e-3
            )


# -- layernorm --------------------------------------------------------------


def test_resolved_layernorm_computes_the_operation(tmp_path):
    from ipu_apps.layernorm.layernorm_128x16 import N_CH, N_TPG
    from ipu_apps.layernorm.layernorm_128x16.gen_test_data import (
        reference_layernorm,
        _pack_fp32_row,
    )

    verdict = resolve("layernorm", shape=(N_CH, N_TPG))
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "layernorm_128x16"

    rng = np.random.RandomState(0x1A7E)
    x = rng.uniform(-1.0, 1.0, size=(N_CH, N_TPG)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=(N_CH,)).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=(N_CH,)).astype(np.float32)

    input_path = tmp_path / "x.bin"
    gamma_path = tmp_path / "gamma.bin"
    beta_path = tmp_path / "beta.bin"
    input_path.write_bytes(b"".join(_pack_fp32_row(x[ch]) for ch in range(N_CH)))
    gamma_path.write_bytes(_pack_fp32_row(gamma))
    beta_path.write_bytes(_pack_fp32_row(beta))
    output_path = tmp_path / "out.bin"

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
        output_path=output_path,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=500_000, state=_wide_state())
    assert cycles > 0

    expected, _, _, _ = reference_layernorm(x, gamma, beta)
    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(
        N_CH, 128
    )[:, :N_TPG]
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


# -- residual_add -------------------------------------------------------


def test_resolved_residual_add_computes_the_operation(tmp_path):
    from ipu_apps.residual_add.residual_add_16x240 import N_CH, N_TOK
    from ipu_apps.residual_add.residual_add_16x240.gen_debug_data import pack_rows

    verdict = resolve("residual_add", shape=(N_TOK, N_CH))
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "residual_add_16x240"

    rng = np.random.RandomState(0x5ADD)
    a = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)

    a_path = tmp_path / "a.bin"
    b_path = tmp_path / "b.bin"
    a_path.write_bytes(pack_rows(a).tobytes())
    b_path.write_bytes(pack_rows(b).tobytes())
    output_path = tmp_path / "out.bin"

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        input_a_path=a_path,
        input_b_path=b_path,
        output_path=output_path,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=_wide_state())
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(
        N_CH, N_TOK
    )
    np.testing.assert_allclose(got, a + b, rtol=1e-4, atol=1e-3)


# -- unfold -----------------------------------------------------------------


def test_resolved_unfold_computes_the_operation(tmp_path):
    from ipu_apps.unfold.unfold_16x16x192 import H, W, C, N_STRIPES

    verdict = resolve("unfold", shape=(H, W, C))
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "unfold_16x16x192"

    rng = np.random.RandomState(0x0F01D)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)

    stripe_h = H // N_STRIPES
    src = np.zeros((N_STRIPES * C, 128), dtype=np.float32)
    for stripe in range(N_STRIPES):
        r0 = stripe * stripe_h
        for ch in range(C):
            block = x[ch, r0 : r0 + stripe_h, :]
            src[stripe * C + ch, : stripe_h * W] = block.reshape(-1)

    input_path = tmp_path / "x.bin"
    input_path.write_bytes(src.tobytes())
    output_path = tmp_path / "out.bin"

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        input_path=input_path,
        output_path=output_path,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=_wide_state())
    assert cycles > 0

    n_tok = (H * W) // 4
    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(4, C, -1)
    for s in range(4):
        r_ph, c_ph = s // 2, s % 2
        expected = x[:, r_ph::2, c_ph::2].reshape(C, n_tok)
        np.testing.assert_allclose(
            got[s, :, :n_tok], expected, rtol=1e-4, atol=1e-3
        )


# -- qk_scores (query-major chain) -------------------------------------


def test_resolved_qk_scores_computes_the_operation(tmp_path):
    from ipu_apps.attention.qk_scores_16x60 import N, D, N_TG, N_TPG

    verdict = resolve("qk_scores", n_tok=N, d=D)
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "qk_scores_16x60"

    rng = np.random.RandomState(0x5C0)
    Q = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)

    q_path = tmp_path / "q.bin"
    k_path = tmp_path / "k.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "out.bin"

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        query_path=q_path,
        key_path=k_path,
        output_path=output_path,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=_wide_state())
    assert cycles > 0

    expected = Q.T @ K
    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(N, N_TG, 128)
    for g in range(N_TG):
        lo = g * N_TPG
        np.testing.assert_allclose(
            got[:, g, :N_TPG], expected[:, lo : lo + N_TPG], rtol=1e-4, atol=1e-3
        )


# -- attn_scores_km (key-major chain) ------------------------------------


def test_resolved_attn_scores_km_computes_the_operation(tmp_path):
    from ipu_apps.attention.attn_scores_km_16x60 import N_TOK, D, N_TG, N_TPG, N_HEADS

    verdict = resolve("attn_scores_km", n_tok=N_TOK, d=D)
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "attn_scores_km_16x60"

    head = 1
    rng = np.random.RandomState(0x5C1)
    n_chan = N_HEADS * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)

    q_path = tmp_path / "q.bin"
    k_path = tmp_path / "k.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "out.bin"

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        input_path=q_path,
        weights_path=k_path,
        output_path=output_path,
        head=head,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=_wide_state())
    assert cycles > 0

    lo = head * D
    q_head = Q[lo : lo + D]
    k_head = K[lo : lo + D]
    expected = q_head.T @ k_head  # [query, key]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(N_TOK, N_TG, 128)
    for g in range(N_TG):
        lo_q = g * N_TPG
        np.testing.assert_allclose(
            got[:, g, :N_TPG], expected[lo_q : lo_q + N_TPG, :].T,
            rtol=1e-4, atol=1e-3,
        )


# -- attn_v (query-major chain, AGG) -------------------------------------


def test_resolved_attn_v_computes_the_operation(tmp_path):
    from ipu_apps.attention.attn_v_16x60 import (
        N_TOK, D, N_HEAD, N_CHAN, LANES, PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
    )

    verdict = resolve("attn_v", n_tok=N_TOK, d=D)
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "attn_v_16x60"

    rng = np.random.RandomState(0xA60)
    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        p_buf[h, :, :N_TOK] = P[h]
    assert p_buf[0].size == P_HEAD_STRIDE_ROWS * LANES

    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]

    p_path = tmp_path / "p.bin"
    v_path = tmp_path / "v.bin"
    p_path.write_bytes(p_buf.tobytes())
    v_path.write_bytes(v_buf.tobytes())
    output_path = tmp_path / "out.bin"

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=_wide_state())
    assert cycles > 0

    # AGG.SUM.FIRST left-folds float32 lane products starting from a Python
    # float (float64), rounding once on the R_ACC write -- see
    # test_attn_v_16x60_wide.py's _agg_reference for why a plain einsum
    # disagrees.
    expected = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            for i in range(N_TOK):
                lanes = P[h, i, :].astype(np.float32) * V[h, t, :].astype(np.float32)
                total = 0.0
                for s in range(N_TOK):
                    total += float(lanes[s])
                expected[h, i, t] = np.float32(total)

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(N_CHAN, LANES)
    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK], expected[h, :, t], rtol=1e-4, atol=1e-3
            )


# -- attn_v_bcast (key-major chain, single ACC.ADD fold) -----------------


def test_resolved_attn_v_bcast_computes_the_operation(tmp_path):
    from ipu_apps.attention.attn_v_bcast_36 import (
        N_TOK, D, N_HEAD, N_CHAN, LANES, PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
    )

    verdict = resolve("attn_v_bcast", d=D)
    assert verdict.supported, verdict.reason
    assert verdict.kernel.name == "attn_v_bcast_36"

    rng = np.random.RandomState(0xA17)
    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for s in range(N_TOK):
            p_buf[h, s, :N_TOK] = P[h, :, s]
    assert p_buf[0].size == P_HEAD_STRIDE_ROWS * LANES

    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]

    p_path = tmp_path / "p.bin"
    v_path = tmp_path / "v.bin"
    p_path.write_bytes(p_buf.tobytes())
    v_path.write_bytes(v_buf.tobytes())
    output_path = tmp_path / "out.bin"

    app = verdict.kernel.app_class(
        inst_path=_assemble(verdict.kernel),
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
        **verdict.kwargs,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=_wide_state())
    assert cycles > 0

    expected = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            acc = np.zeros(N_TOK, dtype=np.float32)
            for s in range(N_TOK):
                prod = (P[h, :, s].astype(np.float32) * np.float32(V[h, t, s])).astype(
                    np.float32
                )
                acc = prod if s == 0 else (acc + prod).astype(np.float32)
            expected[h, :, t] = acc

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(N_CHAN, 2 * LANES)
    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK], expected[h, :, t], rtol=0, atol=0
            )
