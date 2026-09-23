"""Conformance suite for the MobileViT families (registry-driven).

Mirrors ``test_kernel_registry.py``'s softmax conformance test: for each op,
``resolve()`` is asked for the exact shape one of the family's own kernels
declares, and the kernel the registry hands back must actually compute the
operation -- not just claim to. Every kernel in these families (matmul,
projection, layernorm, residual_add, unfold, and the attention ops) is an
exact-shape-only match (no padding, no chunking across kernels), so there is
exactly one query worth asking per kernel: its own declared shape.

The harness is built from the verdict with ``create_harness`` (the query's
params are re-validated and ``build`` supplies any constructor kwargs, e.g.
``attn_scores_km``'s ``head``), the binary comes from ``assemble_kernel``, and
the state from the harness's own ``make_state()``.

Each op's staging convention (channel-major vs query-major vs key-major,
tg-interleaving, one-channel-per-row padding) is copied from that family's
own per-kernel cases -- the query-major (attn_v) and key-major (attn_v_bcast)
P layouts are deliberately NOT interchangeable despite sharing ctor kwarg
names. Getting a layout wrong here would silently mis-stage inputs and
produce a false failure, so nothing is re-derived -- only reused.

Query vocabulary: ``matmul`` and ``projection`` queries carry an optional
``activation`` ("none"/"silu"). The FFN1 expansion kernels fuse silu into
their store, so a plain query at an FFN1 shape must refuse and only the
``activation="silu"`` query may route there (checked below).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ipu_apps.kernel_registry import create_harness, resolve

from fixture_kernel_support import kernel_inst

LANES = 128


def _harness(verdict, params, **bindings):
    assert verdict.supported, verdict.reason
    return create_harness(verdict.kernel.name, params=params,
                          bindings={"inst_path": kernel_inst(verdict.kernel.name), **bindings})


def _rows(x: np.ndarray) -> np.ndarray:
    """Zero-pad each row of a 2-D FP32 array out to LANES lanes."""
    out = np.zeros((x.shape[0], LANES), dtype=np.float32)
    out[:, :x.shape[1]] = x
    return out


# -- matmul -------------------------------------------------------------


def test_resolved_matmul_computes_the_operation(tmp_path):
    from ipu_apps.kernels.matmul.matmul_128x128.app import M, K, N

    query = dict(shape_a=(M, K), shape_b_t=(N, K))
    verdict = resolve("matmul", **query)
    assert verdict.kernel.name == "matmul_128x128", verdict.reason

    rng = np.random.RandomState(0x128128)
    A = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)

    input_path = tmp_path / "a.bin"
    weights_path = tmp_path / "w.bin"
    output_path = tmp_path / "out.bin"
    input_path.write_bytes(A.tobytes())
    weights_path.write_bytes(W.tobytes())

    app = _harness(verdict, query, input_path=input_path, weights_path=weights_path,
                   output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(got, A @ W.T, rtol=1e-4, atol=1e-3)


def test_matmul_silu_shape_routes_only_with_activation():
    """A plain matmul at an FFN1 shape must not reach the silu-fused kernel."""
    from ipu_apps.kernels.matmul.matmul_384x192_x128.app import M, K, N

    query = dict(shape_a=(M, K), shape_b_t=(N, K))
    refused = resolve("matmul", **query)
    assert not refused and "silu" in refused.reason, refused.reason
    verdict = resolve("matmul", **query, activation="silu")
    assert verdict.kernel.name == "matmul_384x192_x128", verdict.reason


# -- projection -----------------------------------------------------------


def test_resolved_projection_computes_the_operation(tmp_path):
    from ipu_apps.kernels.projections.proj_qkv_144_p4.app import K, N_OUT, N_TG, N_TOK, N_STREAM

    query = dict(k=K, n_out=N_OUT)
    verdict = resolve("projection", **query)
    assert verdict.kernel.name == "proj_qkv_144_p4", verdict.reason

    rng = np.random.RandomState(0xC0FFEE)
    D = [
        rng.uniform(-1.0, 1.0, size=(N_TG, K, N_TOK)).astype(np.float32)
        for _ in range(N_STREAM)
    ]
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)

    # The harness takes all streams in one file, (N_STREAM, N_TG, K, N_TOK),
    # and writes one file of raw rows, (N_STREAM, N_TG, N_OUT, LANES).
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(np.stack(D).tobytes())
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(W.tobytes())
    output_path = tmp_path / "output.bin"

    app = _harness(verdict, query, input_path=input_path, weights_path=weights_path,
                   output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N_STREAM, N_TG, N_OUT, LANES)
    for p in range(N_STREAM):
        for tg in range(N_TG):
            np.testing.assert_allclose(got[p, tg][:, :N_TOK], W @ D[p][tg], rtol=1e-4, atol=1e-3)


def test_projection_silu_shape_routes_only_with_activation():
    """A plain projection at an FFN1 shape must not reach the silu-fused kernel."""
    from ipu_apps.kernels.projections.proj_ffn1_144_p4.app import K, N_OUT

    refused = resolve("projection", k=K, n_out=N_OUT)
    assert not refused and "silu" in refused.reason, refused.reason
    verdict = resolve("projection", k=K, n_out=N_OUT, activation="silu")
    assert verdict.kernel.name == "proj_ffn1_144_p4", verdict.reason


# -- layernorm --------------------------------------------------------------


def test_resolved_layernorm_computes_the_operation(tmp_path):
    from ipu_apps.kernels.normalize.layernorm_128x16.app import N_CH, N_TPG
    from ipu_apps.kernels.normalize.layernorm_cases import reference_layernorm

    query = dict(shape=(N_CH, N_TPG))
    verdict = resolve("layernorm", **query)
    assert verdict.kernel.name == "layernorm_128x16", verdict.reason

    rng = np.random.RandomState(0x1A7E)
    x = rng.uniform(-1.0, 1.0, size=(N_CH, N_TPG)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=(N_CH,)).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=(N_CH,)).astype(np.float32)

    input_path = tmp_path / "x.bin"
    gamma_path = tmp_path / "gamma.bin"
    beta_path = tmp_path / "beta.bin"
    # One zero-padded row per channel; gamma/beta stored verbatim as a row.
    input_path.write_bytes(_rows(x).tobytes())
    gamma_path.write_bytes(_rows(gamma[None, :]).tobytes())
    beta_path.write_bytes(_rows(beta[None, :]).tobytes())
    output_path = tmp_path / "out.bin"

    app = _harness(verdict, query, input_path=input_path, gamma_path=gamma_path,
                   beta_path=beta_path, output_path=output_path)
    _, cycles = app.run(max_cycles=500_000)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N_CH, LANES)[:, :N_TPG]
    np.testing.assert_allclose(got, reference_layernorm(x, gamma, beta), atol=1e-4, rtol=1e-4)


# -- residual_add -------------------------------------------------------


def test_resolved_residual_add_computes_the_operation(tmp_path):
    from ipu_apps.kernels.elementwise.residual_add_16x240.app import N_CH, N_TOK

    query = dict(shape=(N_TOK, N_CH))
    verdict = resolve("residual_add", **query)
    assert verdict.kernel.name == "residual_add_16x240", verdict.reason

    rng = np.random.RandomState(0x5ADD)
    a = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)

    a_path = tmp_path / "a.bin"
    b_path = tmp_path / "b.bin"
    a_path.write_bytes(_rows(a).tobytes())
    b_path.write_bytes(_rows(b).tobytes())
    output_path = tmp_path / "out.bin"

    app = _harness(verdict, query, input_a_path=a_path, input_b_path=b_path,
                   output_path=output_path)
    _, cycles = app.run(max_cycles=5_000_000)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N_CH, N_TOK)
    np.testing.assert_allclose(got, a + b, rtol=1e-4, atol=1e-3)


# -- unfold -----------------------------------------------------------------


def test_resolved_unfold_computes_the_operation(tmp_path):
    from ipu_apps.kernels.reshape.unfold_16x16x192.app import H, W, C, N_STRIPES

    query = dict(shape=(H, W, C))
    verdict = resolve("unfold", **query)
    assert verdict.kernel.name == "unfold_16x16x192", verdict.reason

    rng = np.random.RandomState(0x0F01D)
    x = rng.uniform(-1.0, 1.0, size=(C, H, W)).astype(np.float32)

    stripe_h = H // N_STRIPES
    src = np.zeros((N_STRIPES * C, LANES), dtype=np.float32)
    for stripe in range(N_STRIPES):
        r0 = stripe * stripe_h
        for ch in range(C):
            src[stripe * C + ch, : stripe_h * W] = x[ch, r0 : r0 + stripe_h, :].reshape(-1)

    input_path = tmp_path / "x.bin"
    input_path.write_bytes(src.tobytes())
    output_path = tmp_path / "out.bin"

    app = _harness(verdict, query, input_path=input_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    n_tok = (H * W) // 4
    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(4, C, -1)
    for s in range(4):
        r_ph, c_ph = s // 2, s % 2
        expected = x[:, r_ph::2, c_ph::2].reshape(C, n_tok)
        np.testing.assert_allclose(got[s, :, :n_tok], expected, rtol=1e-4, atol=1e-3)


# -- qk_scores (query-major chain) -------------------------------------


def test_resolved_qk_scores_computes_the_operation(tmp_path):
    from ipu_apps.kernels.attention.qk_scores_16x60.app import N, D, N_TG, N_TPG

    query = dict(n_tok=N, d=D)
    verdict = resolve("qk_scores", **query)
    assert verdict.kernel.name == "qk_scores_16x60", verdict.reason

    rng = np.random.RandomState(0x5C0)
    Q = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)

    q_path = tmp_path / "q.bin"
    k_path = tmp_path / "k.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "out.bin"

    app = _harness(verdict, query, query_path=q_path, key_path=k_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    expected = Q.T @ K
    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N, N_TG, LANES)
    for g in range(N_TG):
        lo = g * N_TPG
        np.testing.assert_allclose(got[:, g, :N_TPG], expected[:, lo : lo + N_TPG],
                                   rtol=1e-4, atol=1e-3)


# -- attn_scores_km (key-major chain) ------------------------------------


def test_resolved_attn_scores_km_computes_the_operation(tmp_path):
    from ipu_apps.kernels.attention.attn_scores_km_16x60.app import N_TOK, D, N_TG, N_TPG, N_HEADS

    head = 1
    # `head` is part of the query: the kernel scores one selected head, and
    # the registry validates it and passes it to the constructor via build.
    query = dict(n_tok=N_TOK, d=D, head=head)
    verdict = resolve("attn_scores_km", **query)
    assert verdict.kernel.name == "attn_scores_km_16x60", verdict.reason
    assert verdict.kwargs == {"head": head}

    rng = np.random.RandomState(0x5C1)
    n_chan = N_HEADS * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)

    q_path = tmp_path / "q.bin"
    k_path = tmp_path / "k.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "out.bin"

    app = _harness(verdict, query, input_path=q_path, weights_path=k_path, output_path=output_path)
    assert app.head == head
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    lo = head * D
    expected = Q[lo : lo + D].T @ K[lo : lo + D]  # [query, key]

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N_TOK, N_TG, LANES)
    for g in range(N_TG):
        lo_q = g * N_TPG
        np.testing.assert_allclose(got[:, g, :N_TPG], expected[lo_q : lo_q + N_TPG, :].T,
                                   rtol=1e-4, atol=1e-3)


# -- attn_v (query-major chain, AGG) -------------------------------------


def test_resolved_attn_v_computes_the_operation(tmp_path):
    from ipu_apps.kernels.attention.attn_v_16x60.app import (
        N_TOK, D, N_HEAD, N_CHAN, PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
    )

    query = dict(n_tok=N_TOK, d=D)
    verdict = resolve("attn_v", **query)
    assert verdict.kernel.name == "attn_v_16x60", verdict.reason

    rng = np.random.RandomState(0xA60)
    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    p_buf[:, :, :N_TOK] = P
    assert p_buf[0].size == P_HEAD_STRIDE_ROWS * LANES
    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    v_buf[:, :N_TOK] = V.reshape(N_CHAN, N_TOK)

    p_path = tmp_path / "p.bin"
    v_path = tmp_path / "v.bin"
    p_path.write_bytes(p_buf.tobytes())
    v_path.write_bytes(v_buf.tobytes())
    output_path = tmp_path / "out.bin"

    app = _harness(verdict, query, p_path=p_path, v_path=v_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    # AGG.SUM.FIRST left-folds float32 lane products starting from a Python
    # float (float64), rounding once on the R_ACC write -- a plain einsum
    # disagrees in the last bits.
    expected = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            for i in range(N_TOK):
                lanes = P[h, i, :].astype(np.float32) * V[h, t, :].astype(np.float32)
                total = 0.0
                for s in range(N_TOK):
                    total += float(lanes[s])
                expected[h, i, t] = np.float32(total)

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N_CHAN, LANES)
    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(got[h * D + t, :N_TOK], expected[h, :, t], rtol=1e-4, atol=1e-3)


# -- attn_v_bcast (key-major chain, single ACC.ADD fold) -----------------


def test_resolved_attn_v_bcast_computes_the_operation(tmp_path):
    from ipu_apps.kernels.attention.attn_v_bcast_36.app import (
        N_TOK, D, N_HEAD, N_CHAN, PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
    )

    query = dict(d=D)
    verdict = resolve("attn_v_bcast", **query)
    assert verdict.kernel.name == "attn_v_bcast_36", verdict.reason
    # The optional n_tok must agree with the kernel's fixed token count.
    assert resolve("attn_v_bcast", d=D, n_tok=N_TOK).kernel.name == "attn_v_bcast_36"

    rng = np.random.RandomState(0xA17)
    P = rng.uniform(-1.0, 1.0, size=(N_HEAD, N_TOK, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    # Key-major P: row s of head h holds P[h, :, s].
    p_buf = np.zeros((N_HEAD, N_TOK, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    p_buf[:, :, :N_TOK] = P.transpose(0, 2, 1)
    assert p_buf[0].size == P_HEAD_STRIDE_ROWS * LANES
    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    v_buf[:, :N_TOK] = V.reshape(N_CHAN, N_TOK)

    p_path = tmp_path / "p.bin"
    v_path = tmp_path / "v.bin"
    p_path.write_bytes(p_buf.tobytes())
    v_path.write_bytes(v_buf.tobytes())
    output_path = tmp_path / "out.bin"

    app = _harness(verdict, query, p_path=p_path, v_path=v_path, output_path=output_path)
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    expected = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            acc = np.zeros(N_TOK, dtype=np.float32)
            for s in range(N_TOK):
                prod = (P[h, :, s].astype(np.float32) * np.float32(V[h, t, s])).astype(np.float32)
                acc = prod if s == 0 else (acc + prod).astype(np.float32)
            expected[h, :, t] = acc

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N_CHAN, 2 * LANES)
    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(got[h * D + t, :N_TOK], expected[h, :, t], rtol=0, atol=0)
