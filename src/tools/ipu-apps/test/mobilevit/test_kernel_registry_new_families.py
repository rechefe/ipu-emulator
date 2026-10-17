"""Conformance suite for the MobileViT families (registry-driven).

Mirrors ``test_kernel_registry.py``'s softmax conformance test: for each op,
``resolve()`` is asked for the exact shape one of the family's own kernels
declares, and the kernel the registry hands back must actually compute the
operation -- not just claim to. Every kernel in these families (matmul,
layernorm, residual_add, unfold) is an exact-shape-only match (no padding, no chunking across kernels), so there is
exactly one query worth asking per kernel: its own declared shape.

The harness is built from the verdict with ``create_harness`` (the query's
params are re-validated and ``build`` supplies any constructor kwargs), the binary comes from ``assemble_kernel``, and
the state from the harness's own ``make_state()``.

Each op's staging convention (channel-major, tg-interleaving,
one-channel-per-row padding) is copied from that family's own per-kernel
cases. Getting a layout wrong here would silently mis-stage inputs and
produce a false failure, so nothing is re-derived -- only reused.

Query vocabulary: ``matmul`` queries carry an optional ``activation``
("none"/"silu"). The FFN1 expansion kernels fuse silu into their store, so a
plain query at an FFN1 shape must refuse and only the ``activation="silu"``
query may route there (checked below). The projections moved to
``//kernels/projections`` (E07-02), whose ``test_routing.py`` checks theirs,
and the attention ops to ``//kernels/attention`` (E07-12, E07-13, E07-24),
whose ``test_routing.py`` and kernel cases check theirs.
"""

from __future__ import annotations

import numpy as np
from fixture_kernel_support import kernel_inst

from ipu_apps.kernel_registry import create_harness, resolve

LANES = 128


def _harness(verdict, params, **bindings):
    assert verdict.supported, verdict.reason
    return create_harness(
        verdict.kernel.name,
        params=params,
        bindings={"inst_path": kernel_inst(verdict.kernel.name), **bindings},
    )


def _rows(x: np.ndarray) -> np.ndarray:
    """Zero-pad each row of a 2-D FP32 array out to LANES lanes."""
    out = np.zeros((x.shape[0], LANES), dtype=np.float32)
    out[:, : x.shape[1]] = x
    return out


# -- matmul -------------------------------------------------------------


def test_resolved_matmul_computes_the_operation(tmp_path):
    from ipu_apps.kernels.matmul.matmul_128x128.app import K, M, N

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

    app = _harness(
        verdict, query, input_path=input_path, weights_path=weights_path, output_path=output_path
    )
    _, cycles = app.run(max_cycles=20_000_000)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(got, A @ W.T, rtol=1e-4, atol=1e-3)


def test_matmul_silu_shape_routes_only_with_activation():
    """A plain matmul at an FFN1 shape must not reach the silu-fused kernel."""
    from ipu_apps.kernels.matmul.matmul_384x192_x128.app import K, M, N

    query = dict(shape_a=(M, K), shape_b_t=(N, K))
    refused = resolve("matmul", **query)
    assert not refused and "silu" in refused.reason, refused.reason
    verdict = resolve("matmul", **query, activation="silu")
    assert verdict.kernel.name == "matmul_384x192_x128", verdict.reason


# -- layernorm --------------------------------------------------------------


def test_resolved_layernorm_computes_the_operation(tmp_path):
    from ipu_apps.kernels.normalize.cases import reference_layernorm
    from ipu_apps.kernels.normalize.layernorm_128x16.app import N_CH, N_TPG

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

    app = _harness(
        verdict,
        query,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
        output_path=output_path,
    )
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

    app = _harness(
        verdict, query, input_a_path=a_path, input_b_path=b_path, output_path=output_path
    )
    _, cycles = app.run(max_cycles=5_000_000)
    assert cycles > 0

    got = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N_CH, N_TOK)
    np.testing.assert_allclose(got, a + b, rtol=1e-4, atol=1e-3)


# -- unfold -----------------------------------------------------------------


def test_resolved_unfold_computes_the_operation(tmp_path):
    from ipu_apps.kernels.reshape.unfold_16x16x192.app import N_STRIPES, C, H, W

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
