"""Conformance suite for the MobileViT families (registry-driven).

Mirrors ``test_kernel_registry.py``'s softmax conformance test: for each op,
``resolve()`` is asked for the exact shape one of the family's own kernels
declares, and the kernel the registry hands back must actually compute the
operation -- not just claim to. Every kernel in this family (matmul) is an
exact-shape-only match (no padding, no chunking across kernels), so there is
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
``//kernels/projections`` (E07-02), unfold to ``//kernels/reshape`` (E07-05),
the residual adds to ``//kernels/elementwise`` (E07-06), the LayerNorms to
``//kernels/normalize`` (E07-10) and the attention ops to ``//kernels/attention``
(E07-12, E07-13, E07-24), whose own tests check their routing.
"""

from __future__ import annotations

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


# -- matmul -------------------------------------------------------------


def test_matmul_silu_shape_routes_only_with_activation():
    """A plain matmul at an FFN1 shape must not reach the silu-fused kernel."""
    from ipu_apps.kernels.matmul.matmul_384x192_x128.app import K, M, N

    query = dict(shape_a=(M, K), shape_b_t=(N, K))
    refused = resolve("matmul", **query)
    assert not refused and "silu" in refused.reason, refused.reason
    verdict = resolve("matmul", **query, activation="silu")
    assert verdict.kernel.name == "matmul_384x192_x128", verdict.reason
