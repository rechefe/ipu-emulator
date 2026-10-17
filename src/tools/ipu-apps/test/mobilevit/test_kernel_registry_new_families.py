"""Routing of the old matmul kernels (the last of the MobileViT families here).

``matmul`` queries carry an optional ``activation`` ("none"/"silu"). The FFN1
expansion kernels fuse silu into their store, so a plain query at an FFN1
shape must refuse and only the ``activation="silu"`` query may route there.
The other families moved, with their conformance checks: the projections to
``//kernels/projections`` (E07-02), unfold to ``//kernels/reshape`` (E07-05),
the residual adds to ``//kernels/elementwise`` (E07-06), the LayerNorms to
``//kernels/normalize`` (E07-10) and the attention ops to ``//kernels/attention``
(E07-12, E07-13, E07-24), whose own tests check their routing.
"""

from __future__ import annotations

from ipu_apps.kernel_registry import resolve

# -- matmul -------------------------------------------------------------


def test_matmul_silu_shape_routes_only_with_activation():
    """A plain matmul at an FFN1 shape must not reach the silu-fused kernel."""
    from ipu_apps.kernels.matmul.matmul_384x192_x128.app import K, M, N

    query = dict(shape_a=(M, K), shape_b_t=(N, K))
    refused = resolve("matmul", **query)
    assert not refused and "silu" in refused.reason, refused.reason
    verdict = resolve("matmul", **query, activation="silu")
    assert verdict.kernel.name == "matmul_384x192_x128", verdict.reason
