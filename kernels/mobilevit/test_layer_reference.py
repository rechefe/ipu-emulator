"""The full-layer chains' float64 reference (E07-24, ``mobilevit_layer.reference``).

The attention-scale check the seam audit specified (from
``test_full_l{4,5}_attention_scale_is_visible``): no kernel applies
1/sqrt(head_dim), so the chains run unscaled. Two references, one without the
scale (as the kernels run) and one with it (the real model), must differ, and
their pre-softmax scores by exactly sqrt(head_dim): a reference built only to
mirror the kernels could never show the gap.
"""

from __future__ import annotations

import numpy as np
import pytest
from mobilevit_layer import L4, L5, STAGES, Layer, layer_weights, reference

#: The old tests' seeds, by layer.
SEEDS = {"L4": 0x5CA1E, "L5": 0x5CA15}


@pytest.mark.parametrize("layer", [L4, L5], ids=lambda layer: layer.name)
def test_attention_scale_is_visible(layer: Layer):
    rs = np.random.RandomState(SEEDS[layer.name])
    x = rs.uniform(-1.0, 1.0, size=(layer.channels, layer.tokens)).astype(np.float32)
    w = layer_weights(rs, layer)
    scale = 1.0 / np.sqrt(layer.head_dim)
    unscaled = reference(x, w, heads=layer.heads)
    scaled = reference(x, w, heads=layer.heads, scale=scale)
    assert float(np.max(np.abs(unscaled["resid2"] - scaled["resid2"]))) > 1e-6, (
        "the scale does not reach the layer's output: the check has no teeth"
    )
    # Softmax is not linear in the scale, so the outputs differ by no clean
    # ratio; the scores are, by exactly 1 / scale = sqrt(head_dim).
    nonzero = np.abs(scaled["scores"]) > 1e-9
    ratio = unscaled["scores"][nonzero] / scaled["scores"][nonzero]
    np.testing.assert_allclose(ratio, np.sqrt(layer.head_dim), rtol=1e-6)


@pytest.mark.parametrize("layer", [L4, L5], ids=lambda layer: layer.name)
def test_every_stage_has_its_shape(layer: Layer):
    rs = np.random.RandomState(0)
    x = rs.uniform(-1.0, 1.0, size=(layer.channels, layer.tokens)).astype(np.float32)
    stages = reference(x, layer_weights(rs, layer), heads=layer.heads)
    c, n, h = layer.channels, layer.tokens, layer.heads
    shapes = {
        "qkv": (3 * c, n),
        "scores": (h, n, n),
        "softmax": (h, n, n),
        "ffn1": (layer.hidden, n),
    }
    assert tuple(stages) == STAGES
    for name, value in stages.items():
        assert value.shape == shapes.get(name, (c, n)), name
    np.testing.assert_allclose(stages["softmax"].sum(axis=-1), 1.0, rtol=1e-12)
