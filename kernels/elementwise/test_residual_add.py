"""The residual adds beyond each kernel's cases (E07-06): routing, and the
one-channel-per-row contract of residual_add_16x240.

The old router took ``residual_add shape=tokens,channels``; the entries now
take ``tokens`` and ``channels``. The generic kernel answers by
``num_channels``, bounded by its fixed regions (1024 rows apart).
"""

from __future__ import annotations

import numpy as np
import pytest

from ipu_harness.chain import Chain
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat

#: Each entry's (tokens, channels): the ipu_apps kernels' N_TOK and N_CH.
ENTRIES = {
    "residual_add_16x240": (16, 240),
    "residual_add_64x192": (64, 192),
    "residual_add_256x144": (256, 144),
}


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_each_entry_is_routed_by_its_exact_shape(registry, name):
    tokens, channels = ENTRIES[name]
    verdict = registry.resolve("residual_add", tokens=tokens, channels=channels)
    assert verdict.kernel_name == name, verdict.reason
    assert verdict.label == f"//kernels/elementwise:{name}"
    assert registry.kernel(name).directory.name == "residual_add_rows"
    assert not registry.resolve("residual_add", tokens=tokens, channels=channels + 1)


def test_the_generic_kernel_is_routed_by_its_channel_count(registry):
    verdict = registry.resolve("residual_add", num_channels=160)
    assert verdict.label == "//kernels/elementwise/residual_add", verdict.reason
    refused = registry.resolve("residual_add", num_channels=1025)
    assert not refused and "above its max 1024" in refused.reason


def test_residual_add_16x240_stores_one_channel_per_row():
    """Each channel owns a whole row: its 16 tokens in lanes 0..15, and the
    rest of the row exactly zero (zero padding plus zero padding), so no
    second channel's data shares it."""
    rng = np.random.RandomState(0x5ADD)
    a, b = (rng.uniform(-1.0, 1.0, size=(240, 16)).astype(np.float32) for _ in range(2))
    step = Chain(ElementFormat.FP32).run("residual_add_16x240", {}, {"a": a, "b": b})
    rows = np.frombuffer(step.result.raw_outputs["y"], dtype="<f4").reshape(240, 128)
    np.testing.assert_array_equal(rows[:, :16], step.result.outputs["y"])
    np.testing.assert_array_equal(rows[:, 16:], 0.0)
