"""Routing (E07-10): each LayerNorm entry is found by exactly its shape, and
l2_normalize by any positive (channels, columns).

The old router took ``layernorm shape=channels,tokens`` and ``l2_normalize
shape=channels,columns``; the kernels now take the separate parameters, and
layernorm also ``dim`` (0: the channels, as ops.layernorm reduces them).
"""

from __future__ import annotations

import pytest

from ipu_harness.registry import default_registry

#: Each entry's (channels, tokens): the old kernels' N_CH and tokens.
ENTRIES = {
    "layernorm_128x16": (16, 128),
    "layernorm_256x144": (144, 256),
    "layernorm_64x192": (192, 64),
    "layernorm_16x240": (240, 16),
}


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_each_layernorm_entry_is_routed_by_its_exact_shape(registry, name):
    channels, tokens = ENTRIES[name]
    verdict = registry.resolve("layernorm", channels=channels, tokens=tokens)
    assert verdict.kernel_name == name, verdict.reason
    assert verdict.label == f"//kernels/normalize:{name}"
    assert registry.kernel(name).directory.name == "layernorm"


def test_a_layernorm_shape_no_entry_has_is_refused(registry):
    assert not registry.resolve("layernorm", channels=192, tokens=16)
    assert not registry.resolve("layernorm", channels=100, tokens=64)


@pytest.mark.parametrize("channels, columns", [(4, 16), (3, 133), (256, 4800)])
def test_l2_normalize_takes_any_positive_shape(registry, channels, columns):
    verdict = registry.resolve("l2_normalize", channels=channels, columns=columns)
    assert verdict.label == "//kernels/normalize/l2_normalize_channels", verdict.reason
    assert not registry.resolve("l2_normalize", channels=0, columns=columns)


#: Each packed LayerNorm entry's (channels, tokens) (E07-23).
PACKED = {"packed_layernorm_192x64": (192, 64), "packed_layernorm_240x16": (240, 16)}


@pytest.mark.parametrize("name", sorted(PACKED))
def test_each_packed_layernorm_is_routed_by_its_shape_and_never_as_layernorm(registry, name):
    channels, tokens = PACKED[name]
    verdict = registry.resolve("packed_layernorm", channels=channels, tokens=tokens)
    assert verdict.kernel_name == name, verdict.reason
    assert registry.kernel(name).directory.name == "packed_layernorm"
    # the packed layout's tensors are not layernorm's: no layernorm query reaches it
    assert registry.resolve("layernorm", channels=channels, tokens=tokens).kernel_name != name
