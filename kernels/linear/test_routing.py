"""Routing of the linear family (E07-21): the ``ipu_apps`` routing facts of
``fully_connected``, on the family's registry.

The old router took ``shape=(10, 128)``, ``weight_shape=(64, 128)``, a
``dtype`` and ``wide_mode``. The shapes are now the named parameters
``samples``, ``in_features`` and ``out_features``, each with its one value,
and the data format is the run's element format (INT32 is the old
``wide_mode``), not a parameter.
"""

from __future__ import annotations

import pytest

from ipu_harness.registry import Registry, kernel_tree


@pytest.fixture(scope="module")
def registry():
    loaded = Registry.load(kernel_tree(), families=["linear"])
    assert not loaded.invalid, loaded.invalid
    return loaded


def test_fully_connected_takes_its_one_shape_in_every_format(registry):
    for fmt in (None, "int8", "int32", "e4", "e5"):
        verdict = registry.resolve("fully_connected", format=fmt)
        assert verdict.kernel_name == "fully_connected", verdict.reason
        assert verdict.label == "//kernels/linear/fully_connected"
    full = registry.resolve("fully_connected", samples=10, in_features=128, out_features=64)
    assert full.kernel_name == "fully_connected", full.reason


@pytest.mark.parametrize("params", [{"samples": 11}, {"in_features": 256}, {"out_features": 128}])
def test_another_shape_is_refused(registry, params):
    verdict = registry.resolve("fully_connected", **params)
    assert not verdict
    assert next(iter(params)) in verdict.reason


def test_another_format_is_refused(registry):
    assert not registry.resolve("fully_connected", format="fp32")
