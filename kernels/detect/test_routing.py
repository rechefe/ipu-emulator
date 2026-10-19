"""Routing of the detect family (E07-15): the old queries
``channel_peak shape=(C, N)`` and ``score_threshold shape=(N,)``, as named
parameters on the kernel tree's registry and ``//kernels:query``. The
threshold, temperature and the cases' data are not part of a query: they
have defaults."""

from __future__ import annotations

import json

import pytest

from ipu_harness import query
from ipu_harness.registry import Registry, kernel_tree


@pytest.fixture(scope="module")
def registry():
    loaded = Registry.load(kernel_tree(), families=["detect"])
    assert not loaded.invalid, loaded.invalid
    return loaded


@pytest.mark.parametrize(
    ("op", "params"),
    [("channel_peak", dict(channels=65, columns=4800)), ("score_threshold", dict(count=4800))],
)
def test_each_kernel_takes_its_shape_query(registry, op, params, capsys):
    verdict = registry.resolve(op, **params)
    assert verdict.kernel_name == op, verdict.reason
    assert verdict.label == f"//kernels/detect/{op}"
    assert query.main([op, *(f"{k}={v}" for k, v in params.items()), "--json"], registry) == 0
    assert json.loads(capsys.readouterr().out)["app"] == op


@pytest.mark.parametrize(
    ("op", "params"),
    [
        ("channel_peak", dict(channels=0, columns=2)),
        ("channel_peak", dict(channels=2, columns=0)),
        ("score_threshold", dict(count=0)),
        ("score_threshold", dict(count=16, temperature=0.0)),
    ],
)
def test_empty_shapes_are_refused(registry, op, params):
    verdict = registry.resolve(op, **params)
    assert not verdict, verdict.describe()
