"""Routing of the pooling family (E07-14): the old router facts, on the
kernel tree's registry and ``//kernels:query``.

A ``maxpool2d`` query names ``channels``, ``height``, ``width`` (the old
``shape=(C, H, W)``), ``kernel_size``, ``stride`` and ``padding``. A centred
stride-1 window (``padding = kernel_size // 2``, odd ``kernel_size``) goes to
the unrolled ``maxpool2d_nms7``/``nms9`` at 7 and 9 (cost 0 and 1) and to
``maxpool2d_window`` otherwise (cost 3). A 2x2 stride-2 pool goes to exactly
one of ``maxpool2d_stride2`` and ``maxpool2d_stride2_tail``, by width.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from ipu_harness import query
from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import expand
from ipu_harness.registry import Registry, kernel_tree
from ipu_harness.runner import run_kernel, stage
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES

FP32 = ElementFormat.FP32
CACHE = AssemblyCache()


@pytest.fixture(scope="module")
def registry():
    loaded = Registry.load(kernel_tree(), families=["pooling"])
    assert not loaded.invalid, loaded.invalid
    return loaded


def _window(channels, height, width, kernel_size, padding=None):
    padding = kernel_size // 2 if padding is None else padding
    return dict(
        channels=channels,
        height=height,
        width=width,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )


def _stride2(channels, height, width):
    return dict(channels=channels, height=height, width=width, kernel_size=2, stride=2, padding=0)


def test_the_family_is_three_kernels_and_two_entries_over_one_template(registry):
    names = {m.name for m in registry.kernels("maxpool2d")}
    assert names == {
        "maxpool2d_nms7",
        "maxpool2d_nms9",
        "maxpool2d_stride2",
        "maxpool2d_stride2_tail",
        "maxpool2d_window",
    }
    for entry, k in (("maxpool2d_nms7", 7), ("maxpool2d_nms9", 9)):
        kernel = registry.kernel(entry)
        assert kernel.directory.name == "maxpool2d_nms"
        assert (kernel.fixed["kernel_size"], kernel.fixed["padding"]) == (k, k // 2)
    assert "maxpool2d_nms" in registry.templates  # run and tested, not routed


@pytest.mark.parametrize(
    ("kernel_size", "name"),
    [(1, "maxpool2d_window"), (3, "maxpool2d_window"), (7, "maxpool2d_nms7")]
    + [(9, "maxpool2d_nms9"), (11, "maxpool2d_window"), (127, "maxpool2d_window")],
)
def test_a_centred_window_prefers_the_unrolled_kernel(registry, kernel_size, name):
    verdict = registry.resolve("maxpool2d", **_window(1, 480, 640, kernel_size))
    assert verdict.kernel_name == name, verdict.reason
    entry = name.startswith("maxpool2d_nms")
    assert verdict.label == f"//kernels/pooling{':' if entry else '/'}{name}"


def test_a_query_without_the_window_takes_the_nms7_entry(registry):
    """Each entry's fixed values are its defaults (E06-17), so a query that
    leaves ``kernel_size`` and ``padding`` out is a 7x7 window, not a tie."""
    verdict = registry.resolve("maxpool2d", channels=1, height=4, width=16, stride=1)
    assert verdict.kernel_name == "maxpool2d_nms7", verdict.reason


@pytest.mark.parametrize(
    "params",
    [
        _window(1, 2, 2, 2, padding=1),  # an even window is never centred
        _window(1, 2, 2, 9, padding=3),  # padding must be kernel_size // 2
        _window(1, 2, 2, 3, padding=0),
        _window(1, 2, 2, 129),  # wider than a tile
        dict(_window(1, 2, 2, 3), stride=2),
        _window(0, 2, 2, 3),
        dict(_stride2(1, 2, 256), padding=1),
        dict(_stride2(1, 2, 256), kernel_size=3),
        _stride2(1, 1, 256),  # a 2x2 window does not fit
        _stride2(1, 2, 1),
    ],
)
def test_unsupported_geometry_is_refused(registry, params):
    verdict = registry.resolve("maxpool2d", **params)
    assert not verdict, verdict.describe()


#: The old family test's shapes: compactly tiled widths either side of every
#: tile edge, one or two channels and 64 channels.
SHAPES = [
    (1, 2, 2),
    (1, 2, 127),
    (1, 2, 128),
    (1, 2, 129),
    (1, 2, 255),
    (1, 2, 256),
    (1, 2, 257),
    (1, 2, 258),
    (2, 4, 260),
    (1, 2, 511),
    (1, 2, 512),
    (64, 2, 511),
    (64, 2, 512),
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_exactly_one_stride2_kernel_takes_each_width_and_runs_it(registry, shape):
    """Resolve, run and check each width: the kernel that takes it, its
    registers, its XMEM reads and its output (held to the reference)."""
    channels, height, width = shape
    in_tiles = math.ceil(width / LANES)
    out_tiles = math.ceil(width // 2 / LANES)
    tail = in_tiles < 2 * out_tiles
    name = "maxpool2d_stride2_tail" if tail else "maxpool2d_stride2"
    verdict = registry.resolve("maxpool2d", **_stride2(*shape))
    assert verdict.kernel_name == name, verdict.reason
    assert verdict.alternatives == ()  # the other kernel refuses the width

    kernel = registry.kernel(name)
    case = dataclasses.replace(
        expand(kernel)[0], name="x".join(map(str, shape)), values=_stride2(*shape)
    )
    staged = stage(kernel, case, FP32, cache=CACHE)
    assert staged.placement.region("x").rows == channels * height * in_tiles
    writes = {w.register: w.value for w in staged.writes}
    assert writes["CR5"] == in_tiles
    assert writes.get("CR12", 0) == (out_tiles - 1 if tail else 0)
    result = run_kernel(kernel, case, FP32, cache=CACHE)
    assert result.passed, result.describe()
    reads_per_output_row = 6 * (out_tiles - 1) + 3 if tail else 6 * out_tiles
    assert result.stats.xmem_reads == channels * (height // 2) * reads_per_output_row
    assert np.asarray(result.outputs["y"]).shape == (channels, height // 2, width // 2)


@pytest.mark.parametrize(
    ("params", "name"),
    [
        (_window(1, 480, 640, 9), "maxpool2d_nms9"),
        (_window(2, 3, 133, 3), "maxpool2d_window"),
        (_stride2(1, 4, 256), "maxpool2d_stride2"),
        (_stride2(1, 4, 260), "maxpool2d_stride2_tail"),
    ],
)
def test_the_query_tool_gives_the_same_answer(registry, params, name, capsys):
    argv = ["maxpool2d", *(f"{k}={v}" for k, v in params.items()), "--json"]
    assert query.main(argv, registry) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["app"] == name
