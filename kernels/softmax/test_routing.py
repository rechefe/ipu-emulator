"""Routing of the softmax family (E07-11): the old router tests, on
the kernel tree's registry, and the routing tables of the docs page.

A query is ``softmax`` with ``rows``, ``dim`` and the row length: ``n`` along
each row (``dim = 1``), ``width`` down each column (``dim = 0``). The five
kernels split each axis by that length; the row count never changes the
answer. A query the registry accepts must also plan and place: an accepted
query that cannot run is a routing bug.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ipu_harness import query
from ipu_harness.abi import register_writes
from ipu_harness.packing import stored_rows
from ipu_harness.params import plan_case
from ipu_harness.placement import place
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat

KERNELS = (
    "softmax_columns",
    "softmax_columns_packed",
    "softmax_rows",
    "softmax_rows_long",
    "softmax_rows_partial",
)


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


def along_rows(registry, rows, n):
    """``torch.softmax(x, dim=1)`` on a ``(rows, n)`` matrix."""
    return registry.resolve("softmax", rows=rows, n=n, dim=1)


def down_columns(registry, rows, width):
    """``torch.softmax(x, dim=0)`` on a ``(rows, width)`` matrix."""
    return registry.resolve("softmax", rows=rows, width=width, dim=0)


def runs(verdict):
    """Plan and place the verdict's query, and set its registers, as a run would."""
    kernel = verdict.kernel
    plan = plan_case(kernel, dict(verdict.params), ElementFormat.FP32)
    placement = place(plan, {name: stored_rows(t, plan) for name, t in kernel.tensors.items()})
    register_writes(plan, placement)
    return plan


def test_the_family_is_five_kernels(registry):
    assert sorted(m.name for m in registry.kernels("softmax")) == list(KERNELS)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "softmax_rows_partial"),
        (64, "softmax_rows_partial"),
        (127, "softmax_rows_partial"),
        (128, "softmax_rows"),
        (200, "softmax_rows_long"),
        (300, "softmax_rows_long"),
    ],
)
def test_rows_axis_routing(registry, n, expected):
    verdict = along_rows(registry, 8, n)
    assert verdict.kernel_name == expected, verdict.reason


def test_n_128_goes_to_the_full_width_kernel_over_the_packed_one(registry):
    """softmax_rows_partial takes n = 128 too (P = 1), at a higher cost."""
    verdict = along_rows(registry, 8, 128)
    assert verdict.kernel_name == "softmax_rows"
    assert verdict.alternatives == ("softmax_rows_partial",)


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (1, "softmax_columns_packed"),
        (64, "softmax_columns_packed"),
        (65, "softmax_columns"),
        (96, "softmax_columns"),
        (127, "softmax_columns"),
        (128, "softmax_columns"),
        (384, "softmax_columns"),
    ],
)
def test_columns_axis_routing(registry, width, expected):
    verdict = down_columns(registry, 8, width)
    assert verdict.kernel_name == expected, verdict.reason


@pytest.mark.parametrize("n", [256, 384, 512, 1024])
def test_a_multiple_of_128_is_a_long_row_with_no_tail_chunk(registry, n):
    verdict = along_rows(registry, 8, n)
    assert verdict.kernel_name == "softmax_rows_long"
    plan = runs(verdict)
    assert plan.derived["tail"] == 0  # valid_elements 0: the tail block is a no-op
    assert plan.derived["cpr"] == n // 128  # no phantom tail chunk


@pytest.mark.parametrize("rows", [1, 4, 128, 129, 200, 1000])
def test_every_row_query_is_accepted_and_runs(registry, rows):
    """No row count is capped: each row kernel loops over groups of 128 rows."""
    for n in (1, 8, 16, 17, 32, 33, 64, 65, 100, 127, 128, 129, 200, 300, 511):
        verdict = along_rows(registry, rows, n)
        assert verdict, f"rows={rows} n={n}: {verdict.reason}"
        runs(verdict)


@pytest.mark.parametrize("rows", [1, 4, 128])
def test_every_column_query_is_accepted_and_runs(registry, rows):
    for width in (1, 16, 32, 64, 65, 100, 127, 128, 129, 256, 384):
        verdict = down_columns(registry, rows, width)
        assert verdict, f"rows={rows} width={width}: {verdict.reason}"
        runs(verdict)


def test_the_columns_axis_has_no_row_cap(registry):
    """Column softmax reduces down the rows with full-vector scalars, not one
    slot per row: any row count is a loop bound."""
    runs(down_columns(registry, 5000, 128))


@pytest.mark.parametrize("dim", [1, 0])
def test_no_open_gaps(registry, dim):
    """Every length 1..599 along either axis routes to some kernel."""
    length = "n" if dim == 1 else "width"
    found = query.boundaries(
        registry, "softmax", range(1, 600), lambda v: {"rows": 8, length: v, "dim": dim}
    )
    assert all(run.kernel for run in found), [run.render(length) for run in found]


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_sizes_are_refused(registry, bad):
    assert not along_rows(registry, 8, bad)
    assert not along_rows(registry, bad, 128)
    assert not down_columns(registry, 8, bad)
    assert not down_columns(registry, bad, 16)


def test_the_axis_decides_the_kernel(registry):
    """The same (300, 16) tensor routes by which axis is reduced."""
    assert along_rows(registry, 300, 16).kernel_name == "softmax_rows_partial"
    assert down_columns(registry, 300, 16).kernel_name == "softmax_columns_packed"


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        ("softmax_rows", ["rows=32", "n=128", "dim=1"]),
        ("softmax_rows_partial", ["rows=32", "n=20", "dim=1"]),
        ("softmax_rows_long", ["rows=32", "n=300", "dim=1"]),
        ("softmax_columns", ["rows=32", "width=200", "dim=0"]),
        ("softmax_columns_packed", ["rows=32", "width=20", "dim=0"]),
    ],
)
def test_the_query_tool_gives_the_same_answer(registry, kernel, args, capsys):
    assert query.main(["softmax", *args, "--json"], registry) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["app"] == kernel
    assert answer["use"].startswith(f"//kernels/softmax/{kernel}")


# -- the docs page's routing tables -------------------------------------------

# Runfiles and a plain checkout put the docs in the same place relative to the
# repository root, so walk up until it appears.
DOC = next(
    (
        parent / "docs/content/kernels/softmax.md"
        for parent in Path(__file__).resolve().parents
        if (parent / "docs/content/kernels/softmax.md").exists()
    ),
    None,
)

# `n 1..127   softmax_rows_partial`, `n = 128  softmax_rows`, `n 129..  <name>`
ROW = re.compile(
    r"^(?P<param>n|width)\s+(?:=\s*(?P<exact>\d+)|(?P<start>\d+)\.\.(?P<end>\d+)?)"
    r"\s+(?P<kernel>\w+)\s*$"
)

#: The axis each documented parameter describes.
DIM = {"n": 1, "width": 0}

#: How far past an open-ended run ("n 129..") to probe.
OPEN_ENDED_PROBE = 512


def documented_runs() -> list[tuple[str, int, int | None, str]]:
    # The page is data of :tests: if it is missing, collection fails rather
    # than the tests silently checking nothing.
    if DOC is None:
        raise FileNotFoundError("docs/content/kernels/softmax.md is not in the runfiles")
    found = []
    for block in re.findall(r"^```[^\n]*\n(.*?)^```", DOC.read_text(), re.S | re.M):
        for line in block.splitlines():
            match = ROW.match(line.strip())
            if not match:
                continue
            if match["exact"]:
                start = end = int(match["exact"])
            else:
                start, end = int(match["start"]), int(match["end"]) if match["end"] else None
            found.append((match["param"], start, end, match["kernel"]))
    return found


def documented(registry, param, value):
    return registry.resolve("softmax", rows=8, dim=DIM[param], **{param: value})


def test_the_doc_has_routing_tables():
    """Guards the parser: a rewrite that drops the fences must not leave the
    tests below checking nothing."""
    found = documented_runs()
    assert len(found) >= 5, f"parsed only {len(found)} documented runs from {DOC}"
    assert {param for param, *_ in found} == {"n", "width"}


@pytest.mark.parametrize(("param", "start", "end", "kernel"), documented_runs())
def test_documented_routing_matches_the_registry(registry, param, start, end, kernel):
    for value in {start, end if end is not None else OPEN_ENDED_PROBE}:
        verdict = documented(registry, param, value)
        assert verdict.kernel_name == kernel, (
            f"the docs say {param}={value} routes to {kernel}; the registry answers "
            f"{verdict.kernel_name} ({verdict.reason}). Re-probe the table in {DOC.name}."
        )


@pytest.mark.parametrize(
    ("param", "start", "end", "kernel"),
    [run for run in documented_runs() if run[1] > 1],  # a run from 1 has nothing below it
)
def test_documented_boundaries_are_real_boundaries(registry, param, start, end, kernel):
    """A run documented to start at ``start`` changes hands there."""
    below = documented(registry, param, start - 1)
    assert below.kernel_name != kernel, (
        f"the docs start a {kernel} run at {param}={start}, but {param}={start - 1} "
        "routes there too"
    )
