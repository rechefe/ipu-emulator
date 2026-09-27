"""Softmax family tests, beyond each kernel's own cases and test.py.

* **Layout contract** -- every softmax app writes its output in the SAME
  row-major layout as its input, checked by reshaping the raw output file with
  no app-specific un-packing (so a misplaced element cannot hide behind an
  un-pack step).
* **Router** -- the registry's softmax verdicts agree with the apps
  themselves: every ``supported`` verdict constructs, no shape is left
  uncovered, and torch's ``dim`` convention (negative dims, 1-D, rank > 2) is
  honoured.
* **Docs routing** -- the routing tables in ``docs/content/kernels/softmax.md``
  match what the registry answers, boundaries included.
"""

from __future__ import annotations

import importlib
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_apps.kernel_registry import boundaries, resolve
from ipu_apps.kernel_registry.cases import assemble_kernel
from ipu_apps.kernels.softmax.softmax_columns.app import SoftmaxColumnsApp
from ipu_apps.kernels.softmax.softmax_columns_packed.app import SoftmaxColumnsPackedApp
from ipu_apps.kernels.softmax.softmax_rows.app import SoftmaxRowsApp
from ipu_apps.kernels.softmax.softmax_rows_long.app import SoftmaxRowsLongApp
from ipu_apps.kernels.softmax.softmax_rows_partial.app import SoftmaxRowsPartialApp


# -- layout contract --------------------------------------------------------

# (id, app class, kernel name, ctor kwargs, (rows, cols), softmax axis).
# Shapes are chosen so each app hits its padding-heavy regime -- that is where
# an input/output layout divergence can hide.
LAYOUT_CASES = [
    ("rows",            SoftmaxRowsApp,          "softmax_rows",
     dict(rows=6), (6, 128), 1),
    # N < ps (20 of 32 lanes) AND rows % P != 0 (10 rows, P=4 -> padded to 12):
    # both padding kinds at once.
    ("partial_pad",     SoftmaxRowsPartialApp,   "softmax_rows_partial",
     dict(n=20, rows=10), (10, 20), 1),
    # N == ps and rows % P == 0: no padding at all, must still round-trip.
    ("partial_exact",   SoftmaxRowsPartialApp,   "softmax_rows_partial",
     dict(n=32, rows=8), (8, 32), 1),
    ("partial_p8",      SoftmaxRowsPartialApp,   "softmax_rows_partial",
     dict(n=5, rows=7), (7, 5), 1),
    # n % 128 != 0 -> a padded tail chunk on device.
    ("long_tail",       SoftmaxRowsLongApp,      "softmax_rows_long",
     dict(n=300, rows=5), (5, 300), 1),
    ("long_exact",      SoftmaxRowsLongApp,      "softmax_rows_long",
     dict(n=256, rows=3), (3, 256), 1),
    # width not a multiple of 128 -> padding lanes on device.
    ("columns_pad",     SoftmaxColumnsApp,       "softmax_columns",
     dict(rows=9, width=200), (9, 200), 0),
    ("columns_exact",   SoftmaxColumnsApp,       "softmax_columns",
     dict(rows=9, width=128), (9, 128), 0),
    # width not a pow2 and rows % rows_per_vec != 0.
    ("cols_packed_pad", SoftmaxColumnsPackedApp, "softmax_columns_packed",
     dict(rows=10, width=20), (10, 20), 0),
    ("cols_packed_exact", SoftmaxColumnsPackedApp, "softmax_columns_packed",
     dict(rows=8, width=32), (8, 32), 0),
]


def _reference(x: np.ndarray, axis: int) -> np.ndarray:
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


@pytest.mark.parametrize(
    "app_cls,kernel,kwargs,shape,axis",
    [c[1:] for c in LAYOUT_CASES],
    ids=[c[0] for c in LAYOUT_CASES],
)
def test_output_file_matches_input_layout(app_cls, kernel, kwargs, shape, axis):
    x = (np.random.RandomState(sum(shape)).randn(*shape) * 3.0).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        inst = assemble_kernel(kernel, tmp)
        inp = Path(tmp) / "in.bin"
        outp = Path(tmp) / "out.bin"
        inp.write_bytes(x.tobytes())

        app_cls(inst_path=inst, input_path=inp, output_path=outp,
                **kwargs).run(max_cycles=20_000_000)

        in_bytes = inp.stat().st_size
        out_bytes = outp.stat().st_size
        raw = np.frombuffer(outp.read_bytes(), dtype=np.float32)

    # 1. Same file size -- no padding leaked out, nothing truncated.
    assert out_bytes == in_bytes, (
        f"output file is {out_bytes} B but input was {in_bytes} B: the "
        f"on-device padding is leaking into the output layout"
    )

    # 2. A naive reshape (no app-specific un-packing) must be correct, which
    #    pins element ORDER, not just size.
    out = raw.reshape(shape)
    assert np.abs(out - _reference(x, axis)).max() < 1e-4


# -- router -------------------------------------------------------------------

_CTOR_KWARGS = dict(inst_path="x", input_path="y", output_path=None)


def _construct(verdict):
    mod_name, cls_name = verdict.app_class.rsplit(".", 1)
    app_cls = getattr(importlib.import_module(mod_name), cls_name)
    return app_cls(**_CTOR_KWARGS, **verdict.kwargs)


def _along_rows(rows, n):
    """``torch.softmax(x, dim=1)`` on a ``(rows, n)`` matrix."""
    return resolve("softmax", shape=(rows, n), dim=1)


def _down_columns(rows, width):
    """``torch.softmax(x, dim=0)`` on a ``(rows, width)`` matrix."""
    return resolve("softmax", shape=(rows, width), dim=0)


@pytest.mark.parametrize("n", [1, 8, 16, 17, 32, 33, 64, 65, 100, 127, 128, 129, 200, 300, 511])
@pytest.mark.parametrize("rows", [1, 4, 128, 129, 1000])
def test_supported_rows_verdicts_construct(n, rows):
    """A 'supported' answer must never fail at construction time."""
    verdict = _along_rows(rows, n)
    if verdict.supported:
        _construct(verdict)


@pytest.mark.parametrize("width", [1, 16, 32, 64, 65, 100, 127, 128, 129, 256, 384])
@pytest.mark.parametrize("rows", [1, 4, 128])
def test_supported_column_verdicts_construct(width, rows):
    verdict = _down_columns(rows, width)
    if verdict.supported:
        _construct(verdict)


@pytest.mark.parametrize("n,expected", [
    (1, "softmax_rows_partial"),
    (64, "softmax_rows_partial"),
    (127, "softmax_rows_partial"),
    (128, "softmax_rows"),
    (200, "softmax_rows_long"),
    (300, "softmax_rows_long"),
])
def test_rows_axis_routing(n, expected):
    assert _along_rows(8, n).app_name == expected


@pytest.mark.parametrize("width,expected", [
    (1, "softmax_columns_packed"),
    (64, "softmax_columns_packed"),
    (128, "softmax_columns"),
    (384, "softmax_columns"),
])
def test_columns_axis_routing(width, expected):
    assert _down_columns(8, width).app_name == expected


def test_no_verdict_warns_about_packed_output():
    """Every app writes its output in the same row-major layout as its input,
    so no verdict should be telling callers to un-pack anything."""
    for n in (8, 20, 32, 64, 100, 128, 300):
        for rows in (7, 64, 300):
            assert not any(
                "PACKED" in c or "mis-align" in c
                for c in _along_rows(rows, n).caveats
            )


@pytest.mark.parametrize("n", [256, 384, 512, 1024])
def test_rows_multiple_of_128_supported(n):
    """n > 128 and divisible by 128: softmax_rows_long with no tail chunk."""
    verdict = _along_rows(8, n)
    assert verdict.supported
    assert verdict.app_name == "softmax_rows_long"
    app = _construct(verdict)
    assert app.tail == 0
    assert app.chunks_per_row == n // 128   # no phantom tail chunk


@pytest.mark.parametrize("width", [65, 96, 127])
def test_column_width_65_to_127_supported(width):
    """Formerly a gap; softmax_columns handles it (padding lanes are separate
    columns, so they never pollute a real column's reduce)."""
    verdict = _down_columns(8, width)
    assert verdict.supported
    assert verdict.app_name == "softmax_columns"
    _construct(verdict)
    # The honesty requirement: say that lanes go idle.
    assert any("idle" in c for c in verdict.caveats)


@pytest.mark.parametrize("dim", [1, 0])
def test_no_open_gaps(dim):
    """Every length along either axis routes to some kernel."""
    runs = boundaries("softmax", "shape", range(1, 600), build=lambda v: (8, v), dim=dim)
    assert all(run.kernel for run in runs), [run.render("length") for run in runs]


@pytest.mark.parametrize("rows", [129, 200, 1000])
@pytest.mark.parametrize("n", [8, 32, 128, 300])
def test_rows_over_128_supported(n, rows):
    """Every row app now loops groups of <=128 rows internally, so the former
    row-index overflow (a row index >=128 selecting the unloaded R1) is gone
    and there is no row-count cap on any row-axis path."""
    verdict = _along_rows(rows, n)
    assert verdict.supported
    _construct(verdict)
    # The cap is gone from the advertised limits too, not just the verdict.
    # (Other caveats may mention 128 -- e.g. the packed output layout -- so
    # look for the overflow warning specifically.)
    assert not any("overflow" in c.lower() for c in verdict.caveats)


def test_columns_axis_has_no_row_cap():
    """Column softmax reduces down rows with full-vector scalars -- no cap."""
    verdict = _down_columns(5000, 128)
    assert verdict.supported
    _construct(verdict)


def test_verdict_is_truthy_only_when_supported():
    assert _along_rows(8, 128)
    assert not _along_rows(8, 0)


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_sizes_unsupported(bad):
    assert not _along_rows(8, bad).supported
    assert not _along_rows(bad, 128).supported
    assert not _down_columns(8, bad).supported


def test_describe_reports_both_outcomes():
    assert "NOT SUPPORTED" in _along_rows(8, -1).describe()
    assert "SUPPORTED" in _along_rows(8, 128).describe()


# -- torch's dim convention ---------------------------------------------------

@pytest.mark.parametrize("dim,equiv", [(-1, 1), (-2, 0)])
def test_torch_negative_dims_count_from_the_end(dim, equiv):
    assert resolve("softmax", shape=(32, 300), dim=dim) == resolve("softmax", shape=(32, 300), dim=equiv)


@pytest.mark.parametrize("n", [16, 128, 300])
def test_torch_1d_shape_is_a_single_row(n):
    """A 1-D tensor is one row of n; dim=0 and dim=-1 are the same axis.

    Compared on the routing outcome rather than by ``==``: verdicts now carry
    the resolved shape bundle, which differs between the two spellings even
    when they select the same kernel with the same arguments.
    """
    one_d = resolve("softmax", shape=(n,), dim=0)
    row = _along_rows(1, n)
    assert (one_d.app_name, one_d.kwargs) == (row.app_name, row.kwargs)

    last = resolve("softmax", shape=(n,), dim=-1)
    assert (last.app_name, last.kwargs) == (one_d.app_name, one_d.kwargs)


def test_torch_verdicts_construct():
    for shape, dim in [((32, 300), 1), ((300, 32), 0), ((1000, 64), 1), ((64, 1000), 0)]:
        verdict = resolve("softmax", shape=shape, dim=dim)
        assert verdict.supported
        _construct(verdict)


@pytest.mark.parametrize("shape,dim", [
    ((2, 3, 4), -1),      # reduced axis last  -> (6, 4)
    ((2, 2, 2, 2), 0),    # reduced axis first -> (2, 8)
])
def test_higher_rank_is_flattened_and_disclosed(shape, dim):
    """Rank > 2 is a batch of 2-D problems, so it is flattened rather than
    refused -- and the reshape is stated in the verdict, never silent."""
    verdict = resolve("softmax", shape=shape, dim=dim)
    assert verdict.supported
    assert any("flattened" in n for n in verdict.shapes.notes)


def test_interior_reduction_axis_is_refused():
    """Flattening around a middle axis would require transposing the others,
    silently reinterpreting the caller's memory layout, so it is refused with
    an actionable message instead."""
    verdict = resolve("softmax", shape=(2, 3, 4), dim=1)
    assert not verdict.supported
    assert "interior axis" in verdict.reason


@pytest.mark.parametrize("dim", [2, -3, 99])
def test_torch_rejects_out_of_range_dim(dim):
    with pytest.raises(ValueError, match="out of range"):
        resolve("softmax", shape=(32, 300), dim=dim)


def test_torch_dim_choice_actually_changes_the_answer():
    """The shape is required, not just dim's size: the same tensor routes to
    different apps depending on which axis is reduced."""
    shape = (300, 16)
    assert resolve("softmax", shape=shape, dim=1).app_name == "softmax_rows_partial"
    assert resolve("softmax", shape=shape, dim=0).app_name == "softmax_columns_packed"


def test_torch_agrees_with_real_pytorch_semantics():
    """Pin the registry's dim convention to actual torch, not just to our own
    reading of it: for each shape/dim, the axis torch reduces (the one whose
    output sums to 1) must be the axis the verdict routed to.

    torch is not a project dependency (the apps' own tests reference numpy), so
    this is skipped when it isn't installed rather than adding a build dep.
    """
    torch = pytest.importorskip("torch")

    for shape, dim in [((32, 300), 1), ((300, 32), 0), ((8, 128), -1),
                       ((300, 16), 0), ((64, 1000), -2), ((129, 128), 1)]:
        x = torch.randn(*shape)
        y = torch.softmax(x, dim=dim)
        # The reduced axis is the one that sums to 1.
        assert torch.allclose(y.sum(dim=dim), torch.ones_like(y.sum(dim=dim)), atol=1e-5)

        verdict = resolve("softmax", shape=shape, dim=dim)
        assert verdict.supported
        reduced_is_last = (dim % 2) == 1
        rows_apps = {"softmax_rows", "softmax_rows_partial", "softmax_rows_long"}
        assert (verdict.app_name in rows_apps) == reduced_is_last, (
            f"{shape} dim={dim}: torch reduces "
            f"{'along rows' if reduced_is_last else 'down columns'} but the "
            f"router chose {verdict.app_name}"
        )


def test_case_width_must_be_declared():
    from ipu_apps.kernels.softmax.cases import random_case
    with pytest.raises(ValueError, match="width"):
        random_case(axis=0, defaults={"widht": 10}, max_cycles=100)


@pytest.mark.parametrize("app_cls,kwargs,message", [
    (SoftmaxColumnsApp, dict(width=200), "rows"),               # rows omitted
    (SoftmaxRowsPartialApp, dict(rows=10), "requires n="),       # own size omitted
    (SoftmaxRowsLongApp, dict(rows=4, n=300, width=300), "not width="),
    (SoftmaxColumnsPackedApp, dict(rows=4, n=16), "not n="),     # other axis's size
])
def test_constructors_require_their_sizes(app_cls, kwargs, message):
    """Defaulting a missing size would run the wrong problem without a word:
    128 rows of a 500-row file, silently truncating the output."""
    with pytest.raises(TypeError, match=message):
        app_cls(inst_path="x", input_path="y", output_path=None, **kwargs)
    # softmax_rows alone keeps its defaults: its row length is fixed by the .asm.
    assert SoftmaxRowsApp(inst_path="x", input_path="y", output_path=None).rows == 128


def test_input_file_of_the_wrong_size_is_refused_by_name(tmp_path):
    from ipu_emu.ipu_state import IpuState

    short = tmp_path / "short.bin"
    short.write_bytes(np.zeros(5 * 20, dtype="<f4").tobytes())   # 5 rows, not 10
    app = SoftmaxRowsPartialApp(inst_path="x", input_path=short, output_path=None, rows=10, n=20)
    with pytest.raises(ValueError, match="10 x 20 FP32 values"):
        app.setup(IpuState(wide_vector_debug=True))


# -- docs routing -------------------------------------------------------------

# Runfiles and a plain checkout put the docs in the same place relative to the
# repo root, so walk up until it appears.
_DOC = None
for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "docs/content/kernels/softmax.md"
    if _candidate.exists():
        _DOC = _candidate
        break

# `n 1..127   softmax_rows_partial`, `n = 128  softmax_rows`, `n 129..  <name>`
_ROW = re.compile(
    r"^(?P<param>n|width)\s+(?:=\s*(?P<exact>\d+)|(?P<start>\d+)\.\.(?P<end>\d+)?)"
    r"\s+(?P<kernel>\w+)\s*$"
)

# Which query each documented parameter describes: (dim, shape from the value).
_AXIS = {
    "n": (1, lambda v: (8, v)),
    "width": (0, lambda v: (8, v)),
}

# How far past an open-ended run ("n 129..") to probe. Any value well beyond
# the last boundary exercises the same claim.
_OPEN_ENDED_PROBE = 512


def _documented_runs() -> list[tuple[str, int, int | None, str]]:
    runs = []
    # Fences carry an optional info string (```python), so match that too --
    # pairing a closing fence with the next opening one finds nothing.
    fences = re.findall(r"^```[^\n]*\n(.*?)^```", _DOC.read_text(), re.S | re.M)
    for block in fences:
        for line in block.splitlines():
            match = _ROW.match(line.strip())
            if not match:
                continue
            param, kernel = match["param"], match["kernel"]
            if match["exact"]:
                value = int(match["exact"])
                runs.append((param, value, value, kernel))
            else:
                end = int(match["end"]) if match["end"] else None
                runs.append((param, int(match["start"]), end, kernel))
    return runs


@pytest.mark.skipif(_DOC is None, reason="docs/content/kernels/softmax.md is not in the runfiles")
def test_the_doc_actually_contains_routing_tables():
    """Guards the parser: a doc rewrite that drops the fences must not turn
    this file into a test that silently checks nothing."""
    runs = _documented_runs()
    assert len(runs) >= 5, f"parsed only {len(runs)} documented runs from {_DOC}"
    assert {param for param, *_ in runs} == {"n", "width"}


@pytest.mark.parametrize("param,start,end,kernel", _documented_runs() if _DOC else [])
def test_documented_routing_matches_the_registry(param, start, end, kernel):
    dim, build = _AXIS[param]
    probes = {start, end if end is not None else _OPEN_ENDED_PROBE}
    for value in probes:
        verdict = resolve("softmax", shape=build(value), dim=dim)
        assert verdict.supported, f"{param}={value}: {verdict.reason}"
        assert verdict.app_name == kernel, (
            f"docs say {param}={value} routes to {kernel}, registry answers "
            f"{verdict.app_name}. Re-probe the table in {_DOC.name}."
        )


@pytest.mark.parametrize("param,start,end,kernel", _documented_runs() if _DOC else [])
def test_documented_boundaries_are_real_boundaries(param, start, end, kernel):
    """A run that claims to start at ``start`` must actually change hands
    there -- otherwise the table describes a boundary the kernels do not have."""
    if start == 1:
        return  # nothing below it to differ from
    dim, build = _AXIS[param]
    below = resolve("softmax", shape=build(start - 1), dim=dim)
    assert below.app_name != kernel, (
        f"docs start a {kernel} run at {param}={start}, but {param}={start - 1} "
        f"routes there too, so the documented boundary is not where it says."
    )
