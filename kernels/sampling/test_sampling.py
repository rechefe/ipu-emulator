"""The sampling family's tests (E07-15): the ``ipu_apps`` ``sampling/test.py``
numerics, routing and memory checks, on the kernel tree.

Each run is held to ``ops.sample_descriptors`` (the explicit-indexing
bilinear reference) by the runner; a test here adds what the case x format
tests do not: hand-made patterns, the band's global coordinates, the two
mappings' difference, PyTorch's ``grid_sample``, caller weights in the
table, the cycle saving and the memory footprint.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import expand
from ipu_harness.packing import stored_rows
from ipu_harness.params import PlanError, plan_case
from ipu_harness.placement import place
from ipu_harness.registry import Registry, kernel_tree
from ipu_harness.runner import RunOptions, run_kernel, stage
from ipu_isa.formats import FORMATS, ElementFormat

FP32 = ElementFormat.FP32
ROW_BYTES = FORMATS[FP32].row_bytes
CACHE = AssemblyCache()
DIRECT, SEPARABLE = "sample_descriptors", "sample_descriptors_separable"


@pytest.fixture(scope="module")
def registry():
    loaded = Registry.load(kernel_tree(), families=["sampling"])
    assert not loaded.invalid, loaded.invalid
    return loaded


def _case(kernel, **values):
    base = expand(kernel)[0]
    params = {"channels": 256, "stock": kernel.name == SEPARABLE, **values}
    return dataclasses.replace(base, name="test", values=params)


def _run(registry, tmp_path, name, x, **values):
    """``x`` through kernel ``name``, held to the reference; the output."""
    kernel = registry.kernel(name)
    h, w, _ = x.shape
    path = tmp_path / f"{name}-{len(list(tmp_path.iterdir()))}.npy"
    np.save(path, x)
    case = _case(kernel, height=h, width=w, **values)
    result = run_kernel(kernel, case, FP32, RunOptions(inputs={"x": path}), cache=CACHE)
    assert result.passed, result.describe()
    return np.asarray(result.outputs["y"]), result


MODES = [(False, DIRECT), (True, DIRECT), (True, SEPARABLE)]


@pytest.mark.parametrize(("stock", "name"), MODES, ids=["shared", "stock", "separable"])
@pytest.mark.parametrize("kind", ["zeros", "constant", "ramp", "impulse"])
def test_patterns(registry, tmp_path, stock, name, kind):
    values = np.zeros((2, 2, 256), dtype="<f4")
    if kind == "constant":
        values[:] = np.linspace(-1, 1, 256)
    elif kind == "ramp":
        values[:] = (
            np.arange(2)[:, None, None] * 0.25
            + np.arange(2)[None, :, None] * 0.125
            + np.linspace(-0.5, 0.5, 256)[None, None, :]
        )
    elif kind == "impulse":
        values[1, 0, [0, 127, 128, 255]] = [1, -0.5, 0.25, -0.75]
    y, _ = _run(registry, tmp_path, name, values, stock=stock)
    assert y.shape == (16, 16, 256)


@pytest.mark.parametrize("stock", [False, True], ids=["shared", "stock"])
def test_interior_band_keeps_neighbors_and_global_coordinates(registry, tmp_path, stock):
    """Coarse row 1 of 3: its fine rows 8..15, sampled with rows 0 and 2 as
    neighbours (held to the reference over the whole map)."""
    values = np.random.default_rng(9).uniform(-1, 1, (3, 2, 256)).astype("<f4")
    y, _ = _run(registry, tmp_path, DIRECT, values, stock=stock, cell_row_start=1, cell_row_count=1)
    from ipu_kernels.reference.ops import sample_descriptors

    torch = pytest.importorskip("torch")
    whole = sample_descriptors(
        torch.from_numpy(values.astype(np.float64)), stock=stock, cell_row_start=0, cell_row_count=0
    )["y"].numpy()
    np.testing.assert_allclose(y, whole[8:16], atol=1e-5, rtol=1e-5)


def test_mapping_difference(registry, tmp_path):
    values = np.zeros((2, 2, 256), dtype="<f4")
    values[1, 1] = np.linspace(0.1, 1, 256)
    shared, _ = _run(registry, tmp_path, DIRECT, values, stock=False)
    stock, _ = _run(registry, tmp_path, DIRECT, values, stock=True)
    assert np.max(np.abs(shared - stock)) > 0.1


@pytest.mark.parametrize("name", [DIRECT, SEPARABLE])
def test_stock_matches_torch(registry, tmp_path, name):
    torch = pytest.importorskip("torch")
    values = np.random.default_rng(23).uniform(-1, 1, (2, 3, 256)).astype("<f4")
    y, x = torch.meshgrid(torch.arange(16), torch.arange(24), indexing="ij")
    grid = torch.stack((x, y), dim=-1).float()
    grid -= 3.5
    grid /= torch.tensor([24 - 4.5, 16 - 4.5])
    grid = grid * 2 - 1
    expected = (
        torch.nn.functional.grid_sample(
            torch.from_numpy(values.transpose(2, 0, 1)[None]),
            grid[None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )
    got, _ = _run(registry, tmp_path, name, values, stock=True)
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_arbitrary_nine_taps(registry):
    """Every one of the nine coefficients is read: caller weights written into
    the shared table give their own 3x3 filter."""
    kernel = registry.kernel(DIRECT)
    rng = np.random.default_rng(13)
    values = rng.uniform(-1, 1, (3, 3, 256)).astype("<f4")
    weights = rng.uniform(-0.2, 0.2, (3, 3, 8, 8)).astype("<f4")
    case = _case(kernel, height=3, width=3)
    staged = stage(kernel, case, FP32, cache=CACHE)
    x_start, _ = staged.region_bytes("x")
    padded = np.zeros((5, 5, 256), dtype="<f4")
    padded[1:-1, 1:-1] = values
    staged.model.write_xmem(x_start, padded.tobytes())
    table = np.zeros((9, 128), dtype="<f4")
    table[:, :64] = weights.reshape(9, 64)
    staged.model.write_xmem(staged.region_bytes("k")[0], table.tobytes())
    staged.model.run()
    start, size = staged.region_bytes("y")
    got = np.frombuffer(staged.model.read_xmem(start, size), dtype="<f4").reshape(24, 24, 256)
    expected = np.zeros_like(got)
    for i in range(3):
        for j in range(3):
            for dy in range(3):
                for dx in range(3):
                    expected[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] += (
                        weights[dy, dx, :, :, None] * padded[i + dy, j + dx]
                    )
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_the_separable_kernel_is_more_than_twice_as_fast(registry):
    direct = registry.kernel(DIRECT)
    separable = registry.kernel(SEPARABLE)
    fast = run_kernel(separable, _case(separable, height=2, width=3), FP32, cache=CACHE)
    slow = run_kernel(direct, _case(direct, height=2, width=3, stock=True), FP32, cache=CACHE)
    assert fast.passed and slow.passed
    np.testing.assert_allclose(fast.outputs["y"], slow.outputs["y"], atol=1e-5, rtol=1e-5)
    assert fast.cycles < slow.cycles / 2


def _rows(kernel, **values):
    """Each region's (base row, rows) for a case, without running it."""
    plan = plan_case(kernel, {"channels": 256, **values}, FP32)
    tensors = plan.manifest.tensors
    placement = place(plan, {n: stored_rows(t, plan) for n, t in tensors.items()})
    return {n: (placement.region(n).base_row, placement.region(n).rows) for n in tensors}


@pytest.mark.parametrize(("stock", "name"), MODES, ids=["shared", "stock", "separable"])
def test_full_shape_fits_one_launch(registry, stock, name):
    """SuperPoint's 60 x 80 map: 300 MiB of output in one launch."""
    assert registry.resolve("sample_descriptors", height=60, width=80, channels=256, stock=stock)
    rows = _rows(registry.kernel(name), height=60, width=80, stock=stock)
    assert rows["y"][1] * ROW_BYTES == 300 * 1024**2


def test_the_separable_tables_are_compact(registry):
    tables = _load_tables()
    direct = _rows(registry.kernel(DIRECT), height=60, width=80, stock=True)
    rows = _rows(registry.kernel(SEPARABLE), height=60, width=80, stock=True)
    assert (rows["ky"][1] + rows["kx"][1]) * ROW_BYTES == 70 * 1024
    assert rows["t"][1] * ROW_BYTES == 656 * 1024
    assert rows["y"][1] == direct["y"][1]
    assert rows["y"][0] < direct["y"][0]  # less XMEM before the output
    plan = plan_case(
        registry.kernel(SEPARABLE),
        {"height": 60, "width": 80, "channels": 256, "stock": True},
        FP32,
    )
    ky = np.frombuffer(tables.ky(plan), dtype="<f4").reshape(60, 128)[:, :24].reshape(60, 3, 8)
    kx = np.frombuffer(tables.kx(plan), dtype="<f4").reshape(80, 128)[:, :24].reshape(80, 3, 8)
    assert ky.nbytes + kx.nbytes == 13440
    full = np.frombuffer(tables.stock(plan), dtype="<f4").reshape(60, 80, 3, 3, 128)[..., :64]
    np.testing.assert_array_equal(
        ky[:, None, :, None, :, None] * kx[None, :, None, :, None, :],
        full.reshape(60, 80, 3, 3, 8, 8),
    )


def _load_tables():
    path = Path(__file__).with_name("tables.py")
    spec = importlib.util.spec_from_file_location("sampling_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_routing_keeps_shared_and_prefers_separable_for_stock(registry):
    query = dict(height=60, width=80, channels=256)
    assert registry.resolve("sample_descriptors", **query).kernel_name == DIRECT
    assert registry.resolve("sample_descriptors", **query, stock=False).kernel_name == DIRECT
    assert registry.resolve("sample_descriptors", **query, stock=True).kernel_name == SEPARABLE


@pytest.mark.parametrize(
    "params",
    [
        dict(height=60, width=80, channels=128),
        dict(height=0, width=80, channels=256),
        dict(height=60, width=80, channels=256, cell_row_start=-1),
        dict(height=60, width=80, channels=256, cell_row_count=61),
        dict(height=60, width=80, channels=256, cell_row_start=60),
        dict(height=60, width=80, channels=256, cell_row_start=59, cell_row_count=2),
        dict(height=2, width=3, channels=256, amplitude=0.0),
        dict(height=2, width=3, channels=256, amplitude=-1.0),
    ],
)
def test_invalid_geometry_is_refused(registry, params):
    verdict = registry.resolve("sample_descriptors", **params)
    assert not verdict, verdict.describe()
    stock = registry.resolve("sample_descriptors", **params, stock=True)
    assert not stock, stock.describe()


@pytest.mark.parametrize(("stock", "name"), MODES, ids=["shared", "stock", "separable"])
def test_a_map_too_large_for_xmem_is_refused_when_placed(registry, stock, name):
    """Routing reads no placement (E06-16); the case is refused when placed."""
    with pytest.raises(PlanError):
        _rows(registry.kernel(name), height=120, width=160, stock=stock)
