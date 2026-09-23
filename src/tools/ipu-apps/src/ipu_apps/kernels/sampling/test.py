"""Sampling family tests: numerics of both descriptor kernels, routing between
them, and memory/interface boundaries."""
import numpy as np
import pytest

from ipu_apps.kernel_registry import resolve
from ipu_apps.kernel_registry.cases import KernelCase, assemble_kernel, load_cases, run_case
from ipu_apps.kernel_registry.case_support import prepared_image
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_apps.kernels.sampling.cases import (
    pack_input, precompute_axis_weights, precompute_weights, prepare, reference_rows,
)
from ipu_apps.kernels.sampling.sample_descriptors.app import App, SPEC as DIRECT
from ipu_apps.kernels.sampling.sample_descriptors_separable.app import (
    App as SeparableApp, SPEC as SEPARABLE,
)

SPECS = (DIRECT, SEPARABLE)


@pytest.fixture(scope="module")
def instructions(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("descriptor-asm")
    return {spec.name: assemble_kernel(spec.name, workspace) for spec in SPECS}


def execute(tmp_path, instructions, values, mode="shared", weights=None, start=0, count=None,
            kernel="sample_descriptors"):
    params = dict(shape=values.shape, mode=mode, cell_row_start=start, cell_row_count=count)
    app = {spec.name: spec.app_class for spec in SPECS}[kernel]
    image, layout = pack_input(values, app, mode=mode, weights=weights,
                               cell_row_start=start, cell_row_count=count)
    case = KernelCase(lambda ws: prepared_image(ws, params, image, layout, lambda raw: None))
    run_case(kernel, case, workspace=tmp_path, inst_path=instructions[kernel])
    return np.fromfile(tmp_path / "output.bin", dtype='<f4').reshape(-1, values.shape[1] * 8, 256)


@pytest.mark.parametrize("mode,kernel", [("shared", "sample_descriptors"),
    ("stock", "sample_descriptors"), ("stock", "sample_descriptors_separable")])
@pytest.mark.parametrize("kind", ["zeros", "constant", "ramp", "impulse"])
def test_patterns(tmp_path, instructions, mode, kernel, kind):
    values = np.zeros((2, 2, 256), dtype='<f4')
    if kind == "constant":
        values[:] = np.linspace(-1, 1, 256)
    elif kind == "ramp":
        values[:] = (np.arange(2)[:, None, None] * .25
                     + np.arange(2)[None, :, None] * .125
                     + np.linspace(-.5, .5, 256)[None, None, :])
    elif kind == "impulse":
        values[1, 0, [0, 127, 128, 255]] = [1, -.5, .25, -.75]
    actual = execute(tmp_path, instructions, values, mode, kernel=kernel)
    np.testing.assert_allclose(actual, reference_rows(values, mode), atol=1e-5, rtol=1e-5)


def test_arbitrary_nine_taps(tmp_path, instructions):
    rng = np.random.default_rng(13)
    values = rng.uniform(-1, 1, (3, 3, 256)).astype('<f4')
    weights = rng.uniform(-.2, .2, (3, 3, 8, 8)).astype('<f4')
    actual = execute(tmp_path, instructions, values, weights=weights)
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)))
    expected = np.zeros_like(actual)
    for i in range(3):
        for j in range(3):
            for dy in range(3):
                for dx in range(3):
                    expected[i*8:(i+1)*8, j*8:(j+1)*8] += (
                        weights[dy, dx, :, :, None] * padded[i+dy, j+dx])
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["shared", "stock"])
def test_interior_band_keeps_neighbors_and_global_coordinates(tmp_path, instructions, mode):
    values = np.random.default_rng(9).uniform(-1, 1, (3, 2, 256)).astype('<f4')
    actual = execute(tmp_path, instructions, values, mode, start=1, count=1)
    np.testing.assert_allclose(actual, reference_rows(values, mode, 8, 8), atol=1e-5, rtol=1e-5)


def test_mapping_difference(tmp_path, instructions):
    values = np.zeros((2, 2, 256), dtype='<f4')
    values[1, 1] = np.linspace(.1, 1, 256)
    shared = execute(tmp_path, instructions, values)
    stock = execute(tmp_path, instructions, values, mode="stock")
    assert np.max(np.abs(shared - stock)) > .1
    for actual, mode in ((shared, "shared"), (stock, "stock")):
        np.testing.assert_allclose(actual, reference_rows(values, mode), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode,app", [("shared", App), ("stock", App), ("stock", SeparableApp)])
def test_full_shape_fits_single_launch(mode, app):
    layout = app.memory_layout(shape=(60, 80, 256), mode=mode)
    assert layout.output_rows * 512 == 300 * 1024**2
    assert (layout.input_rows + layout.output_rows) * 512 <= XMEM_SIZE_BYTES
    assert resolve("sample_descriptors", shape=(60, 80, 256), mode=mode)


@pytest.mark.parametrize("params", [
    dict(shape=(60, 80, 128)), dict(shape=(0, 80, 256)),
    dict(shape=(60, 80, 256), mode="unknown"),
    dict(shape=(60, 80, 256), cell_row_start=-1),
    dict(shape=(60, 80, 256), cell_row_count=61),
    dict(shape=(60, 80, 256), cell_row_count=0),
    dict(shape=(120, 160, 256)),
])
def test_invalid_geometry(params):
    assert not resolve("sample_descriptors", **params)
    with pytest.raises(ValueError):
        App.memory_layout(**params)


def test_invalid_weights():
    values = np.zeros((1, 1, 256), dtype='<f4')
    with pytest.raises(ValueError, match="shape"):
        pack_input(values, App, weights=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        pack_input(values, App, weights=np.full((3, 3, 8, 8), np.nan))
    with pytest.raises(ValueError, match="shared mode"):
        pack_input(values, App, mode="stock", weights=precompute_weights())


@pytest.mark.parametrize("amplitude", [0, -1, np.nan, np.inf])
def test_invalid_case_amplitude(tmp_path, amplitude):
    with pytest.raises(ValueError, match="amplitude"):
        prepare(App, tmp_path, amplitude=amplitude)


def test_case_exports_output_and_reports_errors(tmp_path, capsys):
    case = load_cases("sample_descriptors")["default"]
    output = tmp_path / "dense.bin"
    _, cycles = run_case("sample_descriptors", case,
                         options=dict(amplitude=100.0, compare_stock=True), output_path=output)
    assert cycles > 0
    assert output.stat().st_size == 16 * 24 * 256 * 4
    values = np.random.default_rng(31).uniform(-100, 100, (2, 3, 256)).astype('<f4')
    actual = np.fromfile(output, dtype='<f4').reshape(16, 24, 256)
    difference = actual.astype(np.float64) - reference_rows(values, "stock")
    report = capsys.readouterr().out
    assert "uniform[-100,100]" in report
    assert f"max_abs_error={np.max(np.abs(difference)):.9g}" in report
    assert f"mean_abs_error={np.mean(np.abs(difference)):.9g}" in report
    assert f"rmse={np.sqrt(np.mean(difference ** 2)):.9g}" in report


@pytest.mark.parametrize("kernel", [s.name for s in SPECS])
def test_stock_matches_torch(tmp_path, instructions, kernel):
    torch = pytest.importorskip("torch")
    values = np.random.default_rng(23).uniform(-1, 1, (2, 3, 256)).astype('<f4')
    y, x = torch.meshgrid(torch.arange(16), torch.arange(24), indexing="ij")
    grid = torch.stack((x, y), dim=-1).float()
    grid -= 3.5
    grid /= torch.tensor([24 - 4.5, 16 - 4.5])
    grid = grid * 2 - 1
    expected = torch.nn.functional.grid_sample(
        torch.from_numpy(values.transpose(2, 0, 1)[None]), grid[None],
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(execute(tmp_path, instructions, values, "stock", kernel=kernel), expected,
                               atol=1e-5, rtol=1e-5)


def test_measured_cycle_improvement(tmp_path):
    _, direct_cycles = run_case("sample_descriptors", load_cases("sample_descriptors")["stock"],
                                workspace=tmp_path / "direct")
    _, separable_cycles = run_case("sample_descriptors_separable",
                                   load_cases("sample_descriptors_separable")["default"],
                                   workspace=tmp_path / "separable")
    np.testing.assert_allclose(np.fromfile(tmp_path / "separable/output.bin", dtype='<f4'),
                               np.fromfile(tmp_path / "direct/output.bin", dtype='<f4'),
                               atol=1e-5, rtol=1e-5)
    assert separable_cycles < direct_cycles / 2


def test_routing_preserves_shared_and_optimizes_stock():
    assert resolve("sample_descriptors", shape=(60, 80, 256)).app_name == "sample_descriptors"
    assert resolve("sample_descriptors", shape=(60, 80, 256), mode="shared").app_name == "sample_descriptors"
    assert resolve("sample_descriptors", shape=(60, 80, 256), mode="stock").app_name == "sample_descriptors_separable"


def test_compact_tables_and_memory():
    shape = (60, 80, 256)
    ky, kx = precompute_axis_weights(shape)
    assert ky.shape == (60, 3, 8) and kx.shape == (80, 3, 8)
    assert ky.nbytes + kx.nbytes == 13440
    # Compare the compact coefficients with the original, expanded contract.
    np.testing.assert_array_equal(
        ky[:, None, :, None, :, None] * kx[None, :, None, :, None, :],
        precompute_weights(shape, "stock"))
    image, layout = pack_input(np.zeros(shape, dtype="<f4"), SeparableApp, mode="stock")
    assert (layout.crs[6] - (62 * 82 * 2)) * 512 == 70 * 1024
    assert (layout.input_rows - layout.crs[6]) * 512 == 656 * 1024
    assert layout.output_rows == App.memory_layout(shape=shape, mode="stock").output_rows
    assert layout.input_rows < App.memory_layout(shape=shape, mode="stock").input_rows
    assert np.count_nonzero(image[layout.crs[6]:]) == 0


@pytest.mark.parametrize("params", [
    dict(shape=(2, 3, 128), mode="stock"), dict(shape=(2, 3, 256), mode="shared"),
    dict(shape=(2, 3, 256), mode="stock", cell_row_start=-1),
    dict(shape=(2, 3, 256), mode="stock", cell_row_count=0),
    dict(shape=(120, 160, 256), mode="stock"),
])
def test_invalid_separable_geometry(params):
    with pytest.raises(ValueError):
        SeparableApp.memory_layout(**params)


def test_invalid_packing():
    values = np.zeros((1, 1, 256), dtype='<f4')
    with pytest.raises(ValueError, match="stock mode"):
        pack_input(values, SeparableApp, mode="shared")
    with pytest.raises(ValueError, match="shared mode"):
        pack_input(values, SeparableApp, mode="stock", weights=np.ones((3, 3, 8, 8)))
