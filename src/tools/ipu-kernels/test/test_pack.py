"""Packing: per-format bytes, the shared transforms and their lookup (E06-05)."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ipu_apps.kernels.projections.app import ProjectionLayout, ProjectionP4App
from ipu_harness._codec import held
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.packing import pack, stored_rows, transform_for, unpack
from ipu_harness.params import PlanError, plan_case
from ipu_harness.placement import default_rows, place
from ipu_isa.formats import ElementFormat as F

KERNELS = Path(__file__).resolve().parents[1] / "src" / "ipu_kernels"
HARNESS = Path(__file__).resolve().parents[2] / "ipu-harness" / "src" / "ipu_harness"

TOML = """\
[kernel]
op = "pack_test"
formats = ["fp32", "int8", "e4", "e5"]

[params]
c = {{ type = "int", default = 3 }}
h = {{ type = "int", default = 5 }}
w = {{ type = "int", default = 32 }}

[inputs.x]
shape = ["c", "h", "w"]
{pack}
[outputs.y]
shape = ["c", "h", "w"]
{store}
[abi]
x = "CR2"
y = "CR3"

[cases.default]
"""


def manifest(tmp_path: Path, *, pack: str = "", store: str = "", name: str = "k") -> KernelManifest:
    directory = tmp_path / "family" / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(TOML.format(pack=pack, store=store))
    return load_manifest(directory)


def values(fmt: F, shape: tuple[int, ...]) -> np.ndarray:
    stream = np.random.default_rng(5)
    if fmt is F.INT8:
        return stream.integers(-100, 100, size=shape).astype(np.float64)
    return stream.uniform(-1, 1, size=shape)


@pytest.mark.parametrize("fmt", [F.FP32, F.INT8, F.E4, F.E5])
@pytest.mark.parametrize(
    "name", [None, "channel_major", "output_major_weights", "chunk_interleaved"]
)
def test_unpack_of_pack_is_the_tensor_rounded_to_the_format(tmp_path, fmt, name):
    m = manifest(tmp_path, pack=f'pack = "{name}"\n' if name else "")
    plan = plan_case(m, {}, fmt)
    x = values(fmt, (3, 5, 32))
    raw = pack(m.tensors["x"], x, plan)
    assert np.array_equal(unpack(m.tensors["x"], raw, plan), held(x, fmt).astype(np.float64))


@pytest.mark.parametrize(
    "fmt, value, first",
    [
        (F.FP32, 1.0, bytes.fromhex("0000803f")),  # IEEE 754, little-endian
        (F.INT8, -1.0, b"\xff"),  # two's complement
        (F.INT8, 1.5, b"\x02"),  # to nearest, ties to even
        (F.INT8, 300.0, b"\x7f"),  # saturates
        (F.E4, 1.0, b"\x38"),  # 0 0111 000: exponent 7 = bias
        (F.E4, 1e6, b"\x77"),  # saturates to 240 = 0 1110 111
        (F.E5, 1.0, b"\x3c"),  # 0 01111 00: exponent 15 = bias
    ],
)
def test_element_bytes_by_hand(tmp_path, fmt, value, first):
    m = manifest(tmp_path)
    plan = plan_case(m, {"c": 1, "h": 1, "w": 1}, fmt)
    raw = pack(m.tensors["x"], np.full((1, 1, 1), value), plan)
    assert raw[: len(first)] == first and raw[len(first) :] == bytes(len(raw) - len(first))


def test_a_row_major_width_over_128_takes_several_zero_padded_rows(tmp_path):
    m = manifest(tmp_path)
    plan = plan_case(m, {"c": 1, "h": 1, "w": 200}, F.INT8)
    x = np.arange(200, dtype=np.float64).reshape(1, 1, 200) % 100
    raw = pack(m.tensors["x"], x, plan)
    assert len(raw) == 256 and raw[200:] == bytes(56)
    assert np.array_equal(unpack(m.tensors["x"], raw, plan), x)


@pytest.mark.parametrize(
    "name, params, message",
    [
        ("channel_major", {"w": 130}, r"the width \(130\) must fit one row"),
        ("chunk_interleaved", {"w": 48}, r"cols \(48\) must divide 128"),
    ],
)
def test_a_shape_a_transform_cannot_lay_out_is_located_at_pack(tmp_path, name, params, message):
    m = manifest(tmp_path, pack=f'pack = "{name}"\n')
    plan = plan_case(m, params, F.FP32)
    shape = plan.shapes["x"]
    for call in (
        lambda: pack(m.tensors["x"], np.zeros(shape), plan),
        lambda: stored_rows(m.tensors["x"], plan),
    ):
        with pytest.raises(PlanError, match=message) as error:
            call()
        assert error.value.key == "inputs.x.pack"


def test_stored_rows_are_placements_row_counts(tmp_path):
    m = manifest(tmp_path, store='store = "accumulator"\n')
    for fmt in (F.FP32, F.INT8, F.E4):
        plan = plan_case(m, {"c": 1, "h": 2, "w": 128}, fmt)
        for name in ("x", "y"):
            assert stored_rows(m.tensors[name], plan) == default_rows(plan, m.tensors[name])
    int8 = plan_case(m, {"c": 1, "h": 2, "w": 128}, F.INT8)
    assert stored_rows(m.tensors["y"], int8) == 8  # 2 vectors of 128 INT32: 4 narrow rows each


def test_packed_sizes_follow_the_format(tmp_path):
    m = manifest(tmp_path)
    x = np.zeros((3, 5, 32))
    # Row-major: 3 x 5 rows of 128 lanes; 1 byte an element in INT8, 4 in FP32.
    assert len(pack(m.tensors["x"], x, plan_case(m, {}, F.INT8))) == 15 * 128
    assert len(pack(m.tensors["x"], x, plan_case(m, {}, F.FP32))) == 15 * 512
    two = plan_case(m, {"c": 1, "h": 2, "w": 128}, F.INT8)
    assert len(pack(m.tensors["x"], np.zeros((1, 2, 128)), two)) == 256
    two32 = plan_case(m, {"c": 1, "h": 2, "w": 128}, F.FP32)
    assert len(pack(m.tensors["x"], np.zeros((1, 2, 128)), two32)) == 1024


def test_chunk_interleaved_matches_the_convolutions_packing(tmp_path):
    conv = importlib.import_module("ipu_apps.kernels.convolutions.app")
    m = manifest(tmp_path, pack='pack = "chunk_interleaved"\n')
    plan = plan_case(m, {"c": 3, "h": 7, "w": 32}, F.FP32)
    x = values(F.FP32, (3, 7, 32)).astype(np.float32)
    assert pack(m.tensors["x"], x, plan) == conv.pack_input_chunked(x, 32)


def test_an_accumulator_output_is_read_in_the_accumulator_format(tmp_path):
    m = manifest(tmp_path, store='store = "accumulator"\n')
    plan = plan_case(m, {"c": 1, "h": 2, "w": 128}, F.INT8)
    y = np.arange(256, dtype=np.int32).reshape(1, 2, 128) * 1000  # past INT8: INT32 holds it
    raw = y.astype("<i4").tobytes()  # 2 vectors x 128 x 4 B = 8 narrow rows
    assert np.array_equal(unpack(m.tensors["y"], raw, plan), y)
    with pytest.raises(PlanError, match="y's region holds 1024 bytes in int32, given 256") as error:
        unpack(m.tensors["y"], raw[:256], plan)
    assert error.value.key == "outputs.y.size"


@pytest.mark.parametrize("shape", [(3, 5, 31), (3, 5, 33), (3, 6, 32), (15, 32)])
def test_a_wrong_shape_oversize_included_is_rejected(tmp_path, shape):
    m = manifest(tmp_path)
    with pytest.raises(
        PlanError, match=re.escape(f"x has shape {shape}; the case's is (3, 5, 32)")
    ):
        pack(m.tensors["x"], np.zeros(shape), plan_case(m, {}, F.FP32))


def test_a_kernel_pack_py_overrides_a_shared_transform_of_its_name(tmp_path):
    m = manifest(tmp_path, pack='pack = "channel_major"\n')
    (m.directory / "pack.py").write_text(
        "import numpy as np\n\n\nclass Flat:\n"
        "    def rows(self, shape):\n        return 15\n\n"
        "    def to_rows(self, a, plan):\n        return np.zeros((15, 128))\n\n"
        "    def from_rows(self, r, plan, shape):\n        return np.zeros(shape)\n\n\n"
        "channel_major = Flat()\n"
    )
    plan = plan_case(m, {}, F.FP32)
    assert type(transform_for(m.tensors["x"], plan)).__name__ == "Flat"
    assert pack(m.tensors["x"], np.ones((3, 5, 32)), plan) == bytes(15 * 512)


FAMILY_PACK = """\
from ipu_harness.packing import RowMajor


class Family(RowMajor):
    pass


channel_major = Family()
"""


def test_a_family_transform_overrides_a_shared_one_of_its_name(tmp_path, monkeypatch):
    base = tmp_path / "site" / "fake_kernels"
    for sub in ("", "family", "pack"):
        (base / sub).mkdir(parents=True, exist_ok=True)
        (base / sub / "__init__.py").write_text("")
    (base / "family" / "pack.py").write_text(FAMILY_PACK)
    (base / "pack" / "__init__.py").write_text(
        "from ipu_kernels.pack import channel_major, chunk_interleaved\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    plan = plan_case(manifest(tmp_path, pack='pack = "channel_major"\n'), {}, F.FP32)
    m = plan.manifest
    assert type(transform_for(m.tensors["x"], plan, package="fake_kernels")).__name__ == "Family"
    other = manifest(tmp_path, pack='pack = "chunk_interleaved"\n', name="k2")
    plan2 = plan_case(other, {}, F.FP32)
    found = transform_for(other.tensors["x"], plan2, package="fake_kernels")
    assert type(found).__name__ == "ChunkInterleaved"  # not in the family: the shared one


def test_a_name_that_is_no_transform_or_a_broken_pack_py_is_located(tmp_path):
    m = manifest(tmp_path, pack='pack = "channel_major"\n')
    (m.directory / "pack.py").write_text("channel_major = 3\n")
    with pytest.raises(PlanError, match="defines channel_major, but not as a transform") as error:
        transform_for(m.tensors["x"], plan_case(m, {}, F.FP32))
    assert error.value.key == "inputs.x.pack"
    (m.directory / "pack.py").write_text("def broken(:\n")
    with pytest.raises(PlanError, match="SyntaxError"):
        transform_for(m.tensors["x"], plan_case(m, {}, F.FP32))


def test_an_unknown_transform_is_a_located_error(tmp_path):
    m = manifest(tmp_path, pack='pack = "nowhere"\n')
    with pytest.raises(PlanError, match="no transform 'nowhere'") as error:
        transform_for(m.tensors["x"], plan_case(m, {}, F.FP32))
    assert error.value.key == "inputs.x.pack"


# --- the projections, byte for byte (golden for E07-02) -----------------------------------

PROJECTION = """\
[kernel]
op = "projection"
formats = ["fp32"]

[params]
k     = {{ type = "int", default = {k} }}
n_out = {{ type = "int", default = {n_out} }}
n_tg  = {{ type = "int", default = {n_tg} }}
n_tok = {{ type = "int", default = {n_tok} }}

[inputs.data]
shape = [4, "n_tg", "k", "n_tok"]
pack = "channel_major"

[inputs.weights]
shape = ["n_out", "k"]
pack = "output_major_weights"

[outputs.out]
shape = [4, "n_tg", "n_out", 128]

[abi]
data    = "CR2"
weights = "CR3"
out     = "CR4"

[cases.default]
"""

PROJECTIONS = [
    f"proj_{kind}_{width}_p4"
    for kind in ("qkv", "outproj", "ffn1", "ffn2")
    for width in (144, 192, 240)
]


class _Xmem:
    def __init__(self) -> None:
        self.written: dict[int, bytes] = {}

    def write_address(self, address: int, data: bytes) -> None:
        self.written[address] = bytes(data)


@pytest.mark.parametrize("name", PROJECTIONS)
def test_projection_packing_is_byte_identical_to_the_loaders(tmp_path, name):
    app = importlib.import_module(f"ipu_apps.kernels.projections.{name}.app")
    layout = ProjectionLayout(k=app.K, n_out=app.N_OUT, n_tok=app.N_TOK, n_tg=app.N_TG)
    data = values(F.FP32, layout.input_shape).astype(np.float32)
    weights = values(F.FP32, (layout.n_out, layout.k)).astype(np.float32)
    (tmp_path / "d.bin").write_bytes(data.tobytes())
    (tmp_path / "w.bin").write_bytes(weights.tobytes())
    harness = SimpleNamespace(
        layout=layout, input_path=tmp_path / "d.bin", weights_path=tmp_path / "w.bin"
    )
    state = SimpleNamespace(xmem=_Xmem())
    ProjectionP4App._load_data(harness, state)
    ProjectionP4App._load_weights(harness, state)

    directory = tmp_path / "projections" / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(
        PROJECTION.format(k=layout.k, n_out=layout.n_out, n_tg=layout.n_tg, n_tok=layout.n_tok)
    )
    m = load_manifest(directory)
    plan = plan_case(m, {}, F.FP32)
    row = 512
    assert pack(m.tensors["data"], data, plan) == state.xmem.written[layout.data_base_row * row]
    assert (
        pack(m.tensors["weights"], weights, plan)
        == state.xmem.written[layout.weights_base_row * row]
    )
    # Placed with packing's row counts and the logical shapes, the bases are today's.
    placement = place(plan, {name: stored_rows(t, plan) for name, t in m.tensors.items()})
    assert placement.region("data").base_row == layout.data_base_row
    assert placement.region("weights").base_row == layout.weights_base_row
    assert placement.region("out").base_row == layout.output_base_row


# --- what the acceptance criteria forbid ----------------------------------------------------


def test_no_lane_or_row_size_is_defined_outside_ipu_isa():
    defined = re.compile(r"^(LANES|ROW_BYTES|ELEM_BYTES)\s*[:=]", re.MULTILINE)
    for root in (HARNESS, KERNELS):
        for path in sorted(root.rglob("*.py")):
            assert not defined.search(path.read_text(encoding="utf-8")), path


def test_packing_has_no_format_branches():
    text = (HARNESS / "packing.py").read_text(encoding="utf-8")
    assert not re.search(r"\bif\b[^\n]*\b(fmt|format)\b\s*(==|is|in)\b", text)
