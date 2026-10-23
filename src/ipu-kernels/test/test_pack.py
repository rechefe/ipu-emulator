"""Packing: per-variant bytes, the shared transforms and their lookup (E06-05)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.packing import held, pack, stored_rows, transform_for, unpack
from ipu_harness.params import PlanError, plan_case
from ipu_harness.placement import default_rows, place
from ipu_harness.variants import variant
from ipu_isa.formats import ElementFormat as F

KERNELS = Path(__file__).resolve().parents[1] / "src" / "ipu_kernels"
HARNESS = Path(__file__).resolve().parents[2] / "ipu-harness" / "src" / "ipu_harness"

TOML = """\
[kernel]
op = "pack_test"
formats = ["fp32", "e4", "e5"]

[params]
c = {{ type = "int", default = 3 }}
h = {{ type = "int", default = 5 }}
w = {{ type = "int", default = 32 }}

[inputs.x]
shape = ["c", "h", "w"]
{pack}
[outputs.y]
shape = ["c", "h", "w"]

[abi]
x = "CR2"
y = "CR3"

[cases.default]
"""

#: The variants the round trips run in: FP32 mode and two NX8 element types.
VARIANTS = ["fp32", "e4", "e5"]


def manifest(tmp_path: Path, *, pack: str = "", name: str = "k") -> KernelManifest:
    directory = tmp_path / "family" / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(TOML.format(pack=pack))
    return load_manifest(directory)


def values(fmt: object, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(5).uniform(-1, 1, size=shape)


@pytest.mark.parametrize("name_", VARIANTS)
@pytest.mark.parametrize(
    "name", [None, "channel_major", "output_major_weights", "chunk_interleaved"]
)
def test_unpack_of_pack_is_the_tensor_as_the_variant_holds_it(tmp_path, name_, name):
    m = manifest(tmp_path, pack=f'pack = "{name}"\n' if name else "")
    plan = plan_case(m, {}, variant(name_))
    x = values(None, (3, 5, 32))
    packed = pack(m.tensors["x"], x, plan)
    back = unpack(m.tensors["x"], packed.data, plan, packed.metadata)
    assert np.array_equal(back, held(m.tensors["x"], x, plan))
    if name_ == "fp32":
        assert packed.metadata is None
        assert np.array_equal(back, x.astype(np.float32).astype(np.float64))
    else:
        # NX8: each row quantized with its own scale; what a row holds packs
        # to the same bytes again, and is within a step of the input.
        assert packed.metadata is not None
        again = pack(m.tensors["x"], back, plan)
        assert (again.data, again.metadata) == (packed.data, packed.metadata)
        np.testing.assert_allclose(back, x, rtol=0.25, atol=2.0**-6)


@pytest.mark.parametrize(
    "name_, value, first, metadata",
    [
        ("fp32", 1.0, bytes.fromhex("0000803f"), None),  # IEEE 754, little-endian
        ("fp32", -2.5, bytes.fromhex("000020c0"), None),
    ],
)
def test_element_bytes_by_hand(tmp_path, name_, value, first, metadata):
    m = manifest(tmp_path)
    plan = plan_case(m, {"c": 1, "h": 1, "w": 1}, variant(name_))
    packed = pack(m.tensors["x"], np.full((1, 1, 1), value), plan)
    raw = packed.data
    assert raw[: len(first)] == first and raw[len(first) :] == bytes(len(raw) - len(first))
    assert packed.metadata == metadata


def test_an_nx8_row_holds_one_byte_an_element_and_its_scale(tmp_path):
    # e4, a row of 1.0 then -1.0: the row's scale is 2^0 (127, bias 127) and
    # each element is 0 exponent steps below it, so only the sign differs.
    m = manifest(tmp_path)
    plan = plan_case(m, {"c": 1, "h": 1, "w": 128}, variant("e4"))
    x = np.ones((1, 1, 128))
    x[..., 64:] = -1.0
    packed = pack(m.tensors["x"], x, plan)
    assert len(packed.data) == 128 and packed.metadata == (127,)
    assert len(set(packed.data[:64])) == 1 and len(set(packed.data[64:])) == 1
    assert packed.data[0] ^ packed.data[64] == 0x80  # the sign bit
    assert np.array_equal(unpack(m.tensors["x"], packed.data, plan, packed.metadata), x)


def test_a_row_major_width_over_128_takes_several_zero_padded_rows(tmp_path):
    m = manifest(tmp_path)
    plan = plan_case(m, {"c": 1, "h": 1, "w": 200}, F.FP32)
    x = np.arange(200, dtype=np.float64).reshape(1, 1, 200) % 100
    raw = pack(m.tensors["x"], x, plan).data
    assert len(raw) == 2 * 512 and raw[200 * 4 :] == bytes(56 * 4)
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
    m = manifest(tmp_path)
    for name_ in VARIANTS:
        plan = plan_case(m, {"c": 1, "h": 2, "w": 128}, variant(name_))
        for name in ("x", "y"):
            assert stored_rows(m.tensors[name], plan) == default_rows(plan, m.tensors[name])
            assert stored_rows(m.tensors[name], plan) == 2  # one row a vector in both modes


def test_packed_sizes_follow_the_mode(tmp_path):
    m = manifest(tmp_path)
    x = np.zeros((3, 5, 32))
    # Row-major: 3 x 5 rows of 128 lanes; 1 byte an element in NX8 (plus a
    # metadata word a row), 4 in FP32.
    nx8 = pack(m.tensors["x"], x, plan_case(m, {}, variant("e4")))
    assert len(nx8.data) == 15 * 128 and len(nx8.metadata) == 15
    assert len(pack(m.tensors["x"], x, plan_case(m, {}, F.FP32)).data) == 15 * 512
    two = plan_case(m, {"c": 1, "h": 2, "w": 128}, variant("e4"))
    assert len(pack(m.tensors["x"], np.zeros((1, 2, 128)), two).data) == 256
    two32 = plan_case(m, {"c": 1, "h": 2, "w": 128}, F.FP32)
    assert len(pack(m.tensors["x"], np.zeros((1, 2, 128)), two32).data) == 1024


def test_chunk_interleaved_matches_the_convolutions_packing(tmp_path):
    # The SHA-256 of what the old convolutions harness's pack_input_chunked
    # wrote for this input (frozen in E07-25, which deleted it): 3 channels of
    # 7 rows of 32, four rows a 128-element chunk, so 2 chunks of 3 rows.
    m = manifest(tmp_path, pack='pack = "chunk_interleaved"\n')
    plan = plan_case(m, {"c": 3, "h": 7, "w": 32}, F.FP32)
    x = values(F.FP32, (3, 7, 32)).astype(np.float32)
    packed = pack(m.tensors["x"], x, plan).data
    assert len(packed) == 2 * 3 * 128 * 4
    assert hashlib.sha256(packed).hexdigest() == (
        "9a0e4559806eba149ebe3119ba81dbe2e1e5da97cbcc83b6941f9a45e5e288fc"
    )


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
    assert pack(m.tensors["x"], np.ones((3, 5, 32)), plan).data == bytes(15 * 512)


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

#: Each projection's (k, n_out, n_tok, n_tg), today's (E02) data/weights/output
#: base rows, and the SHA-256 of the bytes the old loaders (``_load_data``,
#: ``_load_weights``) wrote for ``values(FP32, ...)``: frozen in E07-02 before
#: the loaders were deleted.
PROJECTIONS = {
    "proj_qkv_144_p4": (144, 432, 128, 2, (0, 1152, 2016), "bb567b61", "207d1547"),
    "proj_outproj_144_p4": (144, 144, 128, 2, (0, 1152, 1440), "bb567b61", "579b3981"),
    "proj_ffn1_144_p4": (144, 288, 128, 2, (0, 1152, 1728), "bb567b61", "9b36ec29"),
    "proj_ffn2_144_p4": (288, 144, 128, 2, (0, 2304, 2736), "efd1b5a5", "91ecf463"),
    "proj_qkv_192_p4": (192, 576, 64, 1, (0, 768, 1920), "c23ffd87", "61d6d371"),
    "proj_outproj_192_p4": (192, 192, 64, 1, (0, 768, 1152), "c23ffd87", "06ad8ca9"),
    "proj_ffn1_192_p4": (192, 384, 64, 1, (0, 768, 1536), "c23ffd87", "eed7989f"),
    "proj_ffn2_192_p4": (384, 192, 64, 1, (0, 1536, 2112), "0cc8eb8f", "950edb4f"),
    "proj_qkv_240_p4": (240, 720, 16, 1, (0, 960, 2400), "49a3db23", "1cc91bb1"),
    "proj_outproj_240_p4": (240, 240, 16, 1, (0, 960, 1440), "49a3db23", "4e7a1c3a"),
    "proj_ffn1_240_p4": (240, 480, 16, 1, (0, 960, 1920), "49a3db23", "e74cfcc5"),
    "proj_ffn2_240_p4": (480, 240, 16, 1, (0, 1920, 2880), "eb9e6454", "1af70750"),
}


@pytest.mark.parametrize("name", sorted(PROJECTIONS))
def test_projection_packing_is_byte_identical_to_the_loaders(tmp_path, name):
    k, n_out, n_tok, n_tg, bases, data_digest, weights_digest = PROJECTIONS[name]
    data = values(F.FP32, (4, n_tg, k, n_tok)).astype(np.float32)
    weights = values(F.FP32, (n_out, k)).astype(np.float32)
    directory = tmp_path / "projections" / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(
        PROJECTION.format(k=k, n_out=n_out, n_tg=n_tg, n_tok=n_tok)
    )
    m = load_manifest(directory)
    plan = plan_case(m, {}, F.FP32)
    digest = hashlib.sha256(pack(m.tensors["data"], data, plan).data).hexdigest()
    assert digest.startswith(data_digest)
    digest = hashlib.sha256(pack(m.tensors["weights"], weights, plan).data).hexdigest()
    assert digest.startswith(weights_digest)
    # Placed with packing's row counts and the logical shapes, the bases are today's.
    placement = place(plan, {name: stored_rows(t, plan) for name, t in m.tensors.items()})
    got = tuple(placement.region(t).base_row for t in ("data", "weights", "out"))
    assert got == bases


# --- what the acceptance criteria forbid ----------------------------------------------------


def test_no_lane_or_row_size_is_defined_outside_ipu_isa():
    defined = re.compile(r"^(LANES|ROW_BYTES|ELEM_BYTES)\s*[:=]", re.MULTILINE)
    for root in (HARNESS, KERNELS):
        for path in sorted(root.rglob("*.py")):
            assert not defined.search(path.read_text(encoding="utf-8")), path


def test_packing_has_no_format_branches():
    text = (HARNESS / "packing.py").read_text(encoding="utf-8")
    assert not re.search(r"\bif\b[^\n]*\b(fmt|format)\b\s*(==|is|in)\b", text)
