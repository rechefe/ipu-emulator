"""Registers from [abi]: bases, integers, element scalars, the CR-read check (E06-06)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_state import IpuState
from ipu_emu.reference import ReferenceModel
from ipu_harness import manifest as manifest_module
from ipu_harness.abi import apply, check_cr_reads, register_writes, template_names
from ipu_harness.manifest import KernelManifest, ManifestError, load_manifest
from ipu_harness.packing import stored_rows
from ipu_harness.params import PlanError, plan_case
from ipu_harness.placement import place
from ipu_isa.formats import ElementFormat as F

SCALE = """\
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = {{ type = "int", min = 1{unit} }}
s    = {{ type = "float" }}

[derived]
neg = "-rows"

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"
neg  = "LR3"

[cases.default]
rows = 4
s = 2.0
"""


def kernel(
    tmp_path: Path, toml: str, family: str = "elementwise", name: str = "scale"
) -> KernelManifest:
    directory = tmp_path / family / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(toml)
    return load_manifest(directory)


def writes(m: KernelManifest, params: dict[str, object], fmt: F) -> dict[str, int]:
    plan = plan_case(m, params, fmt)
    return {w.register: w.value for w in register_writes(plan, place(plan))}


#: Hand-encoded 2.0: FP32 0 10000000 000..., INT8 2, E4 0 1000 000 (exponent 8 = bias 7 + 1).
TWO = {F.FP32: 0x40000000, F.INT8: 0x02, F.E4: 0x40}


@pytest.mark.parametrize("fmt", [F.FP32, F.INT8, F.E4])
def test_scale_bases_and_integers_are_the_same_in_every_format_and_s_is_encoded(tmp_path, fmt):
    m = kernel(tmp_path, SCALE.format(unit=""))
    got = writes(m, {"rows": 4, "s": 2.0}, fmt)
    assert {k: got[k] for k in ("CR2", "CR3", "CR4")} == {"CR2": 0, "CR3": 4, "CR4": 4}
    assert got["CR5"] == TWO[fmt]
    assert got["LR3"] == 0xFFFFFFFC  # -4, two's complement in 32 bits
    # valid_elements 128 in bits 7:0; partition P0 and pad ZERO above it are 0.
    assert got["CR15"] == 0x80


def test_a_negative_int8_scalar_is_its_low_byte(tmp_path):
    m = kernel(tmp_path, SCALE.format(unit=""))
    assert writes(m, {"rows": 4, "s": -3.0}, F.INT8)["CR5"] == 0xFD


def test_a_tensor_in_a_bytes_kernel_is_its_base_in_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_module, "BYTE_UNIT_KERNELS", frozenset({"legacy/scale"}))
    m = kernel(tmp_path, SCALE.format(unit=', unit = "bytes"'), family="legacy")
    got = writes(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert (got["CR2"], got["CR3"]) == (0, 4 * 512)  # base row 4, 512 B a FP32 row
    assert writes(m, {"rows": 4, "s": 2.0}, F.INT8)["CR3"] == 4 * 128


def test_a_tensor_binding_with_an_offset_is_its_base_plus_the_offset(tmp_path):
    toml = SCALE.format(unit="").replace('y    = "CR3"', 'y    = { register = "CR3", offset = -1 }')
    m = kernel(tmp_path, toml)
    assert m.abi["y"] == "CR3" and m.abi_offsets["y"] == -1
    assert writes(m, {"rows": 4, "s": 2.0}, F.FP32)["CR3"] == 3  # y at row 4, minus 1


def test_a_stage_table_binds_a_region_at_its_base_whatever_the_kernels_offset(tmp_path):
    toml = SCALE.format(unit="").replace('y    = "CR3"', 'y    = { register = "CR3", offset = -1 }')
    m = kernel(tmp_path, toml)
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    got = {w.register: w.value for w in register_writes(plan, place(plan), abi={"y": "CR6"})}
    assert got["CR6"] == 4  # y's base row: the [abi] offset is the kernel's binding only


def test_a_derived_value_may_read_a_regions_base(tmp_path):
    """Several offsets of one region (the x128 matmuls' weights + 1..3, E07)."""
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', 'neg = "-rows"\ny1 = "base(y) + 1"\ny2 = "y1 + rows"'
    )
    toml = toml.replace('neg  = "LR3"', 'neg  = "LR3"\ny1   = "CR6"\ny2   = "CR7"')
    m = kernel(tmp_path, toml)
    assert m.placed == {"y1": frozenset({"y"}), "y2": frozenset({"y"})}
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert "y1" not in plan.derived and plan.derived["neg"] == -4
    got = writes(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert (got["CR6"], got["CR7"]) == (5, 9)  # y at row 4: 4 + 1, then 5 + 4


def test_a_region_may_reach_the_program_only_through_a_derived_value(tmp_path):
    """A region bound at a parameter-dependent offset (sample_descriptors'
    per-cell table, E07-15) needs no binding of its own; one that no bound
    value reads still does."""
    toml = SCALE.format(unit="").replace('neg = "-rows"', 'neg = "-rows"\nyr = "base(y) + rows"')
    bound = toml.replace('y    = "CR3"', 'yr   = "CR3"')
    got = writes(kernel(tmp_path, bound), {"rows": 4, "s": 2.0}, F.FP32)
    assert got["CR3"] == 8  # y at row 4, plus rows
    with pytest.raises(ManifestError, match="does not bind tensor 'y'"):
        kernel(tmp_path / "unbound", toml.replace('y    = "CR3"\n', ""))
    # Bound only in some cases: y, which always exists, would lose its base.
    flagged = toml.replace("[params]\n", '[params]\nflag = { type = "bool", default = true }\n')
    flagged = flagged.replace('y    = "CR3"', 'yr   = { register = "CR3", when = "flag" }')
    with pytest.raises(ManifestError, match="does not bind tensor 'y'"):
        kernel(tmp_path / "conditional", flagged)


def test_a_bytes_kernels_base_counts_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_module, "BYTE_UNIT_KERNELS", frozenset({"elementwise/scale"}))
    toml = SCALE.format(unit=', unit = "bytes"').replace(
        'neg = "-rows"', 'neg = "-rows"\nyb = "base(y) + 1"'
    )
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nyb   = "CR6"'))
    assert writes(m, {"rows": 4, "s": 2.0}, F.FP32)["CR6"] == 4 * 512 + 1


@pytest.mark.parametrize(
    ("derived", "where", "match"),
    [
        ('w = "base(z) + 1"', "", "base(z) names no tensor"),
        ('w = "base(1)"', "", "base() takes one tensor name"),
        ('w = "base(y) + w"', "", "names no parameter or derived value: ['w']"),
        ('w = "base(y)"', 'shape = ["w", 128]', "reads ['w'], which read a region's base"),
        (
            'w = "base(x)"',
            'shape = ["rows", 128]\nrow = "w + 4"',
            "only [abi] may read such a value",
        ),
    ],
)
def test_a_placed_value_needs_a_tensor_and_only_abi_reads_it(tmp_path, derived, where, match):
    toml = SCALE.format(unit="").replace('neg = "-rows"', f'neg = "-rows"\n{derived}')
    if where:
        toml = toml.replace('[outputs.y]\nshape = ["rows", 128]', f"[outputs.y]\n{where}")
    with pytest.raises(ManifestError, match=re.escape(match)):
        kernel(tmp_path, toml)


def test_a_placed_value_cannot_be_the_cost(tmp_path):
    toml = SCALE.format(unit="").replace(
        'formats = ["fp32", "int8", "e4"]', 'formats = ["fp32", "int8", "e4"]\ncost = "w"'
    )
    with pytest.raises(ManifestError, match=r"\[kernel\] cost 'w' reads \['w'\]"):
        kernel(tmp_path, toml.replace('neg = "-rows"', 'neg = "-rows"\nw = "base(y)"'))


def test_a_stage_abi_may_bind_a_placed_value(tmp_path):
    toml = SCALE.format(unit="").replace('neg = "-rows"', 'neg = "-rows"\ny1 = "base(y) + 1"')
    m = kernel(tmp_path, toml)
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    got = {w.register: w.value for w in register_writes(plan, place(plan), abi={"y1": "CR8"})}
    assert got["CR8"] == 5


def test_a_derived_dstructure_binds_any_cr(tmp_path):
    """dstructure(valid, partition): the word, without its bit layout in the file."""
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', 'neg = "-rows"\ndtok = "dstructure(rows * 16, 8)"'
    )
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\ndtok = "CR13"'))
    got = writes(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert got["CR13"] == 64 | 8 << 8  # valid elements, then the partition byte


@pytest.mark.parametrize(
    ("expression", "word"),
    [
        ("dstructure(128)", "128"),
        ("dstructure(64, partition=8)", "64 + 8 * 256"),
        ("dstructure(16, 2, 0)", "16 + 2 * 256"),
        ("dstructure(16, pad_mode=1)", "16 + 1 * 8192"),  # the pad mode from bit 13
        ("dstructure(0)", "0"),
    ],
)
def test_a_dstructure_takes_its_last_two_by_position_or_name(tmp_path, expression, word):
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', f'neg = "-rows"\nd = "{expression}"\ne = "{word}"'
    )
    toml = toml.replace('neg  = "LR3"', 'neg  = "LR3"\nd    = "CR13"\ne    = "CR14"')
    got = writes(kernel(tmp_path, toml), {"rows": 4, "s": 2.0}, F.FP32)
    assert got["CR13"] == got["CR14"]


@pytest.mark.parametrize(
    ("expression", "match"),
    [
        ("dstructure(64, 3)", "3 is not a valid Partition"),
        ("dstructure(64, 0, 3)", "3 is not a valid PadMode"),
        ("dstructure(rows / 3)", "whole numbers"),
        ("dstructure(129)", "valid elements must be 0..128"),
        ("dstructure(-1)", "valid elements must be 0..128"),
    ],
)
def test_a_bad_dstructure_is_located(tmp_path, expression, match):
    toml = SCALE.format(unit="").replace('neg = "-rows"', f'neg = "-rows"\nd = "{expression}"')
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nd    = "CR13"'))
    with pytest.raises(PlanError, match=match):
        plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)


@pytest.mark.parametrize(
    "expression",
    [
        "dstructure(64, 0, 0, 1)",
        "dstructure()",
        "dstructure(64, valid=8)",
        "dstructure(64, 8, partition=2)",
    ],
)
def test_a_misshapen_dstructure_call_is_a_load_error(tmp_path, expression):
    toml = SCALE.format(unit="").replace('neg = "-rows"', f'neg = "-rows"\nd = "{expression}"')
    with pytest.raises(ManifestError, match=r"dstructure\(\) (takes|got)"):
        kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nd    = "CR13"'))


def test_a_tensor_pinned_at_row_0_can_be_read_through_cr0(tmp_path):
    toml = SCALE.format(unit="").replace(
        '[inputs.x]\nshape = ["rows", 128]', '[inputs.x]\nshape = ["rows", 128]\nrow = 0'
    )
    toml = toml.replace('x    = "CR2"', 'x    = "CR0"')
    m = kernel(tmp_path, toml)
    got = writes(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert "CR0" not in got and "CR2" not in got  # nothing is written for x
    assert template_names(m)["cr"]["x"] == "CR0"


def test_binding_cr15_replaces_the_default_dstructure(tmp_path):
    m = kernel(tmp_path, SCALE.format(unit="").replace('s    = "CR5"', 's    = "CR15"'))
    got = writes(m, {"rows": 4, "s": 2.0}, F.FP32)
    assert got["CR15"] == 0x40000000 and "CR5" not in got


def test_a_str_parameter_cannot_be_bound(tmp_path):
    toml = SCALE.format(unit="").replace(
        's    = { type = "float" }',
        's    = { type = "float" }\nmode = { type = "str", default = "a" }',
    )
    with pytest.raises(ManifestError, match="mode is a str parameter"):
        kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nmode = "CR6"'))


def test_template_names_and_apply_through_the_register_file(tmp_path):
    m = kernel(tmp_path, SCALE.format(unit=""))
    names = template_names(m)
    assert dict(names["cr"]) == {"x": "CR2", "y": "CR3", "rows": "CR4", "s": "CR5"}
    assert dict(names["lr"]) == {"neg": "LR3"}
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    model = ReferenceModel(F.FP32)
    apply(model, register_writes(plan, place(plan)))
    assert [model.read_register("cr", i) for i in (2, 3, 4, 5)] == [0, 4, 4, 0x40000000]
    assert model.read_register("lr", 3) == 0xFFFFFFFC


# --- the CR-read check -----------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "    SET lr1 cr9;;",  # a CR operand
        "    ADD lr0 lr1 cr9;;",  # an LR-or-CR operand spelled CRn
        "    BZ cr9 top;;",  # a pseudo-instruction (BEQ cr9 CR0 top)
        "    SET lr1 CR9;;",  # any case
        "    SET lr0 cr2; SET lr1 cr9;;",  # not the word's first instruction
    ],
)
def test_a_read_of_an_unbound_cr_names_its_line(tmp_path, line):
    m = kernel(tmp_path, SCALE.format(unit=""))
    program = f"top:\n    SET lr0 cr2;;\n{line}\n    B top;;\n"
    message = rf"scale.asm, rendered line 3 \({re.escape(repr(line.strip()))}\): reads CR9"
    with pytest.raises(ConfigurationError, match=message):
        check_cr_reads(m, program)


def test_hardwired_bound_and_dstructure_crs_and_labels_named_like_crs_pass(tmp_path):
    m = kernel(tmp_path, SCALE.format(unit=""))
    program = (
        "cr9:\n"  # a label, not a register
        "    SET lr0 cr0;;\n    SET lr1 cr1;;\n    SET lr2 cr5;;\n"
        "    ADD lr0 lr1 lr2;;\n"  # an LR-or-CR operand spelled LRn
        "    MULT.VE lr0 cr2 0 lr0 cr15;;\n"
        "    B cr9;;\n"
    )
    check_cr_reads(m, program)


def test_a_value_that_does_not_fit_a_register_is_located(tmp_path):
    toml = SCALE.format(unit="").replace(
        'neg = "-rows"', 'neg = "-rows"\nhuge = "rows * 1000000000"'
    )
    m = kernel(tmp_path, toml.replace('neg  = "LR3"', 'neg  = "LR3"\nhuge = "LR4"'))
    assert writes(m, {"rows": 4, "s": 2.0}, F.FP32)["LR4"] == 4_000_000_000  # an unsigned word
    plan = plan_case(m, {"rows": 5, "s": 2.0}, F.FP32)
    with pytest.raises(PlanError, match="does not fit a 32-bit register") as error:
        register_writes(plan, place(plan))
    assert error.value.key == "abi.huge"


# --- one register for two pinned regions (vector_tour's acc and relu, E07-26) -------------

#: ``y2`` is stored right after ``y``'s 4 rows; the program reaches it from CR3.
TWO_OUTPUTS = (
    SCALE.format(unit="")
    .replace(
        '[outputs.y]\nshape = ["rows", 128]\n',
        "[outputs.y]\nshape = [4, 128]\nrow = 8\n\n[outputs.y2]\nshape = [4, 128]\nrow = ROW\n",
    )
    .replace('y    = "CR3"', 'y    = "CR3"\ny2   = { register = "CR3", offset = -4 }')
)


def test_two_pinned_regions_share_a_register_that_reaches_both(tmp_path):
    m = kernel(tmp_path, TWO_OUTPUTS.replace("ROW", "12"))
    assert writes(m, {"rows": 4, "s": 2.0}, F.FP32)["CR3"] == 8


@pytest.mark.parametrize(
    "row",
    [pytest.param("13", id="row-plus-offset-9-not-8"), pytest.param('"rows + 8"', id="expression")],
)
def test_regions_whose_pins_disagree_do_not_share_a_register(tmp_path, row):
    with pytest.raises(ManifestError, match="y2 and y both bind CR3"):
        kernel(tmp_path, TWO_OUTPUTS.replace("ROW", row))


def test_a_shared_register_a_caller_pin_splits_is_refused(tmp_path):
    m = kernel(tmp_path, TWO_OUTPUTS.replace("ROW", "12"))
    plan = plan_case(m, {"rows": 4, "s": 2.0}, F.FP32)
    with pytest.raises(PlanError, match="y2 .region base -4. = 8, but y .region base. put 16") as e:
        register_writes(plan, place(plan, pins={"y": 16}))  # a chain places y elsewhere
    assert e.value.key == "abi.y2"


# --- the projections' 13 CRs (guards E07-02) ----------------------------------------------

PROJECTION = """\
[kernel]
op = "projection"
formats = ["fp32"]

[params]
k     = {{ type = "int", default = {k} }}
n_out = {{ type = "int", default = {n_out} }}
n_tg  = {{ type = "int", default = {n_tg} }}
n_tok = {{ type = "int", default = {n_tok} }}

[derived]
neg_one         = "-1"
full_bound      = "126"
w_stride        = "ceil(k / 128)"
chunk_count     = "ceil(k / 128)"
last_chunk      = "ceil(k / 128) - 1"
tail_bound      = "k - 128 * (ceil(k / 128) - 1) - 2"
data_stream_str = "k * n_tg"
out_stream_str  = "n_tg * n_out"
stream_count    = "4"

[inputs.data]
shape = [4, "n_tg", "k", "n_tok"]
pack = "channel_major"

[inputs.weights]
shape = ["n_out", "k"]
pack = "output_major_weights"

[outputs.out]
shape = [4, "n_tg", "n_out", 128]

[abi]
data            = "CR2"
weights         = "CR3"
out             = "CR4"
neg_one         = "CR5"
full_bound      = "CR6"
tail_bound      = "CR7"
w_stride        = "CR8"
n_out           = "CR9"
chunk_count     = "CR10"
last_chunk      = "CR11"
data_stream_str = "CR12"
out_stream_str  = "CR13"
stream_count    = "CR14"

[cases.default]
"""

#: Each projection's (k, n_out, n_tok, n_tg) and the CR2..CR15 the ipu_apps
#: harness's setup() wrote (E02 baselines; frozen in E07-02, which deleted it).
PROJECTIONS = {
    "proj_qkv_144_p4": (144, 432, 128, 2, (0, 1152, 2016, -1, 126, 14, 2, 432, 2, 1, 288, 864, 4)),
    "proj_outproj_144_p4": (
        144,
        144,
        128,
        2,
        (0, 1152, 1440, -1, 126, 14, 2, 144, 2, 1, 288, 288, 4),
    ),
    "proj_ffn1_144_p4": (144, 288, 128, 2, (0, 1152, 1728, -1, 126, 14, 2, 288, 2, 1, 288, 576, 4)),
    "proj_ffn2_144_p4": (288, 144, 128, 2, (0, 2304, 2736, -1, 126, 30, 3, 144, 3, 2, 576, 288, 4)),
    "proj_qkv_192_p4": (192, 576, 64, 1, (0, 768, 1920, -1, 126, 62, 2, 576, 2, 1, 192, 576, 4)),
    "proj_outproj_192_p4": (
        192,
        192,
        64,
        1,
        (0, 768, 1152, -1, 126, 62, 2, 192, 2, 1, 192, 192, 4),
    ),
    "proj_ffn1_192_p4": (192, 384, 64, 1, (0, 768, 1536, -1, 126, 62, 2, 384, 2, 1, 192, 384, 4)),
    "proj_ffn2_192_p4": (384, 192, 64, 1, (0, 1536, 2112, -1, 126, 126, 3, 192, 3, 2, 384, 192, 4)),
    "proj_qkv_240_p4": (240, 720, 16, 1, (0, 960, 2400, -1, 126, 110, 2, 720, 2, 1, 240, 720, 4)),
    "proj_outproj_240_p4": (
        240,
        240,
        16,
        1,
        (0, 960, 1440, -1, 126, 110, 2, 240, 2, 1, 240, 240, 4),
    ),
    "proj_ffn1_240_p4": (240, 480, 16, 1, (0, 960, 1920, -1, 126, 110, 2, 480, 2, 1, 240, 480, 4)),
    "proj_ffn2_240_p4": (480, 240, 16, 1, (0, 1920, 2880, -1, 126, 94, 4, 240, 4, 3, 480, 240, 4)),
}


@pytest.mark.parametrize("name", sorted(PROJECTIONS))
def test_the_projections_abi_reproduces_todays_crs(tmp_path, name):
    k, n_out, n_tok, n_tg, crs = PROJECTIONS[name]
    toml = PROJECTION.format(k=k, n_out=n_out, n_tg=n_tg, n_tok=n_tok)
    m = kernel(tmp_path, toml, family="projections", name=name)
    # As E07-02 runs it: logical shapes, the shared transforms, packing's row counts.
    plan = plan_case(m, {}, F.FP32)
    placement = place(plan, {n: stored_rows(t, plan) for n, t in m.tensors.items()})
    got = {int(w.register[2:]): w.value for w in register_writes(plan, placement)}
    today = {cr: value & 0xFFFFFFFF for cr, value in enumerate(crs, start=2)}
    # Today's harness left CR15 at the state's default dstructure.
    assert got == {**today, 15: IpuState().regfile.get_cr(15)}
