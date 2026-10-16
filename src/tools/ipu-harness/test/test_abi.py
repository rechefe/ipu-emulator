"""Registers from [abi]: bases, integers, element scalars, the CR-read check (E06-06)."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ipu_apps.kernels.projections.app import ProjectionLayout, ProjectionP4App
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_state import IpuState
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
    state = IpuState()
    apply(state, register_writes(plan, place(plan)))
    assert [state.regfile.get_cr(i) for i in (2, 3, 4, 5)] == [0, 4, 4, 0x40000000]
    assert state.regfile.get_lr(3) == 0xFFFFFFFC


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

PROJECTIONS = [
    f"proj_{kind}_{width}_p4"
    for kind in ("qkv", "outproj", "ffn1", "ffn2")
    for width in (144, 192, 240)
]


class _Recorder:
    def __init__(self) -> None:
        self.crs: dict[int, int] = {}

    def set_cr(self, index: int, value: int) -> None:
        self.crs[index] = value & 0xFFFFFFFF

    def write_address(self, address: int, data: bytes) -> None:
        pass


@pytest.mark.parametrize("name", PROJECTIONS)
def test_the_projections_abi_reproduces_todays_crs(tmp_path, name):
    app = importlib.import_module(f"ipu_apps.kernels.projections.{name}.app")
    layout = ProjectionLayout(k=app.K, n_out=app.N_OUT, n_tok=app.N_TOK, n_tg=app.N_TG)
    (tmp_path / "d.bin").write_bytes(np.zeros(layout.input_shape, dtype="<f4").tobytes())
    (tmp_path / "w.bin").write_bytes(np.zeros((layout.n_out, layout.k), dtype="<f4").tobytes())
    recorder = _Recorder()
    harness = SimpleNamespace(
        layout=layout, input_path=tmp_path / "d.bin", weights_path=tmp_path / "w.bin"
    )
    harness._load_data = lambda state: ProjectionP4App._load_data(harness, state)
    harness._load_weights = lambda state: ProjectionP4App._load_weights(harness, state)
    ProjectionP4App.setup(harness, SimpleNamespace(regfile=recorder, xmem=recorder))
    toml = PROJECTION.format(k=layout.k, n_out=layout.n_out, n_tg=layout.n_tg, n_tok=layout.n_tok)
    m = kernel(tmp_path, toml, family="projections", name=name)
    # As E07-02 runs it: logical shapes, the shared transforms, packing's row counts.
    plan = plan_case(m, {}, F.FP32)
    placement = place(plan, {n: stored_rows(t, plan) for n, t in m.tensors.items()})
    got = {int(w.register[2:]): w.value for w in register_writes(plan, placement)}
    # Today's harness leaves CR15 at the state's default dstructure.
    assert got == {**recorder.crs, 15: IpuState().regfile.get_cr(15)}
