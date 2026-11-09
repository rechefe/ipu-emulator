"""What a kernel starts with (:mod:`ipu_harness.kernel_start`): R_MASK all ones,
every other register the launch does not write garbage, seeded.

The toy kernel scales a tensor and uses LR0 and LR1 only; every other register
is as the start left it, which is what the tests read.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.errors import ConfigurationError
from ipu_harness import stages
from ipu_harness.cases import expand
from ipu_harness.chain import Chain
from ipu_harness.kernel_start import GARBAGE, HOST, RESET_TO_ONES
from ipu_harness.manifest import ManifestError, check_abi, load_manifest
from ipu_harness.runner import RunOptions, run_kernel, stage
from ipu_harness.variants import variant
from ipu_isa.registers import REGISTER_DEFINITIONS

E4 = variant("e4")
ALL_ONES = bytes([255]) * 128

#: y = x * s; sets LR0 and LR1 and touches no other register.
ASM = """\
    SET LR0 CR0 ;
    SET LR1 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15.L {{ cr.qf }}.L ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15.L {{ cr.qf }}.L {{ cr.y }} LR1 ;;
    ADD LR1 LR1 CR1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
"""

TOML = """\
[kernel]
op = "scale8"
formats = ["e4"]
{extra}
[params]
rows = {{ type = "int", min = 1, max = 8 }}
s    = {{ type = "float" }}

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[derived]
qf = "case_qf"

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"
qf   = "CR7"

[golden]
reference = "reference.py:reference"
tolerance = {{ e4 = "EXACT" }}

[cases.default]
rows = 2
s = 2.0
"""

REFERENCE = """\
import numpy as np


def reference(x, *, rows, s):
    return {"y": np.asarray(x, dtype=np.float64) * s}
"""

HOOK = 'def run(ctx):\n    ctx.run_stage("scale8.asm")\n    ctx.run_stage("scale8.asm")\n'

X = (np.arange(2 * 128).reshape(2, 128) % 17 - 8).astype(np.float64)
PARAMS = {"rows": 2, "s": 2.0}


def _kernel(root: Path, name: str = "scale8", *, extra: str = "", hook: bool = False) -> Path:
    directory = root / "elementwise" / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text(ASM)
    (directory / "reference.py").write_text(REFERENCE)
    if hook:
        (directory / "stages.py").write_text(HOOK)
        extra += '\nstages = "stages.py:run"'
    toml = TOML.format(extra=extra).replace('op = "scale8"', f'op = "{name}"')
    (directory / "kernel.toml").write_text(toml)
    return directory


def _case(manifest, name: str = "default"):
    return next(c for c in expand(manifest) if c.name == name)


def _registers(model, banks=GARBAGE):
    return {
        (bank, i): model.read_register(bank, i)
        for bank in banks
        for i in range(REGISTER_DEFINITIONS[bank]["count"])
    }


@pytest.fixture
def scale(tmp_path):
    return load_manifest(_kernel(tmp_path))


# --- the groups cover every register ---------------------------------------------------


def test_every_register_is_in_exactly_one_group():
    grouped = [*GARBAGE, *RESET_TO_ONES, *HOST]
    assert sorted(grouped) == sorted(REGISTER_DEFINITIONS)


# --- garbage ---------------------------------------------------------------------------


def test_lrs_and_data_registers_start_as_garbage(scale):
    model = stage(scale, _case(scale), E4).model
    for bank in GARBAGE:
        definition = REGISTER_DEFINITIONS[bank]
        zero = bytes(definition["size_bytes"]) if definition["vector"] else 0
        for i in range(definition["count"]):
            assert model.read_register(bank, i) != zero, (bank, i)
    # The program sets LR0 and LR1 when it runs; LR7 is whatever the start left.
    assert model.read_register("lr", 7) not in (0, 1)


def test_the_same_case_starts_the_same_way_and_another_case_does_not(scale):
    case = _case(scale)
    first = _registers(stage(scale, case, E4).model)
    again = _registers(stage(scale, case, E4).model)
    other = _registers(stage(scale, dataclasses.replace(case, name="other"), E4).model)
    assert first == again
    assert first != other


def test_a_chain_step_gets_its_own_garbage(scale):
    chain = Chain(E4)
    seen = []
    for _ in range(2):
        chain.run(scale, PARAMS, {"x": X})
        seen.append(chain.model.read_register("lr", 9))  # the program never sets LR9
    assert seen[0] != seen[1]


def test_a_program_that_sets_its_registers_runs_on_garbage(scale):
    result = run_kernel(scale, _case(scale), E4, RunOptions())
    assert result.passed, result.describe()


# --- R_MASK ----------------------------------------------------------------------------


def test_r_mask_starts_all_ones(scale):
    assert stage(scale, _case(scale), E4).model.read_register("r_mask") == ALL_ONES


def test_every_chain_step_resets_r_mask(scale):
    chain = Chain(E4)
    chain.run(scale, PARAMS, {"x": X})
    chain.model.write_register("r_mask", bytes(128))  # as a step that masks lanes leaves it
    chain.run(scale, PARAMS, {"x": X})
    assert chain.model.read_register("r_mask") == ALL_ONES


def test_a_kernel_that_keeps_its_mask_keeps_it(tmp_path):
    keeper = load_manifest(_kernel(tmp_path, extra="keeps_mask = true"))
    assert keeper.keeps_mask
    chain = Chain(E4)
    chain.run(keeper, PARAMS, {"x": X})
    chain.model.write_register("r_mask", bytes(128))
    chain.run(keeper, PARAMS, {"x": X})
    assert chain.model.read_register("r_mask") == bytes(128)


def test_a_kept_mask_on_a_fresh_model_is_garbage(tmp_path):
    keeper = load_manifest(_kernel(tmp_path, extra="keeps_mask = true"))
    mask = stage(keeper, _case(keeper), E4).model.read_register("r_mask")
    assert mask not in (ALL_ONES, bytes(128))


def test_keeps_mask_must_be_a_bool(tmp_path):
    with pytest.raises(ManifestError, match="keeps_mask"):
        load_manifest(_kernel(tmp_path, extra='keeps_mask = "yes"'))


# --- the host's registers --------------------------------------------------------------


def test_the_crs_are_exactly_the_abi_writes_and_the_default_ds(scale):
    model = stage(scale, _case(scale), E4).model
    cr = [model.read_register("cr", i) for i in range(16)]
    assert cr[0] == 0 and cr[1] == 1
    assert cr[4] == 2 and cr[7] != 0  # rows; the E4 case's QF half
    assert cr[15] == 0x1080  # the default DS
    bound = {0, 1, 2, 3, 4, 5, 7, 15}
    assert [cr[i] for i in range(16) if i not in bound] == [0] * 8


# --- stages ----------------------------------------------------------------------------


def test_every_stage_of_a_kernel_gets_a_fresh_start(tmp_path, monkeypatch):
    manifest = load_manifest(_kernel(tmp_path, hook=True))
    seen = []
    real = stages.apply

    def spy(model, writes):
        seen.append((model.read_register("lr", 9), model.read_register("r_mask")))
        real(model, writes)
        model.write_register("r_mask", bytes(128))  # what a stage that masks lanes leaves

    monkeypatch.setattr(stages, "apply", spy)
    run_kernel(manifest, _case(manifest), E4, RunOptions())
    assert len(seen) == 2
    assert seen[0][0] != seen[1][0]  # each stage has its own garbage
    assert seen[0][1] == seen[1][1] == ALL_ONES  # and the mask is back to ones


# --- [abi] and stage tables take no LR -------------------------------------------------


def test_an_lr_in_abi_is_refused(tmp_path):
    with pytest.raises(ManifestError, match="binds LR3: LRs start as garbage"):
        directory = _kernel(tmp_path)
        toml = directory / "kernel.toml"
        toml.write_text(toml.read_text().replace('rows = "CR4"', 'rows = "LR3"'))
        load_manifest(directory)


def test_a_stage_table_binding_an_lr_is_refused(scale):
    with pytest.raises(ConfigurationError, match="binds LR3: LRs start as garbage"):
        check_abi(scale, {"x": "LR3"}, where="stage a")
