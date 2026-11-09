"""Table IDs (:mod:`ipu_harness.table_ids`): a tensor's CR holds a bare ID 0-15,
the harness binds each ID to the tensor's rows on the model, and a chain hands
an ID from one step to the next (or retags it).

Every test runs a kernel through the public harness (``stage``, ``run_kernel``,
``Chain``); the expected IDs and rows are written by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scale_kernel import write_scale

from ipu_emu.errors import ConfigurationError
from ipu_harness.cases import expand
from ipu_harness.chain import Chain, Resident
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.runner import RunOptions, run_kernel, stage
from ipu_harness.table_ids import PINNED, TableBinder
from ipu_harness.variants import variant
from ipu_isa.faults import FaultId

FP32, E4 = variant("fp32"), variant("e4")

TABLES = "\n[tables]\nx = {}\ny = {}\n"


def _scale(tmp_path: Path, *replacements: tuple[str, str], tables: bool = True) -> KernelManifest:
    """The scale kernel (x on CR2, y on CR3), with text replacements in its kernel.toml."""
    manifest = write_scale(tmp_path, with_reference=False)
    toml = manifest.directory / "kernel.toml"
    text = toml.read_text()
    for old, new in replacements:
        assert old in text
        text = text.replace(old, new)
    toml.write_text(text + (TABLES if tables else ""))
    return load_manifest(manifest.directory)


def _default(manifest: KernelManifest):
    return next(c for c in expand(manifest) if c.name == "default")


# -- the IDs a run writes, and the tables it binds --------------------------------------------


def test_stage_writes_the_ids_not_the_base_rows(tmp_path):
    staged = stage(_scale(tmp_path), _default(_scale(tmp_path / "p")), FP32)
    model, placement = staged.model, staged.placement
    # x is the first tensor, y the second: IDs 0 and 1, whatever their rows.
    assert model.read_register("cr", 2) == 0
    assert model.read_register("cr", 3) == 1
    assert placement.region("y").base_row == 4  # y's row: 1 is not it
    assert dict(staged.tables.ids) == {"x": 0, "y": 1}


def test_the_bound_tables_are_the_placed_rows(tmp_path):
    manifest = _scale(tmp_path)
    staged = stage(manifest, _default(manifest), FP32)
    x, y = staged.placement.region("x"), staged.placement.region("y")
    assert staged.model.tables() == {0: (x.base_row, 4), 1: (y.base_row, 4)}
    assert (x.base_row, y.base_row) == (0, 4)


def test_a_kernel_that_declares_no_tables_is_not_held_to_table_sizes(tmp_path):
    manifest = _scale(tmp_path, tables=False)
    staged = stage(manifest, _default(manifest), FP32)
    first, rows = staged.model.tables()[0]
    assert first == 0 and rows > 4  # reaches past x, so a run can measure its tables


def test_scattered_storage_gives_the_same_outputs(tmp_path):
    plain = _scale(tmp_path / "plain")
    scattered = _scale(
        tmp_path / "scattered",
        ('[inputs.x]\nshape = ["rows", 128]', '[inputs.x]\nshape = ["rows", 128]\nrow = 777'),
        ('[outputs.y]\nshape = ["rows", 128]', '[outputs.y]\nshape = ["rows", 128]\nrow = 9'),
    )
    want = run_kernel(plain, _default(plain), FP32)
    got = run_kernel(scattered, _default(scattered), FP32)
    assert want.passed and got.passed
    assert got.placement.region("x").base_row == 777 and got.placement.region("y").base_row == 9
    assert got.raw_outputs == want.raw_outputs and got.cycles == want.cycles


def test_the_ids_are_the_binders_to_choose(tmp_path):
    """No program relies on an ID's value: a permutation of the IDs runs the same."""
    manifest = _scale(tmp_path)
    case = _default(manifest)
    plain = run_kernel(manifest, case, FP32)
    order = (11, 4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15)
    permuted = run_kernel(manifest, case, FP32, RunOptions(table_id_order=order))
    assert permuted.passed
    assert dict(permuted.table_ids) == {"x": 11, "y": 4}
    assert permuted.raw_outputs == plain.raw_outputs and permuted.cycles == plain.cycles


def test_an_access_outside_its_table_faults(tmp_path):
    manifest = write_scale(tmp_path, with_reference=False)
    program = manifest.directory / "scale.asm"
    prefetch = "    LD.MULT R0 LR1 {{ cr.x }} ;;\n    HALT ;;"
    program.write_text(program.read_text().replace("    HALT ;;", prefetch))
    toml = manifest.directory / "kernel.toml"
    toml.write_text(toml.read_text() + TABLES)
    manifest = load_manifest(manifest.directory)
    result = run_kernel(manifest, _default(manifest), FP32)
    assert not result.passed
    assert result.fault is not None and result.fault.fault is FaultId.F6


# -- the binder -------------------------------------------------------------------------------


def _binder(tmp_path: Path, **kwargs) -> TableBinder:
    manifest = _scale(tmp_path)
    staged = stage(manifest, _default(manifest), FP32)
    return TableBinder(staged.plan, staged.placement, **kwargs)


def test_free_tables_take_the_lowest_free_ids_in_abi_order(tmp_path):
    binder = _binder(tmp_path)
    assert dict(binder.bind({"y": "CR3", "x": "CR2"}).ids) == {"y": 0, "x": 1}


def test_a_tensor_on_cr0_is_table_0_and_the_others_go_around_it(tmp_path):
    binder = _binder(tmp_path)
    got = binder.bind({"y": "CR3", "x": "CR0"})
    assert dict(got.ids) == {"y": 1, "x": 0}
    assert got.pinned == frozenset({0})
    assert PINNED == {"CR0": 0, "CR1": 1}


def test_a_tensor_keeps_its_id_in_a_later_program(tmp_path):
    binder = _binder(tmp_path)
    binder.bind({"x": "CR2", "y": "CR3"})
    assert dict(binder.bind({"y": "CR7"}).ids) == {"y": 1}  # another CR, one ID


def test_a_tensor_with_an_id_cannot_become_a_pinned_table(tmp_path):
    binder = _binder(tmp_path)
    binder.bind({"x": "CR2", "y": "CR3"})
    with pytest.raises(ConfigurationError, match="y has table ID 1.*pinned table 0"):
        binder.bind({"y": "CR0"})


def test_more_than_sixteen_tables_are_refused(tmp_path):
    binder = _binder(tmp_path)
    binder.ids.update({f"t{i}": i for i in range(15)})  # fifteen IDs taken
    assert dict(binder.bind({"x": "CR2"}).ids) == {"x": 15}
    with pytest.raises(ConfigurationError, match="a program has 16 IDs"):
        binder.bind({"y": "CR3"})


def test_a_hint_is_kept_when_free_and_otherwise_the_table_is_retagged(tmp_path):
    kept = _binder(tmp_path / "kept", hints={"x": 5})
    assert dict(kept.bind({"x": "CR2", "y": "CR3"}).ids) == {"x": 5, "y": 0}
    assert kept.retagged == {}
    retagged = _binder(tmp_path / "retag", hints={"x": 0})
    got = retagged.bind({"y": "CR3", "x": "CR2"})  # y is bound first and takes 0
    assert dict(got.ids) == {"y": 0, "x": 1}
    assert retagged.retagged == {"x": (0, 1)}


def test_a_kernel_written_buffer_is_never_on_a_pinned_table(tmp_path):
    binder = _binder(tmp_path, kernel_written=frozenset({"x"}))
    with pytest.raises(ConfigurationError, match="never handed over through a pinned table"):
        binder.bind({"x": "CR0", "y": "CR3"})
    assert dict(binder.bind({"x": "CR2", "y": "CR3"}).ids) == {"x": 0, "y": 1}  # a free one is fine


# -- chains: keep, retag, copy ----------------------------------------------------------------

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
op = "{name}"
formats = ["e4"]

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
{abi}

[tables]
x = {{}}
y = {{}}

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

FREE = 'x = "CR2"\ny = "CR3"\nrows = "CR4"\ns = "CR5"\nqf = "CR7"'
Y_FIRST = 'y = "CR3"\nx = "CR2"\nrows = "CR4"\ns = "CR5"\nqf = "CR7"'
PINNED_X = 'x = "CR0"\ny = "CR3"\nrows = "CR4"\ns = "CR5"\nqf = "CR7"'

X = (np.arange(2 * 128).reshape(2, 128) % 17 - 8).astype(np.float64)
PARAMS = {"rows": 2, "s": 2.0}


def _chain_kernel(root: Path, name: str, abi: str) -> KernelManifest:
    directory = root / "elementwise" / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text(ASM)
    (directory / "reference.py").write_text(REFERENCE)
    (directory / "kernel.toml").write_text(TOML.format(name=name, abi=abi))
    return load_manifest(directory)


@pytest.fixture
def free(tmp_path):
    return _chain_kernel(tmp_path, "free", FREE)


@pytest.fixture
def y_first(tmp_path):
    return _chain_kernel(tmp_path, "y_first", Y_FIRST)


@pytest.fixture
def pinned(tmp_path):
    return _chain_kernel(tmp_path, "pinned", PINNED_X)


def test_a_pinned_input_may_sit_anywhere(pinned):
    chain = Chain(E4)
    step = chain.run(pinned, PARAMS, {"x": X}, check=True, at={"x": 300})
    assert step.result.passed
    assert step.result.placement.region("x").base_row == 300
    assert dict(step.result.table_ids) == {"x": 0, "y": 1}


def test_a_step_hands_its_output_id_to_the_next(free, y_first):
    chain = Chain(E4)
    one = chain.run(y_first, PARAMS, {"x": X})  # y is bound first: ID 0, x is ID 1
    assert dict(one.result.table_ids) == {"y": 0, "x": 1}
    assert one.resident["y"].table_id == 0 and one.resident["y"].kernel_written
    two = chain.run(free, PARAMS, {"x": one.resident["y"]}, check=True)
    assert two.result.passed
    assert dict(two.result.table_ids) == {"x": 0, "y": 1}  # x keeps the ID y had
    assert two.retags == () and chain.retags == ()
    assert chain.model.tables() == {
        0: (two.result.placement.region("x").base_row, 2),
        1: (two.result.placement.region("y").base_row, 2),
    }


def test_a_taken_id_is_retagged_and_counted(y_first):
    chain = Chain(E4)
    one = chain.run(y_first, PARAMS, {"x": X})  # y is bound first: ID 0
    assert one.resident["y"].table_id == 0
    two = chain.run(y_first, PARAMS, {"x": one.resident["y"]}, check=True)
    # y takes 0 before x is bound, so x cannot keep 0: its table is retagged to 1.
    assert two.result.passed
    assert dict(two.result.table_ids) == {"y": 0, "x": 1}
    (retag,) = two.retags
    assert (retag.step, retag.kernel, retag.tensor, retag.was, retag.now) == (
        1,
        "y_first",
        "x",
        0,
        1,
    )
    assert chain.retags == two.retags
    # The rows never move: x is where the first step left y.
    assert two.result.placement.region("x").base_row == one.resident["y"].base_row


def test_a_kernel_written_buffer_is_refused_by_a_pinned_input(free, pinned):
    chain = Chain(E4)
    one = chain.run(free, PARAMS, {"x": X})
    with pytest.raises(ConfigurationError, match="never handed over through a pinned table"):
        chain.run(pinned, PARAMS, {"x": one.resident["y"]})


def test_a_host_copy_goes_to_a_pinned_input_and_is_counted(free, pinned):
    chain = Chain(E4)
    one = chain.run(free, PARAMS, {"x": X})
    assert chain.host_writes == 0
    copy = chain.write(chain.read(one.resident["y"]), pack=one.resident["y"].pack)
    assert chain.host_writes == 1
    assert copy.table_id is None and not copy.kernel_written
    two = chain.run(pinned, PARAMS, {"x": copy}, check=True)
    assert two.result.passed
    assert dict(two.result.table_ids)["x"] == 0


def test_a_host_written_resident_takes_the_next_free_id(free):
    chain = Chain(E4)
    resident = chain.write(X, pack=None)
    step = chain.run(free, PARAMS, {"x": resident}, check=True)
    assert step.result.passed
    assert dict(step.result.table_ids) == {"x": 0, "y": 1}


def test_a_part_of_a_kernel_written_tensor_is_kernel_written(free, pinned):
    chain = Chain(E4)
    whole = chain.reserve(4)
    chain.run(free, PARAMS, {"x": X}, at={"y": whole.slice(0, 2)})
    assert whole.kernel_written and whole.table_id == 1
    with pytest.raises(ConfigurationError, match="pinned table"):
        chain.run(pinned, PARAMS, {"x": whole.slice(0, 2)})


def test_a_resident_made_by_hand_has_no_id_and_no_author():
    resident = Resident(64, 2, None, E4)
    assert resident.table_id is None and not resident.kernel_written
    assert resident.slice(0, 1).table_id is None
