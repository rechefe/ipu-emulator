"""kernel.toml: the scale kernel loads into the expected view; each schema rule has a failing fixture."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from ipu_harness.manifest import KernelManifest, ManifestError, load_manifest
from ipu_isa.formats import ElementFormat

# The epic's worked example (06-kernel-framework/00-epic.md), line for line.
SCALE = """\
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = { type = "int", min = 1, max = 256 }
s    = { type = "float" }

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[golden]
reference = "reference.py:reference"
tolerance = { fp32 = "EXACT", int8 = "EXACT", e4 = "FP8_ULP" }

[cases.default]
rows = 4
s = 2.0

[cases.one_row]
rows = 1
s = -3.0

[cases.sweep]
rows = [1, 7, 33, 128]
s = 1.5
formats = ["fp32", "e4"]

[cases.mine]
rows = 2
s = 2.0
inputs = { x = "data/x.npy" }
expected = { y = "data/y.npy" }
"""


def kernel(tmp_path: Path, toml: str, name: str = "scale") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(toml)
    return directory


def test_the_scale_example_loads_into_the_expected_view(tmp_path):
    m = load_manifest(kernel(tmp_path, SCALE))
    assert (m.name, m.op, m.asm.name) == ("scale", "scale", "scale.asm")
    assert m.formats == (ElementFormat.FP32, ElementFormat.INT8, ElementFormat.E4)
    assert (m.params["rows"].type, m.params["rows"].min, m.params["rows"].max) == ("int", 1, 256)
    assert m.params["s"].type == "float"
    assert m.tensors["x"].role == "input" and m.tensors["x"].shape == ("rows", 128)
    assert m.tensors["y"].role == "output" and m.tensors["y"].store == "element"
    assert dict(m.abi) == {"x": "CR2", "y": "CR3", "rows": "CR4", "s": "CR5"}
    assert m.golden.reference == "reference.py:reference"
    assert m.golden.tolerance[ElementFormat.E4] == "FP8_ULP"
    assert list(m.cases) == ["default", "one_row", "sweep", "mine"]
    assert dict(m.cases["default"].values) == {"rows": 4, "s": 2.0}
    assert m.cases["sweep"].values["rows"] == (1, 7, 33, 128)
    assert m.cases["sweep"].formats == (ElementFormat.FP32, ElementFormat.E4)
    assert m.cases["mine"].inputs["x"] == m.directory / "data/x.npy"
    assert m.cases["default"].tier == "default" and m.stages is None


def test_the_view_is_immutable(tmp_path):
    m = load_manifest(kernel(tmp_path, SCALE))
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.op = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        m.params["new"] = m.params["s"]  # type: ignore[index]
    with pytest.raises(TypeError):
        m.cases["default"].values["rows"] = 5  # type: ignore[index]
    assert isinstance(m, KernelManifest)


def test_registry_overrides_set_the_name_and_parameter_defaults(tmp_path):
    m = load_manifest(kernel(tmp_path, SCALE), overrides={"name": "scale_8", "params": {"rows": 8}})
    assert (m.name, m.params["rows"].default, m.asm.name) == ("scale_8", 8, "scale.asm")
    with pytest.raises(ManifestError, match="outside its domain"):
        load_manifest(kernel(tmp_path, SCALE, "again"), overrides={"params": {"rows": 999}})


def test_the_default_reference_is_reference_py_when_present(tmp_path):
    toml = SCALE.replace('reference = "reference.py:reference"\n', "")
    without = load_manifest(kernel(tmp_path, toml))
    assert without.golden.reference is None
    with_file = kernel(tmp_path, toml, "scale2")
    (with_file / "reference.py").write_text("")
    assert load_manifest(with_file).golden.reference == "reference.py:reference"


def line_of(toml: str, needle: str) -> int:
    return next(i for i, line in enumerate(toml.splitlines(), start=1) if needle in line)


# Each rule: (an edit of the scale example, the line it makes wrong, the message).
RULES = [
    ('s    = "CR5"', 's    = "CR1"', "[abi] binds CR1, a hardwired constant"),
    ('s    = "CR5"', 's    = "CR0"', "[abi] binds CR0, a hardwired constant"),
    ('s    = "CR5"', 's    = "CR16"', "is no CR or LR register"),
    ('s    = "CR5"', 's    = "CR4"', "both bind CR4"),
    (
        's    = "CR5"',
        's    = "CR5"\nz    = "CR6"',
        "[abi] z names no parameter, tensor or derived value",
    ),
    ('y    = "CR3"', "", "[abi] does not bind tensor 'y'"),
    ("[cases.default]", "[cases.first]", "needs [cases.default]"),
    ("rows = 4", "rowz = 4", "unknown key 'rowz' in [cases.default]; did you mean 'rows'?"),
    ("rows = 4", "rows = 0", "rows = 0 is outside its domain"),
    ("rows = 4", 'rows = "four"', "rows must be int"),
    ("[golden]", "[goldn]", "unknown key 'goldn' in kernel.toml; did you mean 'golden'?"),
    (
        'formats = ["fp32", "int8", "e4"]',
        'formats = ["fp32", "fp16"]',
        "unknown element format 'fp16'",
    ),
    ('formats = ["fp32", "e4"]', 'formats = ["fp32", "e5"]', "formats ['e5'] are not the kernel's"),
    (
        'inputs = { x = "data/x.npy" }',
        'inputs = { x = "../x.npy" }',
        "must be a path inside the kernel directory",
    ),
    (
        'inputs = { x = "data/x.npy" }',
        'inputs = { x = "/tmp/x.npy" }',
        "must be a path inside the kernel directory",
    ),
    (
        'inputs = { x = "data/x.npy" }',
        'inputs = { y = "data/y.npy" }',
        "inputs 'y' is not one of the tensors ['x']",
    ),
    (
        'reference = "reference.py:reference"',
        'reference = "../ref.py:reference"',
        "must be a path inside",
    ),
    (
        'reference = "reference.py:reference"',
        'reference = "reference.py"',
        "must be 'file.py:function'",
    ),
    (
        'rows = { type = "int", min = 1, max = 256 }',
        'rows = { type = "integer" }',
        "type 'integer' must be one of",
    ),
    (
        'rows = { type = "int", min = 1, max = 256 }',
        'rows = { type = "int", maxx = 2 }',
        "did you mean 'max'?",
    ),
    (
        'rows = { type = "int", min = 1, max = 256 }',
        'rows = { type = "int", unit = "bytes" }',
        "counts bytes",
    ),
    (
        'shape = ["rows", 128]\n\n[outputs.y]',
        'shape = ["cols", 128]\n\n[outputs.y]',
        "names no parameter or derived value",
    ),
    (
        "tolerance = { fp32",
        'tolerance = { e5 = "X", fp32',
        "tolerance for e5, a format the kernel does not run in",
    ),
    ("rows = [1, 7, 33, 128]", "rows = []", "sweeps over no values"),
]


@pytest.mark.parametrize(("old", "new", "message"), RULES)
def test_each_rule_names_the_file_line_and_key(tmp_path, old, new, message):
    assert old in SCALE, old
    toml = SCALE.replace(old, new, 1)
    with pytest.raises(ManifestError) as error:
        load_manifest(kernel(tmp_path, toml))
    text = str(error.value)
    assert message in text, text
    m = re.match(r"(.+kernel\.toml):(\d+): ", text)
    assert m, text
    # The line is one of the edited lines (a deleted line: the table it was in).
    edited = {line_of(toml, line) for line in new.splitlines() if line.strip()}
    assert int(m.group(2)) in (edited or {line_of(toml, "[abi]")}), text


def test_a_missing_program_and_a_missing_section_are_errors(tmp_path):
    directory = kernel(tmp_path, SCALE)
    (directory / "scale.asm").unlink()
    with pytest.raises(ManifestError, match="the program scale.asm is not in the kernel directory"):
        load_manifest(directory)
    no_golden = re.sub(r"\[golden\]\n.*?\n\n", "", SCALE, flags=re.S)
    with pytest.raises(
        ManifestError, match=r"kernel.toml:1: kernel.toml needs a \[golden\] section"
    ):
        load_manifest(kernel(tmp_path, no_golden, "other"))


def test_formats_come_from_the_isa_enum():
    import ipu_harness.manifest as manifest

    source = Path(manifest.__file__).read_text()
    for name in ("e1", "e7", "int32"):
        assert f'"{name}"' not in source, f"{name} restated in manifest.py"


def test_derived_values_and_lr_presets_and_scratch(tmp_path):
    toml = SCALE.replace(
        "[abi]\n",
        "[scratch.log2e]\nshape = [1, 128]\nfill = 1.4426950408889634\n\n"
        '[derived]\nhalf = "rows // 2"\n\n[abi]\nlog2e = "CR6"\nhalf = "LR3"\n',
    )
    m = load_manifest(kernel(tmp_path, toml))
    assert m.tensors["log2e"].role == "scratch" and m.tensors["log2e"].fill == pytest.approx(
        1.4427, 1e-4
    )
    assert m.derived["half"] == "rows // 2" and m.abi["half"] == "LR3"
