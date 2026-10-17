"""kernel.toml: the scale kernel loads into the expected view; each schema rule has a failing fixture."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from ipu_harness.manifest import ManifestError, load_manifest
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
    assert m.golden.tolerance[ElementFormat.E4].name == "FP8_ULP"
    assert m.golden.tolerance[ElementFormat.E4].ulps == 1
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
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.tensors["x"].row = 3  # type: ignore[misc]
    for mapping in (
        m.params,
        m.cases["default"].values,
        m.golden.tolerance,
        m.cases["mine"].inputs,
    ):
        with pytest.raises(TypeError):
            mapping["new"] = None  # type: ignore[index]


def test_a_registry_entry_fixes_parameters_and_narrows_formats(tmp_path):
    entry = {"name": "scale_8", "formats": ["fp32", "e4"], "params": {"rows": 8}}
    m = load_manifest(kernel(tmp_path, SCALE), overrides=entry, source="registry.toml [scale_8]")
    assert (m.name, m.asm.name, m.formats) == (
        "scale_8",
        "scale.asm",
        (ElementFormat.FP32, ElementFormat.E4),
    )
    assert dict(m.fixed) == {"rows": 8}
    # rows has no template default: the fixed value stands in.
    assert (m.params["rows"].choices, m.params["rows"].default) == ((8,), 8)
    # The template's cases run with the entry's value.
    assert m.cases["default"].values["rows"] == 8 and m.cases["sweep"].values["rows"] == 8


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"params": {"rows": 999}}, "rows = 999 is outside its domain"),
        ({"params": 3}, "params 3 must be a table"),
        ({"params": {"cols": 1}}, "params.cols is no parameter of the template"),
        ({"name": 5}, "name 5 must be an identifier"),
        ({"name": ""}, "name '' must be an identifier"),
        ({"formats": ["e5"]}, "must be a non-empty subset"),
        ({"rows": 4}, "unknown overrides ['rows']"),
    ],
)
def test_bad_overrides_name_their_source(tmp_path, overrides, message):
    with pytest.raises(ManifestError) as error:
        load_manifest(kernel(tmp_path, SCALE), overrides=overrides, source="registry.toml [x]")
    assert str(error.value).startswith("registry.toml [x]: override of ")
    assert message in str(error.value)


def test_the_default_reference_is_reference_py_when_present(tmp_path):
    toml = SCALE.replace('reference = "reference.py:reference"\n', "")
    without = load_manifest(kernel(tmp_path, toml))
    assert without.golden.reference is None
    with_file = kernel(tmp_path, toml, "scale2")
    (with_file / "reference.py").write_text("")
    assert load_manifest(with_file).golden.reference == "reference.py:reference"


def line_of(toml: str, needle: str) -> int:
    lines = [i for i, line in enumerate(toml.splitlines(), start=1) if needle in line]
    assert len(lines) == 1, (needle, lines)
    return lines[0]


P = 'rows = { type = "int", min = 1, max = 256 }'
S = 's    = { type = "float" }'

# Each rule: an edit of the scale example (old -> new), the message, and the line it must name.
RULES = [
    # query_required (routing) is a bool.
    (P, P.replace(" }", ", query_required = 1 }"), "must be bool, not 1", "rows = {"),
    # Tolerance tiers (E06-10): a known name, or an override with its why.
    ('e4 = "FP8_ULP" }', 'e4 = "LOOSE" }', "'LOOSE' is no tier", 'e4 = "LOOSE"'),
    (
        'e4 = "FP8_ULP" }',
        'e4 = { tier = "FP8_ULP", ulps = 2 } }',
        "overrides a tier: it needs 'tier' and a 'why'",
        "tolerance = {",
    ),
    (
        'e4 = "FP8_ULP" }',
        'e4 = { tier = "FP8_ULP", ulps = -1, why = "x" } }',
        "-1 must not be negative",
        "tolerance = {",
    ),
    (
        'fp32 = "EXACT"',
        'fp32 = { tier = "FP32_ORDER", ulps = 2, why = "x" }',
        "is never read for fp32 at this tier",
        "tolerance = {",
    ),
    (
        'e4 = "FP8_ULP" }',
        'e4 = { tier = "FP8_ULP", rtol = 0.1, why = "x" } }',
        "is never read for e4 at this tier",
        "tolerance = {",
    ),
    # An input's range (E06-08): a non-empty [lo, hi], per format only for the kernel's formats.
    (
        '[inputs.x]\nshape = ["rows", 128]',
        '[inputs.x]\nshape = ["rows", 128]\nrange = [1]',
        "must be [lo, hi]",
        "range = [1]",
    ),
    (
        '[inputs.x]\nshape = ["rows", 128]',
        '[inputs.x]\nshape = ["rows", 128]\nrange = [3, 2]',
        "is empty",
        "range = [3, 2]",
    ),
    (
        '[inputs.x]\nshape = ["rows", 128]',
        '[inputs.x]\nshape = ["rows", 128]\nrange = { e5 = [0, 1] }',
        "range for e5",
        "range = { e5",
    ),
    (
        '[outputs.y]\nshape = ["rows", 128]',
        '[outputs.y]\nshape = ["rows", 128]\nrange = [0, 1]',
        "unknown key 'range' in [outputs.y]",
        "range = [0, 1]",
    ),
    (
        '[inputs.x]\nshape = ["rows", 128]',
        '[inputs.x]\nshape = ["rows", 128]\nrange = { int8 = [0.2, 0.8] }',
        "[0.2, 0.8] holds no int8 value",
        "range = { int8",
    ),
    (
        '[inputs.x]\nshape = ["rows", 128]',
        '[inputs.x]\nshape = ["rows", 128]\nrange = [-500, 500]',
        "[inputs.x] range [-500, 500] is outside int8's [-128, 127]",
        "range = [-500, 500]",
    ),
    ('s    = "CR5"', 's    = "CR1"', "[abi] binds CR1, a hardwired constant", 's    = "CR1"'),
    ('s    = "CR5"', 's    = "CR0"', "[abi] binds CR0, a hardwired constant", 's    = "CR0"'),
    ('s    = "CR5"', 's    = "CR16"', "is no CR or LR register", 's    = "CR16"'),
    ('x    = "CR2"', 'x    = "CR0"', "is read through CR0 (always 0); pin it", 'x    = "CR0"'),
    (
        's    = "CR5"',
        's    = { register = "CR5", offset = 1 }',
        "only a tensor's binding takes an offset",
        "s    = { register",
    ),
    (
        'y    = "CR3"',
        'y    = { register = "CR3", shift = 1 }',
        "unknown key 'shift'",
        "y    = { register",
    ),
    ('y    = "CR3"', "y    = { offset = 1 }", "needs 'register'", "y    = { offset"),
    (
        '[outputs.y]\nshape = ["rows", 128]',
        '[outputs.y]\nshape = ["rows", 128]\nafter = "z"',
        "[outputs.y] after 'z' is no tensor",
        'after = "z"',
    ),
    (
        '[outputs.y]\nshape = ["rows", 128]',
        '[outputs.y]\nshape = ["rows", 128]\nafter = "y"',
        "after forms a cycle through 'y'",
        'after = "y"',
    ),
    (
        '[outputs.y]\nshape = ["rows", 128]',
        '[outputs.y]\nshape = ["rows", 128]\nrow = 3\nafter = "x"',
        "is pinned by row; it takes no 'after'",
        'after = "x"',
    ),
    (
        '[outputs.y]\nshape = ["rows", 128]',
        '[outputs.y]\nshape = ["rows", 128]\nrow = "rowz + 1"',
        "names no parameter or derived value: ['rowz']",
        'row = "rowz',
    ),
    ('s    = "CR5"', 's    = "cr05"', "is no CR or LR register", 's    = "cr05"'),
    ('rows = "CR4"', 'rows = "LR16"', "is no CR or LR register", 'rows = "LR16"'),
    ('s    = "CR5"', 's    = "LR5"', "an element scalar (a float); it binds a CR", 's    = "LR5"'),
    ('s    = "CR5"', 's    = "CR4"', "both bind CR4", 's    = "CR4"'),
    ('s    = "CR5"', 's    = "CR5"\nz    = "CR6"', "[abi] z names no parameter", 'z    = "CR6"'),
    ('y    = "CR3"\n', "", "[abi] does not bind tensor 'y'", "[outputs.y]"),
    ("[cases.default]", "[cases.first]", "needs [cases.default]", "[cases.first]"),
    (
        "rows = 4",
        "rowz = 4",
        "unknown key 'rowz' in [cases.default]; did you mean 'rows'?",
        "rowz = 4",
    ),
    ("rows = 4", "rows = 0", "rows = 0 is outside its domain", "rows = 0"),
    ("rows = 4", 'rows = "four"', "[cases.default] rows must be int", 'rows = "four"'),
    ("rows = 4", "rows = true", "[cases.default] rows must be int", "rows = true"),
    (
        "[golden]",
        "[goldn]",
        "unknown key 'goldn' in kernel.toml; did you mean 'golden'?",
        "[goldn]",
    ),
    (
        'formats = ["fp32", "int8", "e4"]',
        'formats = ["fp32", "fp16"]',
        "unknown element format 'fp16'",
        '"fp16"',
    ),
    (
        'formats = ["fp32", "e4"]',
        'formats = ["fp32", "e5"]',
        "formats ['e5'] are not the kernel's",
        '"e5"',
    ),
    ('formats = ["fp32", "e4"]', "formats = []", "formats must not be empty", "formats = []"),
    (
        'inputs = { x = "data/x.npy" }',
        'inputs = { x = "../x.npy" }',
        "must be a file path inside",
        "../x.npy",
    ),
    (
        'inputs = { x = "data/x.npy" }',
        'inputs = { x = "/tmp/x.npy" }',
        "must be a file path inside",
        "/tmp/x.npy",
    ),
    (
        'inputs = { x = "data/x.npy" }',
        'inputs = { x = "data/" }',
        "must be a file path inside",
        '"data/"',
    ),
    (
        'inputs = { x = "data/x.npy" }',
        'inputs = { y = "data/y.npy" }',
        "'y' is not one of the tensors ['x']",
        "inputs = { y",
    ),
    (
        'expected = { y = "data/y.npy" }',
        'expected = { y = "../y.npy" }',
        "must be a file path inside",
        "../y.npy",
    ),
    (
        'expected = { y = "data/y.npy" }',
        'expected = { x = "x.npy" }',
        "'x' is not one of the tensors ['y']",
        "expected = { x",
    ),
    (
        "s = 2.0\n\n[cases.one_row]",
        's = 2.0\ngenerator = "../../gen.py:make"\n\n[cases.one_row]',
        "must be ../<file> directly in the family's folder",
        "../../gen.py",
    ),
    (
        "s = 2.0\n\n[cases.one_row]",
        's = 2.0\ngenerator = "../sub/gen.py:make"\n\n[cases.one_row]',
        "must be ../<file> directly in the family's folder",
        "../sub/gen.py",
    ),
    (
        "s = 2.0\n\n[cases.one_row]",
        's = 2.0\ngenerator = "gen.txt:make"\n\n[cases.one_row]',
        "must be 'file.py:function'",
        "gen.txt",
    ),
    (
        "s = 2.0\n\n[cases.one_row]",
        's = 2.0\ntier = "weekly"\n\n[cases.one_row]',
        "'weekly' must be default or nightly",
        "weekly",
    ),
    (
        'formats = ["fp32", "int8", "e4"]',
        'formats = ["fp32", "int8", "e4"]\nstages = "/abs.py:run"',
        "must be a file path inside",
        "/abs.py",
    ),
    (
        'reference = "reference.py:reference"',
        'reference = "../ref.py:reference"',
        "must be a file path inside",
        "../ref.py",
    ),
    (
        'reference = "reference.py:reference"',
        'reference = "reference.py"',
        "must be 'file.py:function'",
        'reference = "reference.py"',
    ),
    (
        'reference = "reference.py:reference"',
        'reference = "r.py:f"\nop = "scale2"',
        "both a reference and an op",
        'op = "scale2"',
    ),
    (P, 'rows = { type = "integer" }', "type 'integer' must be one of", "integer"),
    (P, 'rows = { type = "int", maxx = 2 }', "did you mean 'max'?", "maxx"),
    (P, 'rows = { type = "int", unit = "bytes" }', "counts bytes", '"bytes"'),
    (P, 'rows = { type = "int", min = 5, max = 1 }', "min 5 is above max 1", "min = 5"),
    (
        P,
        'rows = { type = "int", multiple_of = 0 }',
        "multiple_of must be at least 1",
        "multiple_of = 0",
    ),
    (
        S,
        's    = { type = "float", multiple_of = 2 }',
        'multiple_of needs type = "int"',
        "multiple_of = 2",
    ),
    (S, 's    = { type = "str", min = 1 }', "min and max need type int or float", 'type = "str"'),
    (S, 's    = { type = "float", choices = [] }', "choices must not be empty", "choices = []"),
    (
        P,
        'rows = { type = "int", max = 9, choices = [4, 99] }',
        "rows = 99 is outside its domain",
        "choices = [4, 99]",
    ),
    (S, 's    = { type = "float", max = nan }', "must be finite", "nan"),
    ("s = 2.0\n\n[cases.one_row]", "s = inf\n\n[cases.one_row]", "must be finite", "s = inf"),
    (S, 'tier = { type = "int" }', "'tier' must be an identifier", 'tier = { type = "int" }'),
    (S, '_s = { type = "float" }', "'_s' must be an identifier", "_s = {"),
    (
        'shape = ["rows", 128]\n\n[outputs.y]',
        'shape = ["cols", 128]\n\n[outputs.y]',
        "names no parameter or derived value",
        '"cols"',
    ),
    (
        'shape = ["rows", 128]\n\n[outputs.y]',
        "shape = []\n\n[outputs.y]",
        "at least one dimension",
        "shape = []",
    ),
    (
        'shape = ["rows", 128]\n\n[outputs.y]',
        'shape = ["rows", 128]\nrow = -1\n\n[outputs.y]',
        "row -1 must not be negative",
        "row = -1",
    ),
    (
        "[outputs.y]\nshape",
        '[outputs.y]\nstore = "fp16"\nshape',
        "'fp16' must be element or accumulator",
        'store = "fp16"',
    ),
    ("[outputs.y]", "[outputs.x]", "'x' is declared twice", "[outputs.x]"),
    (
        "[abi]\n",
        '[derived]\ns = "rows"\n\n[abi]\n',
        "[derived] s is also a parameter",
        's = "rows"',
    ),
    (
        "[abi]\n",
        '[derived]\nh = "nope + 1"\n\n[abi]\n',
        "names no parameter or derived value: ['nope']",
        "nope",
    ),
    (
        "tolerance = { fp32",
        'tolerance = { e5 = "X", fp32',
        "tolerance for e5, a format the kernel does not run in or accumulate in",
        "e5 =",
    ),
    ("rows = [1, 7, 33, 128]", "rows = []", "sweeps over no values", "rows = []"),
    (
        "[abi]\n",
        '[derived]\nh = "rows ** 2"\n\n[abi]\n',
        "is outside the shape grammar",
        "rows ** 2",
    ),
    (
        "[abi]\n",
        '[derived]\nh = "min(rows)"\n\n[abi]\n',
        "min() takes at least 2 positional arguments",
        "min(rows)",
    ),
    (S, 'format = { type = "str" }', "'format' must be an identifier", "format = {"),
]


@pytest.mark.parametrize(("old", "new", "message", "needle"), RULES)
def test_each_rule_names_the_file_line_and_key(tmp_path, old, new, message, needle):
    assert old in SCALE, old
    toml = SCALE.replace(old, new, 1)
    with pytest.raises(ManifestError) as error:
        load_manifest(kernel(tmp_path, toml))
    text = str(error.value)
    assert message in text, text
    m = re.match(r"(.+kernel\.toml):(\d+): ", text)
    assert m, text
    assert int(m.group(2)) == line_of(toml, needle), text


def test_cr15_and_derived_shapes_load(tmp_path):
    toml = SCALE.replace('s    = "CR5"', 's    = "CR15"').replace(
        '[outputs.y]\nshape = ["rows", 128]',
        '[outputs.y]\nshape = ["blocks", "max(rows, 2) * 4"]\n\n[derived]\nblocks = "ceil(rows / 128)"',
    )
    m = load_manifest(kernel(tmp_path, toml))
    assert m.abi["s"] == "CR15" and m.tensors["y"].shape == ("blocks", "max(rows, 2) * 4")


def test_a_missing_program_section_or_file_and_bad_toml_are_errors(tmp_path):
    directory = kernel(tmp_path, SCALE)
    (directory / "scale.asm").unlink()
    with pytest.raises(ManifestError, match="the program scale.asm is not in the kernel directory"):
        load_manifest(directory)
    no_kernel = SCALE.split("\n\n", 1)[1]
    with pytest.raises(
        ManifestError, match=r"kernel.toml:1: kernel.toml needs a \[kernel\] section"
    ):
        load_manifest(kernel(tmp_path, no_kernel, "other"))
    with pytest.raises(ManifestError, match=r"kernel.toml:1: cannot read it"):
        load_manifest(tmp_path / "nowhere")
    with pytest.raises(ManifestError, match=r"kernel.toml:7: "):
        load_manifest(kernel(tmp_path, SCALE.replace("s    = {", "s    = {{"), "broken"))


def test_golden_is_optional(tmp_path):
    no_golden = re.sub(r"\[golden\]\n.*?\n\n", "", SCALE, flags=re.S)
    assert load_manifest(kernel(tmp_path, no_golden)).golden.reference is None


def test_formats_come_from_the_isa_enum(tmp_path):
    # Every ElementFormat is accepted by name, so manifest.py keeps no list of its own.
    toml = SCALE.replace(
        'formats = ["fp32", "int8", "e4"]', f"formats = {[f.value for f in ElementFormat]}"
    )
    toml = toml.replace('formats = ["fp32", "e4"]', 'formats = ["fp32"]')
    assert load_manifest(kernel(tmp_path, toml)).formats == tuple(ElementFormat)


def test_a_tier_override_keeps_its_why(tmp_path):
    toml = SCALE.replace(
        'fp32 = "EXACT"', 'fp32 = { tier = "FP32_ORDER", rtol = 3e-5, why = "measured on case X" }'
    )
    tier = load_manifest(kernel(tmp_path, toml)).golden.tolerance[ElementFormat.FP32]
    assert tier == ("FP32_ORDER", 3e-5, 1e-6, None, "measured on case X")


ACCUMULATED = SCALE.replace(
    '[outputs.y]\nshape = ["rows", 128]',
    '[outputs.y]\nshape = ["rows", 128]\nstore = "accumulator"',
)


def test_an_accumulator_format_may_name_a_tier(tmp_path):
    # INT8 accumulates in INT32: an accumulator output in an int8 case is compared there.
    toml = ACCUMULATED.replace(
        'fp32 = "EXACT", int8 = "EXACT", e4 = "FP8_ULP"', 'fp32 = "EXACT", int32 = "INT8_LSB"'
    )
    tiers = load_manifest(kernel(tmp_path, toml)).golden.tolerance
    assert tiers[ElementFormat.INT32].name == "INT8_LSB"


@pytest.mark.parametrize(
    ("toml", "unread"),
    [
        # Every output is an accumulator output: an E4 run compares it as FP32.
        (ACCUMULATED, "int8"),
        (ACCUMULATED.replace('int8 = "EXACT", ', ""), "e4"),
        # An element output is never compared in an accumulator format.
        (SCALE.replace('fp32 = "EXACT"', 'fp32 = "EXACT", int32 = "EXACT"'), "int32"),
    ],
)
def test_a_tier_no_output_is_compared_at_is_an_error(tmp_path, toml, unread):
    with pytest.raises(ManifestError, match=rf"tolerance for {unread} is never read"):
        load_manifest(kernel(tmp_path, toml))


def test_an_input_range_is_per_format(tmp_path):
    both = SCALE.replace(
        'shape = ["rows", 128]\n\n[outputs', 'shape = ["rows", 128]\nrange = [-2, 2]\n\n[outputs'
    )
    x = load_manifest(kernel(tmp_path, both)).tensors["x"]
    # A bare [lo, hi] holds in every format the kernel runs in.
    kernel_formats = (ElementFormat.FP32, ElementFormat.INT8, ElementFormat.E4)
    assert dict(x.value_range) == dict.fromkeys(kernel_formats, (-2.0, 2.0))
    table = SCALE.replace(
        'shape = ["rows", 128]\n\n[outputs',
        'shape = ["rows", 128]\nrange = { int8 = [-3, 3] }\n\n[outputs',
    )
    assert dict(load_manifest(kernel(tmp_path, table, "t")).tensors["x"].value_range) == {
        ElementFormat.INT8: (-3.0, 3.0)
    }


def test_derived_values_lr_presets_and_scratch(tmp_path):
    toml = SCALE.replace(
        "[abi]\n",
        "[scratch.log2e]\nshape = [1, 128]\nfill = 1.4426950408889634\n\n"
        '[derived]\nhalf = "rows // 2"\n\n[abi]\nlog2e = "CR6"\nhalf = "LR3"\n',
    )
    m = load_manifest(kernel(tmp_path, toml))
    assert m.tensors["log2e"].role == "scratch" and m.tensors["log2e"].fill == 1.4426950408889634
    assert m.derived["half"] == "rows // 2" and m.abi["half"] == "LR3"
    assert m.params["s"].type == "float" and m.cases["default"].values["s"] == 2.0
    assert isinstance(
        load_manifest(kernel(tmp_path, SCALE.replace("s = 2.0", "s = 2"), "ints"))
        .cases["default"]
        .values["s"],
        float,
    )


def test_only_the_loader_and_ipu_as_abi_read_kernel_toml():
    """load_manifest is the reader of kernel.toml; ipu-as --abi reads only its [abi] table,
    because ipu_as cannot import ipu_harness; ipu_harness.family reads registry.toml
    (E06-17) and hands each entry to load_manifest. Any other reader is a second schema."""
    root = Path(__file__).resolve().parents[4]
    if not (root / "src/tools").is_dir():
        pytest.skip("needs the workspace sources")
    allowed = {
        "src/tools/ipu-harness/src/ipu_harness/manifest.py",
        "src/tools/ipu-as-py/src/ipu_as/cli.py",
        "src/tools/ipu-harness/src/ipu_harness/family.py",
    }
    readers = []
    for path in sorted((root / "src/tools").glob("*/src/**/*.py")):
        text = path.read_text(encoding="utf-8")
        if "kernel.toml" in text and re.search(r"\btoml(?:lib|i)?\.loads?\(", text):
            readers.append(str(path.relative_to(root)))
    assert sorted(set(readers) - allowed) == []
