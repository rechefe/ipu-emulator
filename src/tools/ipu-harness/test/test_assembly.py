"""Assembly with the case as template values (E06-07)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ipu_as.errors import TemplateError
from ipu_harness.assembly import AssemblyCache, assemble, template_values
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.params import plan_case
from ipu_isa.formats import ElementFormat as F

HARNESS = Path(__file__).resolve().parents[1] / "src" / "ipu_harness"
KERNELS = Path(__file__).resolve().parents[2] / "ipu-kernels" / "src" / "ipu_kernels"

TOML = """\
[kernel]
op = "scale"
formats = ["fp32", "int8", "e4"]

[params]
rows = { type = "int", min = 1, default = 4 }
s    = { type = "float", default = 2.0 }

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[cases.default]
"""

#: The program reads its registers by [abi] name and nothing else.
SCALE_ASM = """\
    SET lr0 {{ cr.x }} ;;
    SET lr1 {{ cr.y }} ;;
    SET lr2 {{ cr.rows }} ;;
    BKPT ;;
"""


def kernel(tmp_path: Path, asm: str, name: str = "scale") -> KernelManifest:
    directory = tmp_path / "elementwise" / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text(asm)
    (directory / "kernel.toml").write_text(TOML)
    return load_manifest(directory)


def test_cr_names_render_and_the_program_is_the_same_in_every_format(tmp_path):
    m = kernel(tmp_path, SCALE_ASM)
    results = [assemble(plan_case(m, {}, fmt)) for fmt in (F.FP32, F.INT8, F.E4)]
    assert "SET lr0 CR2 ;;" in results[0].rendered and "SET lr2 CR4 ;;" in results[0].rendered
    assert results[0].words == results[1].words == results[2].words
    assert len(results[0].words) == 4


def test_the_template_reads_params_format_and_shapes(tmp_path):
    asm = (
        "{% for i in range(rows) %}    BKPT ;;\n{% endfor %}"
        "{% if format == 'e4' %}    BKPT ;;\n{% endif %}"
        "{% if shapes['x'] == [rows, 128] %}    BKPT ;;\n{% endif %}"
    )
    m = kernel(tmp_path, asm)
    assert len(assemble(plan_case(m, {"rows": 2}, F.FP32)).words) == 3
    assert len(assemble(plan_case(m, {"rows": 2}, F.E4)).words) == 4
    values = template_values(plan_case(m, {"rows": 2}, F.E4))
    assert dict(values) == {
        "rows": 2,
        "s": 2.0,
        "format": "e4",
        "shapes": {"x": [2, 128], "y": [2, 128]},
    }


@pytest.mark.parametrize(
    "line, message",
    [
        ("    SET lr0 {{ cols }} ;;\n", "'cols' is undefined"),
        ("    SET lr0 {{ cr.z }} ;;\n", "z"),
    ],
)
def test_an_undefined_name_is_located_in_the_program(tmp_path, line, message):
    m = kernel(tmp_path, "    BKPT ;;\n" + line)
    with pytest.raises(TemplateError, match=message) as error:
        assemble(plan_case(m, {}, F.FP32))
    assert error.value.location is not None and error.value.location.line == 2
    assert error.value.location.source == "scale.asm"


def test_a_program_cannot_declare_parameters_of_its_own(tmp_path):
    # kernel.toml is the declaration: an ipu:params header is not read, so a
    # name it declares is undefined.
    asm = "{# ipu:params\nextra: int = 3\n#}\n    SET lr0 {{ cr.x }} ;;\n{% if extra %}    BKPT ;;\n{% endif %}"
    m = kernel(tmp_path, asm)
    with pytest.raises(TemplateError, match="'extra' is undefined"):
        assemble(plan_case(m, {}, F.FP32))


def test_the_cache_encodes_each_distinct_rendered_program_once(tmp_path):
    cache = AssemblyCache()
    m = kernel(tmp_path, SCALE_ASM)
    first = assemble(plan_case(m, {"rows": 4}, F.FP32), cache)
    assert assemble(plan_case(m, {"rows": 4}, F.INT8), cache) is first
    # The program does not use rows: another rows renders alike, so no new assembly.
    assert assemble(plan_case(m, {"rows": 9}, F.FP32), cache) is first
    assert cache.assembled == 1
    uses_rows = kernel(tmp_path, "{% for i in range(rows) %}    BKPT ;;\n{% endfor %}", "loop")
    assemble(plan_case(uses_rows, {"rows": 2}, F.FP32), cache)
    assemble(plan_case(uses_rows, {"rows": 3}, F.FP32), cache)
    assemble(plan_case(uses_rows, {"rows": 2}, F.E4), cache)
    assert cache.assembled == 3


def test_includes_resolve_in_the_kernel_and_family_folders(tmp_path):
    m = kernel(tmp_path, '{% include "family_tail.asm.j2" %}\n{% include "local.asm.j2" %}\n')
    (m.directory.parent / "family_tail.asm.j2").write_text("    BKPT ;;\n")
    (m.directory / "local.asm.j2").write_text("    SET lr0 {{ cr.x }} ;;\n")
    result = assemble(plan_case(m, {}, F.FP32))
    assert result.rendered.splitlines() == ["    BKPT ;;", "    SET lr0 CR2 ;;"]


def test_neither_the_harness_nor_the_kernel_library_imports_jinja2():
    imports = re.compile(r"^\s*(import|from)\s+jinja2\b", re.MULTILINE)
    for root in (HARNESS, KERNELS):
        for path in sorted(root.rglob("*.py")):
            assert not imports.search(path.read_text(encoding="utf-8")), path
