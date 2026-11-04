"""The kernel page's data: the program's header and the ABI table (E06-21)."""

from __future__ import annotations

import pytest
from scale_kernel import write_scale

from ipu_harness.kernel_page import abi_table, design_notes, page_data
from ipu_harness.manifest import load_manifest


@pytest.mark.parametrize(
    ("program", "notes"),
    [
        (
            "{#- k.asm -- what it does.\n\n    Design notes\n    - one\n-#}\nHALT ;;\n",
            "k.asm -- what it does.\n\nDesign notes\n- one",
        ),
        ("\n{# k.asm #}\nHALT ;;\n", "k.asm"),
        ("HALT ;;\n{#- late -#}\n", None),  # not leading
        ("{#- -#}\nHALT ;;\n", None),  # empty
        ("HALT ;;\n", None),
    ],
)
def test_design_notes_are_the_leading_comment(program, notes):
    assert design_notes(program) == notes


def test_the_abi_table_names_each_binding_its_kind(tmp_path):
    m = write_scale(tmp_path, with_reference=False)
    toml = (m.directory / "kernel.toml").read_text()
    toml = toml.replace("[derived]\n", '[derived]\nhalf = "rows // 2"\n')
    toml = toml.replace("[abi]\n", '[abi]\nhalf = "CR8"\n')
    (m.directory / "kernel.toml").write_text(toml)
    m = load_manifest(m.directory)
    rows = {r["name"]: (r["register"], r["kind"], r["when"]) for r in abi_table(m)}
    assert rows == {
        "half": ("CR8", "derived", None),
        "x": ("CR2", "region", None),
        "y": ("CR3", "region", None),
        "rows": ("CR4", "int", None),
        "s": ("CR5", "element scalar", None),
        "qf": ("CR6", "derived", None),
    }
    data = page_data(m)
    assert data["design_notes"].startswith("scale.asm -- y = x * s")
    assert data["params"]["s"] == {"type": "float", "unit": "count"}
    assert data["tolerance"] == {"fp32": {"y": "EXACT"}, "e4": {"y": "EXACT"}}
