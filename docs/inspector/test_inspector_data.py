"""What the Cycle Inspector page fetches is complete and valid (E08-16).

Tagged ``docs``: it builds every curated trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from ipu_gen import inspector_trace

DATA = Path("docs/inspector")
KERNELS = json.loads((DATA / "kernels.json").read_text(encoding="utf-8"))["kernels"]


@pytest.mark.parametrize("entry", KERNELS, ids=lambda entry: entry["id"])
def test_each_trace_validates_against_the_schema(entry):
    document = json.loads((DATA / f"{entry['id']}.json").read_text(encoding="utf-8"))
    jsonschema.validate(document, inspector_trace.schema())
    assert document["kernel"] == entry["kernel"]
    assert document["cycles"], "a trace with no cycles"


def test_the_isa_reference_is_there():
    isa = json.loads(Path("vscode-ipu-asm/generated/data/isa.json").read_text(encoding="utf-8"))
    assert isa["instructions"]
