"""The Reference model API page's generated parts (E09-16).

Every public symbol and every schema field is rendered, a schema change
changes the page with no doc edit, and How-to: use the reference model from
cocotb links each call it shows to its entry on the page.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import re
from pathlib import Path

import pytest

import ipu_emu.commit
import ipu_emu.reference
from docs import reference_api_page as page
from ipu_emu.reference import ReferenceModel

ROOT = Path(__file__).parents[1]
API_PAGE = ROOT / "docs" / "content" / "reference" / "reference-model-api.md"
HOW_TO = ROOT / "docs" / "content" / "how-to" / "reference-model-cocotb.md"
EXAMPLE = ROOT / "docs" / "examples" / "test_reference_model_cocotb.py"

#: Headings the hand-written page gives an id the how-to may link.
PAGE_ANCHORS = {"api", "commit-record-schema", "faults", "compatibility"}


@pytest.fixture(scope="module")
def api() -> str:
    return page.render_api()


def _ids(text: str) -> list[str]:
    return re.findall(r"\{ #([\w-]+) \}", text)


def test_every_name_of_the_stable_module_is_rendered(api):
    ids = set(_ids(api))
    assert set(ipu_emu.reference.__all__) <= ids
    members = {
        n
        for n, m in vars(ReferenceModel).items()
        if not n.startswith("_") and (inspect.isfunction(m) or isinstance(m, property))
    }
    assert members and {f"ReferenceModel-{m}" for m in members} <= ids


def test_every_record_type_is_rendered(api):
    ids = set(_ids(api))
    records = {n for n in ipu_emu.commit.__all__ if f"ipu_emu.commit.{n}" not in page.INTERNAL}
    assert {"CommitRecord", "SlotOp", "RegWrite", "XmemWrite", "FaultInfo", "Outcome"} <= records
    assert records <= ids
    assert "CommitCollector" not in ids  # internal


def test_anchors_are_unique(api):
    ids = _ids(api)
    assert len(ids) == len(set(ids))
    assert set(ids) == set(page.documented_symbols().values())


def test_every_dataclass_field_has_a_meaning():
    missing = []
    for group in page.groups():
        for module, name in group.symbols:
            cls = getattr(module, name)
            if not dataclasses.is_dataclass(cls):
                continue
            meanings = dict(page.parse_docstring(cls.__doc__).sections.get("Attributes", []))
            missing += [
                f"{name}.{f.name}" for f in dataclasses.fields(cls) if f.name not in meanings
            ]
    assert missing == [], "add an Attributes entry to the docstring"


def test_signatures_come_from_the_code(api):
    assert (
        "ReferenceModel.run(max_cycles: int | None = None, *, trace: TraceWriter | None = None)"
        in api
    )
    assert "class ReferenceModel(fmt: ElementFormat = ElementFormat.INT8)" in api
    assert f"API_VERSION = '{ipu_emu.reference.API_VERSION}'" in api


def test_docstring_roles_become_links(api):
    assert "[`reset`](#ReferenceModel-reset)" in api
    assert "- [`ConfigurationError`](#ConfigurationError): " in api
    assert ":meth:" not in api and ":class:" not in api
    assert not re.search(r"(?<!`)``(?!`)", api)  # reST literals, not fences


def test_parse_docstring_sections():
    doc = page.parse_docstring(
        """Summary.

        More text.

        Args:
            name: First line
                continued.
            index: Other.

        Returns:
            A value
            on two lines.
        """
    )
    assert doc.text == "Summary.\n\nMore text."
    assert doc.sections["Args"] == [("name", "First line continued."), ("index", "Other.")]
    assert doc.sections["Returns"] == [("", "A value on two lines.")]


# -- schemas -------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(page.SCHEMA_FILES))
def test_every_schema_field_is_rendered_with_a_meaning(name):
    schema = page.load_schemas()[name]
    text = page.render_schema(name, schema, "x")
    fields = page.schema_fields(schema)
    assert {f.path for f in fields} >= set(schema["properties"])
    for field in fields:
        assert f"| `{field.path}` |" in text
        assert page._meaning(name, field), f"{name}: {field.path} has no meaning: describe it"


@pytest.mark.parametrize("name", sorted(page.SCHEMA_FILES))
def test_field_meanings_name_undescribed_fields(name):
    by_path = {f.path: f for f in page.schema_fields(page.load_schemas()[name])}
    for path in page.FIELD_MEANINGS.get(name, {}):
        assert path in by_path, f"{name}: FIELD_MEANINGS names {path}, which the schema lacks"
        assert not by_path[path].description, f"{name}: {path} is described in the schema"


def test_a_schema_change_changes_the_page():
    schema = page.load_schemas()["commit_record"]
    before = page.render_schema("commit_record", schema, "x")
    changed = copy.deepcopy(schema)
    changed["$id"] = changed["$id"].rsplit("/", 1)[0] + "/1.1.0"
    changed["properties"]["retire_tick"] = {"type": "integer", "description": "A new field."}
    changed["properties"]["cycle"]["description"] = "Changed meaning."
    del changed["properties"]["arith_events"]
    changed["$defs"]["slot_op"]["properties"]["lane"] = {"type": "string", "description": "L."}
    after = page.render_schema("commit_record", changed, "x")
    assert after != before
    assert "version `1.1.0`" in after
    assert "| `retire_tick` | integer | no | A new field. |" in after
    assert "| `cycle` | integer (≥ 0) | yes | Changed meaning. |" in after
    assert "| `slots[].lane` | string | no | L. |" in after
    assert "`arith_events` |" not in after


def test_schema_rules_are_rendered():
    rules = page.schema_rules(page.load_schemas()["commit_record"])
    assert rules == [
        'When `outcome` is `"fault"`: `fault` is an object, `reg_writes` is empty, '
        "`xmem_writes` is empty. Otherwise `fault` is `null`."
    ]


def test_a_field_of_one_alternative_names_its_form():
    fields = {f.path: f for f in page.schema_fields(page.load_schemas()["commit_record"])}
    assert fields["reg_writes[].register"].form is None  # in both forms
    assert fields["reg_writes[].index"].form == (1, 2)
    assert fields["reg_writes[].data"].form == (2, 2)


def test_faults_mark_f8_not_modelled():
    rows = {
        line.split("]")[0].lstrip("| ["): line for line in page.render_faults().splitlines()[2:]
    }
    assert sorted(rows) == [f"F-{n}" for n in range(1, 10)]
    assert "**Not modelled**" in rows["F-8"] and "OQ-28" in rows["F-8"]
    assert sum("Not modelled" in row for row in rows.values()) == 1


# -- the page and the how-to ------------------------------------------------------


def test_write_all_writes_every_snippet_the_page_includes(tmp_path):
    written = {p.name for p in page.write_all(tmp_path)}
    snippets = {n for n in written if n.endswith(".md")}
    included = set(
        re.findall(r'^--8<-- "generated/reference-api/([\w.-]+)"$', API_PAGE.read_text(), re.M)
    )
    assert included == snippets
    assert set(page.SCHEMA_FILES.values()) <= written
    assert (
        (tmp_path / "version.md")
        .read_text()
        .startswith(f"API version **`{ipu_emu.reference.API_VERSION}`**")
    )


def _shown_code() -> str:
    """The example sections the how-to includes."""
    text = EXAMPLE.read_text(encoding="utf-8")
    return "\n".join(re.findall(r"# --8<-- \[start:[\w-]+\]\n(.*?)# --8<-- \[end:", text, re.S))


def _used(symbols: dict[str, str], code: str) -> dict[str, set[str]]:
    """Each call the code makes -> the anchors it may link to."""
    used: dict[str, set[str]] = {}
    for qualname, anchor in symbols.items():
        if "." in qualname:
            member = qualname.rsplit(".", 1)[1]
            pattern = rf"\.{member}\("
            if qualname == "ReferenceModel.status":
                pattern = r"\.status\b"
        elif qualname[0].islower():  # a function: a call, not a dict key
            pattern = rf"\b{qualname}\("
        else:
            pattern = rf"\b{qualname}\b"
        if re.search(pattern, code):
            key = qualname.rsplit(".", 1)[-1] if "." in qualname else qualname
            used.setdefault(key, set()).add(anchor)
    return used


def test_the_cocotb_how_to_links_every_call_it_uses():
    used = _used(page.documented_symbols(), _shown_code())
    assert {"ReferenceModel", "step", "run", "to_json", "TraceWriter"} <= set(used)
    how_to = HOW_TO.read_text(encoding="utf-8")
    linked = set(re.findall(r"reference-model-api\.md#([\w-]+)\)", how_to))
    missing = sorted(name for name, anchors in used.items() if not anchors & linked)
    assert missing == [], "link each call to its entry on the Reference model API page"


def test_the_how_to_links_only_existing_anchors():
    how_to = HOW_TO.read_text(encoding="utf-8")
    linked = set(re.findall(r"reference-model-api\.md#([\w-]+)\)", how_to))
    known = set(page.documented_symbols().values()) | PAGE_ANCHORS
    assert linked and linked <= known


def test_every_name_the_model_re_exports_is_listed(api):
    ids = set(_ids(api))
    names = page.reexported(ipu_emu.reference)
    assert names  # TraceWriter and read_trace since API 1.2.0
    assert set(names) <= ids
