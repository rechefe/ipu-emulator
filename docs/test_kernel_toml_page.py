"""The ``kernel.toml`` reference page and the "Write a kernel" how-to (E06-21).

The reference is generated from the loader's docstring and key tuples, and a
key the loader accepts but the docstring does not describe fails the
generator; the how-to's program listing is the ``scale`` kernel's file, verbatim, and it
includes the kernel's ``kernel.toml`` and ``reference.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docs import kernel_toml_page as page
from docs.doc_blocks import fenced_blocks
from ipu_harness import family, manifest

REFERENCE = Path("docs/content/reference/kernel-toml.md")
HOWTO = Path("docs/content/how-to/write-a-kernel.md")
SCALE = Path("kernels/elementwise/scale")

DOC = """Intro.

Schema
------

``[kernel]`` (required)
    ``op`` (str): the operation. See :func:`a.b`.

    A second paragraph.

``[abi]``
    ``<name> = "CR2"``.

Prose at the margin,
over two lines.
"""


def test_the_schema_section_is_markdown_headings_and_text() -> None:
    assert page.schema_section(DOC).split("\n\n") == [
        "### `[kernel]` (required)",
        "`op` (str): the operation. See `a.b`.",
        "A second paragraph.",
        "### `[abi]`",
        '`<name> = "CR2"`.',
        "Prose at the margin, over two lines.\n",
    ]
    with pytest.raises(ValueError, match="no 'Schema' section"):
        page.schema_section("No schema here.")


def test_a_key_counts_as_described_only_in_its_own_tables_section() -> None:
    schema = {"[kernel]": ("op",), "[abi] table binding": ("op", "name"), "[cases]": ("x",)}
    # "op" is in [kernel]'s text, not [abi]'s; [abi] writes "<name> = ...", not "name".
    assert page.undocumented(DOC, schema) == [
        "[abi] table binding op",
        "[abi] table binding name",
        "[cases] (no section)",
    ]


def test_an_undescribed_key_fails_the_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    assert page.undocumented(manifest.__doc__ or "", manifest.SCHEMA_KEYS) == []
    schema = {**manifest.SCHEMA_KEYS, "[kernel]": (*manifest.SCHEMA_KEYS["[kernel]"], "budget")}
    assert page.undocumented(manifest.__doc__ or "", schema) == ["[kernel] budget"]
    monkeypatch.setattr(manifest, "SCHEMA_KEYS", schema)
    with pytest.raises(ValueError, match=r"does not describe: \[kernel\] budget"):
        page.schema_page()


def test_the_generated_snippets_hold_every_table_and_key(tmp_path: Path) -> None:
    assert page.main([str(tmp_path)]) == 0
    schema = (tmp_path / "schema.md").read_text(encoding="utf-8")
    for heading in ("### `[kernel]` (required)", "### `[abi]`", "### `[golden]` (optional)"):
        assert heading in schema
    assert "``" not in schema and ":func:" not in schema
    for table, keys in manifest.SCHEMA_KEYS.items():
        row = f"| `{table}` | " + ", ".join(f"`{k}`" for k in keys) + " |"
        assert row in schema
    registry = (tmp_path / "registry.md").read_text(encoding="utf-8")
    assert registry.startswith("```toml\n# kernels/projections/registry.toml\n[[kernel]]")
    assert all(f"`{key}`" in registry for key in family.ENTRY_KEYS)


def test_the_reference_page_includes_both_snippets() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    for name in ("schema.md", "registry.md"):
        assert f'--8<-- "generated/kernel-toml/{name}"' in text


def _listing(lang: str, contains: str) -> str:
    """The how-to's ``lang`` block that holds ``contains``."""
    found = [b.body for b in fenced_blocks(HOWTO, lang) if contains in b.body]
    assert len(found) == 1, f"{HOWTO}: {len(found)} {lang} blocks hold {contains!r}"
    return found[0]


def test_the_howto_program_listing_is_the_scale_kernels_file() -> None:
    # The asm block stays a copy: //docs:test_doc_asm_blocks assembles it.
    listing = _listing("asm", "row_loop:")
    assert listing == (SCALE / "scale.asm").read_text(encoding="utf-8"), (
        f"{HOWTO}: the scale.asm listing differs from {SCALE / 'scale.asm'}; copy the file in"
    )


@pytest.mark.parametrize(("name", "lang"), [("kernel.toml", "toml"), ("reference.py", "python")])
def test_the_howto_includes_the_scale_kernels_files(name: str, lang: str) -> None:
    """kernel.toml and reference.py are included from the kernel folder (E09-09):
    the block is one snippet line, which build_docs resolves at the file's repo path."""
    assert (SCALE / name).is_file()
    _listing(lang, f'--8<-- "{SCALE / name}"')
