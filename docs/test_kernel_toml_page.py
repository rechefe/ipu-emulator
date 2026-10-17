"""The ``kernel.toml`` reference page and the "Write a kernel" how-to (E06-21).

The reference is generated from the loader's docstring and key tuples, and a
key the loader accepts but the docstring does not describe fails the
generator; the how-to's listings are the ``scale`` kernel's files, verbatim.
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


@pytest.mark.parametrize(
    ("name", "lang", "contains"),
    [
        ("scale.asm", "asm", "row_loop:"),
        ("kernel.toml", "toml", "[kernel]"),
        ("reference.py", "python", "def reference("),
    ],
)
def test_the_howto_listings_are_the_scale_kernels_files(
    name: str, lang: str, contains: str
) -> None:
    listing = _listing(lang, contains)
    assert listing == (SCALE / name).read_text(encoding="utf-8"), (
        f"{HOWTO}: the {name} listing differs from {SCALE / name}; copy the file in"
    )
