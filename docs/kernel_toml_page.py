"""The generated parts of the ``kernel.toml`` reference page (E06-21).

    bazel run //docs:kernel_toml_page -- "$PWD/out"

writes two Markdown snippets into the given directory:

* ``schema.md``: the schema section of the :mod:`ipu_harness.manifest`
  docstring, which is what the loader checks, followed by a table of every key
  of every table from ``manifest.SCHEMA_KEYS``, the tuples the loader accepts;
* ``registry.md``: the ``registry.toml`` example of :mod:`ipu_harness.family`
  and its entry keys (``family.ENTRY_KEYS``).

``reference/kernel-toml.md`` holds the hand-written intro and includes both
through ``pymdownx.snippets``, as the command-line tools page does. A key the
loader accepts but the docstring never names fails the generator, and so the
docs build and ``test_kernel_toml_page``.
"""

from __future__ import annotations

import inspect
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from ipu_harness import family, manifest

#: Sphinx roles the docstrings use: ``:func:`a.b``` reads as `a.b`.
_ROLE = re.compile(r":(?:func|data|mod|class|meth|attr):`~?([^`]+)`")


def markdown(text: str) -> str:
    """Docstring markup as Markdown.

    Args:
        text: Text with double backticks and Sphinx roles.

    Returns:
        The text with each of them as single-backtick code.
    """
    return _ROLE.sub(r"`\1`", text.replace("``", "`"))


def _paragraphs(lines: list[str]) -> list[str]:
    """Indented body lines as paragraphs, each on one line."""
    out: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return out


def schema_section(doc: str) -> str:
    """The ``Schema`` section of a docstring as Markdown.

    A line at the left margin followed by indented lines is a table's heading
    and its description; a paragraph at the margin with nothing indented under
    it is prose.

    Args:
        doc: The module docstring.

    Returns:
        Markdown: one ``###`` heading per table, its text, then the prose.

    Raises:
        ValueError: The docstring has no ``Schema`` section.
    """
    lines = inspect.cleandoc(doc).splitlines()
    try:
        i = lines.index("Schema") + 2  # the title and its underline
    except ValueError:
        raise ValueError("the docstring has no 'Schema' section") from None
    parts: list[str] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif i + 1 < len(lines) and lines[i + 1][:1].isspace() and lines[i + 1].strip():
            parts.append(f"### {markdown(line)}\n")  # a table and its description
            i += 1
            body = []
            while i < len(lines) and (not lines[i].strip() or lines[i][0].isspace()):
                body.append(lines[i])
                i += 1
            parts += [markdown(text) + "\n" for text in _paragraphs(body)]
        else:
            body = []
            while i < len(lines) and lines[i].strip() and not lines[i][0].isspace():
                body.append(lines[i])
                i += 1
            parts.append(markdown(" ".join(body)) + "\n")
    return "\n".join(parts)


def _table(rows: Iterable[tuple[str, str]], header: tuple[str, str]) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines) + "\n"


def keys_table(schema: Mapping[str, tuple[str, ...]]) -> str:
    """Every table and its keys, as a Markdown table.

    Args:
        schema: Table name to its keys (``manifest.SCHEMA_KEYS``).

    Returns:
        The table.
    """
    rows = ((f"`{table}`", ", ".join(f"`{key}`" for key in keys)) for table, keys in schema.items())
    return _table(rows, ("Table", "Keys"))


def _example(doc: str) -> str:
    """The ``::`` literal block of a docstring, as a fenced TOML block."""
    lines = inspect.cleandoc(doc).splitlines()
    start = next(i for i, line in enumerate(lines) if line.endswith("::")) + 1
    body: list[str] = []
    for line in lines[start:]:
        if line and not line[0].isspace():
            break
        body.append(line)
    text = inspect.cleandoc("\n".join(body))
    return f"```toml\n{text}\n```\n"


def undocumented(doc: str, schema: Mapping[str, tuple[str, ...]]) -> list[str]:
    """The keys of ``schema`` a docstring does not describe.

    Args:
        doc: The docstring.
        schema: Table name to its keys.

    Returns:
        ``"<table> <key>"`` for each key never written as ````key````.
    """
    return [
        f"{table} {key}"
        for table, keys in schema.items()
        for key in keys
        if f"``{key}``" not in doc
    ]


def schema_page() -> str:
    """``schema.md``: the schema text, then the table of keys.

    Raises:
        ValueError: The loader accepts a key the docstring does not describe.
    """
    missing = undocumented(manifest.__doc__ or "", manifest.SCHEMA_KEYS)
    if missing:
        raise ValueError(
            f"ipu_harness.manifest's docstring does not describe: {', '.join(missing)}"
        )
    return (
        schema_section(manifest.__doc__ or "")
        + "\n## Every key\n\n"
        + "The tables and keys the loader accepts; any other key is an error that names "
        "the closest valid one.\n\n" + keys_table(manifest.SCHEMA_KEYS)
    )


def registry_page() -> str:
    """``registry.md``.

    Returns:
        The ``registry.toml`` example of :mod:`ipu_harness.family`, then its
        entry keys.
    """
    keys = ", ".join(f"`{key}`" for key in family.ENTRY_KEYS)
    return _example(family.__doc__ or "") + f"\nA `[[kernel]]` entry takes {keys}.\n"


def main(argv: list[str] | None = None) -> int:
    """Write the snippets into the directory given.

    Args:
        argv: ``[OUTPUT_DIR]``; default the command line.

    Returns:
        The exit code: 0, or 2 for a usage error.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: kernel_toml_page.py OUTPUT_DIR", file=sys.stderr)
        return 2
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema.md").write_text(schema_page(), encoding="utf-8")
    (out / "registry.md").write_text(registry_page(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
