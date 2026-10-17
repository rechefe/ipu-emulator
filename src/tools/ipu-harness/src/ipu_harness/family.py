"""A family's ``registry.toml``: shape variants over template kernels (E06-17, Q76).

::

    # kernels/projections/registry.toml
    [[kernel]]
    name = "proj_ffn1_192_p4"        # kernel name == Bazel target name
    template = "proj_p4"             # kernels/projections/proj_p4/ (an ordinary kernel folder)
    params = { k = 192, n_out = 384, n_tok = 64, n_tg = 1, activation = "silu" }
    doc = "Layer 4 FFN1"             # one line, for the kernel page and query output
    formats = ["fp32"]               # optional: a subset of the template's
    cases = ["default", "edge"]      # optional: the template cases it runs, default among them

An entry is its template loaded with those values as overrides
(:func:`ipu_harness.manifest.load_manifest`): each fixed parameter becomes the
only value its domain allows, which is the old exact-match ``supports``.
Errors name the line: ``kernels/projections/registry.toml:12: ...``.

A kernel is named by a folder (``kernels/<family>/<name>``) or by an entry
(``kernels/<family>/registry.toml::<name>``); :func:`load_kernel` takes either,
so the test entry, the CLI and the routing registry load both the same way.
Bazel reads the entries from a generated ``registry.bzl`` beside the TOML
(:func:`render_bzl`; a test fails when it is stale).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ipu_harness._toml_source import Locator, ManifestError, Reader, decode_error_line, where
from ipu_harness.manifest import TOML_NAME, KernelManifest, load_manifest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = [
    "BZL_NAME",
    "ENTRY_SEPARATOR",
    "REGISTRY_NAME",
    "Entry",
    "entry_ref",
    "load_entry",
    "load_kernel",
    "read_registry",
    "render_bzl",
]

REGISTRY_NAME = "registry.toml"
#: The generated file Bazel loads the entries from.
BZL_NAME = "registry.bzl"
#: Between a ``registry.toml`` path and an entry's name in a kernel reference.
ENTRY_SEPARATOR = "::"

_ENTRY_KEYS = ("name", "template", "params", "formats", "cases", "doc")
_ENTRY_HEADER = re.compile(r"^\s*\[\[\s*kernel\s*\]\]")
_KEY = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


class _EntryLocator(Locator):
    """Lines of ``[[kernel]]`` entries (``("kernel", "<index>", key)``), which
    the table locator cannot tell apart."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.entries: list[dict[str, int]] = []
        for number, line in enumerate(text.splitlines(), start=1):
            if _ENTRY_HEADER.match(line):
                self.entries.append({"": number})
            elif self.entries and (key := _KEY.match(line)):
                self.entries[-1].setdefault(key.group(1), number)

    def line(self, path: tuple[str, ...]) -> int:
        if len(path) >= 2 and path[0] == "kernel" and path[1].isdigit():
            index = int(path[1])
            if index < len(self.entries):
                keys = self.entries[index]
                return keys.get(path[2], keys[""]) if len(path) > 2 else keys[""]
        return super().line(path)


@dataclass(frozen=True)
class Entry:
    """One ``[[kernel]]`` of a ``registry.toml``.

    Attributes:
        name: The kernel's name (also its Bazel target's).
        template: The template folder, beside the ``registry.toml``.
        line: Where the entry starts in the file.
        params: Fixed parameter values.
        formats: The formats it runs in, if it narrows the template's.
        cases: The template cases it runs, if it chooses (``default`` among them).
        doc: One line describing it.
    """

    name: str
    template: str
    line: int
    params: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    formats: tuple[str, ...] | None = None
    cases: tuple[str, ...] | None = None
    doc: str | None = None

    def overrides(self) -> dict[str, Any]:
        """The template overrides it stands for (``load_manifest``)."""
        out: dict[str, Any] = {"name": self.name, "params": dict(self.params)}
        if self.formats is not None:
            out["formats"] = list(self.formats)
        if self.cases is not None:
            out["cases"] = list(self.cases)
        return out


def _strings(r: Reader, raw: object, path: tuple[str, ...]) -> tuple[str, ...]:
    items = r.typed(raw, list, path)
    if not items:
        raise r.fail(path, f"{where(path)} must not be empty")
    return tuple(r.text(item, path) for item in items)


def _entry(r: Reader, family: Path, index: int, raw: object) -> Entry:
    path = ("kernel", str(index))
    table = r.table(raw, path)
    r.known(table, _ENTRY_KEYS, path)
    for key in ("name", "template"):
        if key not in table:
            raise r.fail(path, f"[[kernel]] #{index + 1} needs '{key}'")
    name = r.text(table["name"], (*path, "name"))
    if not name.isidentifier():
        raise r.fail((*path, "name"), f"[[kernel]] name {name!r} must be an identifier")
    template = r.text(table["template"], (*path, "template"))
    if not (family / template / TOML_NAME).is_file():
        raise r.fail(
            (*path, "template"),
            f"[[kernel]] {name}: template {template!r} is no kernel folder in {family.name}/",
        )
    entry = Entry(name, template, r.locator.line(path))
    if "params" in table:
        entry = replace(entry, params=MappingProxyType(r.table(table["params"], (*path, "params"))))
    if "formats" in table:
        entry = replace(entry, formats=_strings(r, table["formats"], (*path, "formats")))
    if "cases" in table:
        entry = replace(entry, cases=_strings(r, table["cases"], (*path, "cases")))
    if "doc" in table:
        entry = replace(entry, doc=r.text(table["doc"], (*path, "doc")))
    return entry


def read_registry(registry: Path) -> tuple[Entry, ...]:
    """The entries of a ``registry.toml``, checked against its family folder.

    Args:
        registry: ``kernels/<family>/registry.toml``.

    Returns:
        The entries, in file order.

    Raises:
        ManifestError: The file breaks the schema (an unknown key, a missing
            name or template, an unknown template, a duplicate name); the
            message names the line.
    """
    try:
        text = registry.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestError(f"{registry}:1: cannot read it: {error.strerror}") from None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"{registry}:{decode_error_line(str(error))}: {error}") from None
    r = Reader(registry, _EntryLocator(text))
    r.known(data, ("kernel",), ())
    listed = r.typed(data.get("kernel", []), list, ("kernel",))
    entries: list[Entry] = []
    seen: dict[str, Entry] = {}
    for index, raw in enumerate(listed):
        entry = _entry(r, registry.parent, index, raw)
        if entry.name in seen:
            raise ManifestError(
                f"{registry}:{entry.line}: kernel {entry.name} is also declared at line "
                f"{seen[entry.name].line}"
            )
        seen[entry.name] = entry
        entries.append(entry)
    return tuple(entries)


def load_entry(registry: Path, entry: Entry) -> KernelManifest:
    """The kernel an entry declares: its template with the entry's overrides.

    Raises:
        ManifestError: The template fails to load, or an override does not fit
            it (a parameter it lacks, a value outside its domain); errors from
            the overrides name the entry's line.
    """
    manifest = load_manifest(
        registry.parent / entry.template,
        overrides=entry.overrides(),
        source=f"{registry}:{entry.line}",
    )
    return replace(manifest, doc=entry.doc, registry=registry)


def entry_ref(registry: Path, name: str) -> str:
    """``<registry.toml>::<name>``: how a tool names an entry."""
    return f"{registry}{ENTRY_SEPARATOR}{name}"


def load_kernel(ref: str | Path) -> KernelManifest:
    """A kernel by its folder, or by ``<registry.toml>::<name>`` for an entry.

    Raises:
        ManifestError: It fails to load, or the registry has no such entry.
    """
    path, sep, name = str(ref).partition(ENTRY_SEPARATOR)
    if not sep:
        return load_manifest(Path(path))
    registry = Path(path)
    for entry in read_registry(registry):
        if entry.name == name:
            return load_entry(registry, entry)
    raise ManifestError(f"{registry}:1: no entry named {name!r}")


def render_bzl(registry: Path) -> str:
    """``registry.bzl``: the entries as Starlark, for ``ipu_kernel_family()``.

    Bazel cannot read TOML at load time; this checked-in file carries what the
    macro needs (each entry's name and template), and a test fails when it is
    stale (IMPLEMENTATION-DECISIONS, E06-17).
    """
    lines = [
        f'"""Generated from {REGISTRY_NAME} by //kernels:update_registry_bzl. Do not edit."""',
        "",
        "ENTRIES = [",
    ]
    lines += [
        f'    {{"name": "{e.name}", "template": "{e.template}"}},' for e in read_registry(registry)
    ]
    lines.append("]")
    return "\n".join(lines) + "\n"
