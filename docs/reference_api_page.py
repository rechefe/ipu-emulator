"""Generates the code-derived parts of the Reference model API page (E09-16).

``docs/content/reference/reference-model-api.md`` is hand-written prose that
includes four generated snippets (``pymdownx.snippets``), so the parts that
can drift come from the code and the schema, never from a doc edit:

* ``version.md``: ``API_VERSION`` and the schema versions;
* ``api.md``: every public symbol, from its signature and Google docstring;
* ``schema.md``: every field of the commit record and trace header schemas;
* ``faults.md``: the fault IDs a commit record or ``Status`` can name.

It also copies the two schema files beside them, for download.

    bazel run //docs:reference_api_page -- OUT_DIR

The ``build_docs`` genrule runs it into ``generated/reference-api/``.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import json
import re
import sys
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any

import ipu_emu.commit
import ipu_emu.errors
import ipu_emu.reference
import ipu_emu.trace
import ipu_isa.faults
import ipu_isa.formats
import ipu_isa.trace_schema
from ipu_isa.faults import FAULTS, ArithmeticEvent, FaultId

#: Names in a listed module's ``__all__`` that are not part of the reference
#: contract: they take or return emulator internals (``IpuState``).
INTERNAL = frozenset({"ipu_emu.commit.CommitCollector", "ipu_emu.trace.trace_header"})

#: The schema files, by schema name (``ipu_isa.trace_schema``).
SCHEMA_FILES = {"commit_record": "commit_record.v1.json", "trace": "trace.v1.json"}

#: The page's path from the site root; snippet links are relative to it.
FAULTS_PAGE = "../requirements/faults.md"


@dataclass(frozen=True)
class Group:
    """One section of the API listing: its symbols, as (module, name) pairs."""

    title: str
    intro: str
    symbols: tuple[tuple[ModuleType, str], ...]


def _defined_here(module: ModuleType, name: str) -> bool:
    """False for a name *module* re-exports (``reference.TraceWriter``): it is
    listed once, under the module that defines it."""
    value = getattr(module, name)
    if not (inspect.isclass(value) or inspect.isfunction(value)):
        return True  # a constant or a type alias (``Outcome``) has no home module
    return value.__module__ == module.__name__


def _public(module: ModuleType, *only: str) -> tuple[tuple[ModuleType, str], ...]:
    names = only or tuple(n for n in module.__all__ if _defined_here(module, n))
    return tuple((module, n) for n in names if f"{module.__name__}.{n}" not in INTERNAL)


def groups() -> tuple[Group, ...]:
    """The documented symbols: the stable module first, then what it hands out."""
    return (
        Group(
            "The model",
            "`ipu_emu.reference`: the names in its `__all__` (`TraceWriter` and "
            "`read_trace`, which it re-exports, are listed under Traces).",
            _public(ipu_emu.reference),
        ),
        Group(
            "Commit records",
            "`ipu_emu.commit`: what `step()` returns. `to_json()` gives the "
            "[schema document](#commit-record-schema).",
            _public(ipu_emu.commit),
        ),
        Group(
            "Traces",
            "`ipu_emu.trace`: the writer `run()` takes, and its reader.",
            _public(ipu_emu.trace),
        ),
        Group(
            "Schema helpers",
            "`ipu_isa.trace_schema`: the schema versions and the digests a trace header holds.",
            _public(ipu_isa.trace_schema),
        ),
        Group(
            "Argument and error types",
            "The types the methods take from `ipu_isa`, and the error they raise.",
            _public(ipu_isa.formats, "ElementFormat")
            + _public(ipu_isa.faults, "FaultId")
            + _public(ipu_emu.errors, "ConfigurationError"),
        ),
    )


# -- anchors and docstrings ----------------------------------------------------


def anchor(qualname: str) -> str:
    """The heading id of a symbol: ``ReferenceModel.step`` -> ``ReferenceModel-step``.

    Case is kept, so ``SCHEMA_VERSION`` and ``schema_version()`` get two ids.
    """
    return qualname.replace(".", "-")


def documented_symbols() -> dict[str, str]:
    """Every documented symbol's qualified name (``ReferenceModel.step``) -> its anchor."""
    out: dict[str, str] = {}
    for group in groups():
        for module, name in group.symbols:
            out[name] = anchor(name)
            obj = getattr(module, name)
            if inspect.isclass(obj):
                for member in _members(obj):
                    qualname = f"{name}.{member}"
                    out[qualname] = anchor(qualname)
    return out


_SECTION = re.compile(r"^(Args|Arguments|Returns|Raises|Attributes|Yields):\s*$")
_ROLE = re.compile(r":(?:meth|class|func|data|attr|exc|mod):`(~?)([\w.]+)`")


@dataclass
class Docstring:
    """A Google docstring, split into its summary text and sections."""

    text: str = ""
    sections: dict[str, list[tuple[str, str]]] = dataclasses.field(default_factory=dict)


def parse_docstring(doc: str | None) -> Docstring:
    """Split a Google docstring into its free text and its named sections.

    Args, Raises and Attributes entries are ``(name, text)``; a Returns
    section is one entry with an empty name.
    """
    out = Docstring()
    if not doc:
        return out
    lines = inspect.cleandoc(doc).splitlines()
    text: list[str] = []
    section: str | None = None
    for line in lines:
        match = _SECTION.match(line)
        if match:
            section = match.group(1).replace("Arguments", "Args")
            out.sections[section] = []
            continue
        if section is None:
            text.append(line)
            continue
        entries = out.sections[section]
        stripped = line.strip()
        if not stripped:
            continue
        if section in ("Returns", "Yields"):
            if entries:
                entries[-1] = ("", f"{entries[-1][1]} {stripped}")
            else:
                entries.append(("", stripped))
            continue
        head = re.match(r"^ {4}(\*{0,2}[\w.]+)(?: \([^)]*\))?:\s*(.*)$", line)
        if head:
            entries.append((head.group(1), head.group(2)))
        elif entries:
            name, body = entries[-1]
            entries[-1] = (name, f"{body} {stripped}".strip())
    out.text = "\n".join(text).strip()
    return out


def markdown(text: str, owner: str, symbols: Mapping[str, str]) -> str:
    """Docstring markup as Markdown: ``code`` to `code`, Sphinx roles to links.

    A role names a symbol relative to *owner* (``ReferenceModel``) or the page;
    a symbol the page documents becomes a link to its heading.
    """

    def role(match: re.Match[str]) -> str:
        tilde, target = match.groups()
        shown = target.rsplit(".", 1)[-1] if tilde else target
        for candidate in _candidates(target, owner):
            if candidate in symbols:
                return f"[`{shown}`](#{symbols[candidate]})"
        return f"`{shown}`"

    text = _ROLE.sub(role, text)
    return text.replace("``", "`")


def _candidates(target: str, owner: str) -> Iterator[str]:
    parts = target.split(".")
    for start in range(len(parts)):
        tail = ".".join(parts[start:])
        if owner:
            yield f"{owner}.{tail}"
        yield tail


def _paragraph(text: str) -> str:
    """Docstring paragraphs joined into Markdown paragraphs (no hard wraps)."""
    blocks = [" ".join(line.strip() for line in block.splitlines()) for block in text.split("\n\n")]
    return "\n\n".join(b for b in blocks if b)


# -- rendering symbols ----------------------------------------------------------


def _members(cls: type) -> list[str]:
    """The public methods and properties *cls* defines itself, in source order."""
    out = []
    for name, member in vars(cls).items():
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(member, (property, classmethod, staticmethod)) or inspect.isfunction(member):
            if name == "__init__":
                continue  # the class heading shows the constructor
            out.append(name)
    return out


def _annotation(annotation: object) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _default(value: object) -> str:
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    return repr(value)


def signature(function: Callable[..., Any], name: str, *, skip_self: bool) -> str:
    """``name(arg: type = default, *, kw: type) -> type``, annotations as written."""
    sig = inspect.signature(function)
    params = []
    star = False
    for index, param in enumerate(sig.parameters.values()):
        if skip_self and index == 0:
            continue
        if param.kind is param.KEYWORD_ONLY and not star:
            params.append("*")
            star = True
        if param.kind is param.VAR_POSITIONAL:
            star = True
            prefix = "*"
        elif param.kind is param.VAR_KEYWORD:
            prefix = "**"
        else:
            prefix = ""
        text = prefix + param.name
        annotation = _annotation(param.annotation)
        if annotation:
            text += f": {annotation}"
        if param.default is not inspect.Parameter.empty:
            text += f" = {_default(param.default)}" if annotation else f"={_default(param.default)}"
        params.append(text)
    returns = _annotation(sig.return_annotation)
    return f"{name}({', '.join(params)})" + (f" -> {returns}" if returns else "")


def _data_comment(module: ModuleType, name: str) -> str:
    """The ``#:`` comment lines just above ``name = ...`` in *module*'s source."""
    lines = inspect.getsource(module).splitlines()
    for index, line in enumerate(lines):
        if re.match(rf"^{re.escape(name)}\s*(:[^=]*)?=", line):
            comment: list[str] = []
            back = index - 1
            while back >= 0 and lines[back].startswith("#:"):
                comment.insert(0, lines[back][2:].strip())
                back -= 1
            return " ".join(comment)
    return ""


def _heading(level: int, title: str, qualname: str) -> str:
    return f"{'#' * level} `{title}` {{ #{anchor(qualname)} }}"


def _code(text: str) -> str:
    return f"```python\n{text}\n```"


def _sections(doc: Docstring, owner: str, symbols: Mapping[str, str]) -> list[str]:
    out = []
    labels = {"Args": "Arguments", "Returns": "Returns", "Yields": "Yields", "Raises": "Raises"}
    for key, label in labels.items():
        entries = doc.sections.get(key)
        if not entries:
            continue
        out.append(f"**{label}**")
        items = []
        for name, body in entries:
            text = markdown(body, owner, symbols)
            if not name:
                items.append(f"- {text}")
            elif key == "Raises" and name in symbols:
                items.append(f"- [`{name}`](#{symbols[name]}): {text}")
            else:
                items.append(f"- `{name}`: {text}")
        out.append("\n".join(items))
    return out


def _field_table(cls: type, doc: Docstring, owner: str, symbols: Mapping[str, str]) -> str:
    meanings = dict(doc.sections.get("Attributes", []))
    rows = ["| Field | Type | Default | Meaning |", "|---|---|---|---|"]
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            default = f"`{_default(field.default)}`"
        elif field.default_factory is not dataclasses.MISSING:
            default = f"`{_default(field.default_factory())}`"
        else:
            default = ""
        meaning = markdown(meanings.get(field.name, ""), owner, symbols)
        rows.append(
            f"| `{field.name}` | `{_cell(str(field.type))}` | {_cell(default)} | {_cell(meaning)} |"
        )
    return "\n".join(rows)


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _render_enum(
    obj: type[enum.Enum], name: str, path: str, symbols: Mapping[str, str]
) -> list[str]:
    doc = parse_docstring(obj.__doc__)
    bases = ", ".join(b.__name__ for b in obj.__bases__)
    blocks = [_heading(4, name, name), _code(f"class {name}({bases})  # {path}")]
    if doc.text:
        blocks.append(markdown(_paragraph(doc.text), name, symbols))
    rows = ["| Member | Value |", "|---|---|"]
    rows += [f"| `{m.name}` | `{m.value!r}` |" for m in obj]
    blocks.append("\n".join(rows))
    return blocks + [_render_member(obj, name, m, symbols) for m in _members(obj)]


def _render_class(obj: type, name: str, path: str, symbols: Mapping[str, str]) -> list[str]:
    doc = parse_docstring(obj.__doc__)
    is_data = dataclasses.is_dataclass(obj)
    init = vars(obj).get("__init__")
    if init is not None and not is_data:
        head = "class " + signature(init, name, skip_self=True).removesuffix(" -> None")
    else:
        bases = [b.__name__ for b in obj.__bases__ if b is not object]
        head = f"class {name}" + (f"({', '.join(bases)})" if bases else "")
    blocks = [_heading(4, name, name), _code(f"{head}  # {path}")]
    if doc.text:
        blocks.append(markdown(_paragraph(doc.text), name, symbols))
    if is_data:
        frozen = obj.__dataclass_params__.frozen  # type: ignore[attr-defined]
        blocks.append(f"A {'frozen ' if frozen else ''}dataclass.")
        blocks.append(_field_table(obj, doc, name, symbols))
    blocks += _sections(doc, name, symbols)
    return blocks + [_render_member(obj, name, m, symbols) for m in _members(obj)]


def _render_function(
    obj: Callable[..., Any], name: str, path: str, symbols: Mapping[str, str]
) -> list[str]:
    doc = parse_docstring(obj.__doc__)
    blocks = [
        _heading(4, f"{name}()", name),
        _code(f"{signature(obj, name, skip_self=False)}  # {path}"),
    ]
    if doc.text:
        blocks.append(markdown(_paragraph(doc.text), "", symbols))
    return blocks + _sections(doc, "", symbols)


def _render_data(
    module: ModuleType, obj: object, name: str, path: str, symbols: Mapping[str, str]
) -> list[str]:
    """A constant, or a ``Literal`` alias; documented by its ``#:`` comment."""
    if typing.get_origin(obj) is typing.Literal:
        code = f"{name} = Literal[{', '.join(map(repr, typing.get_args(obj)))}]  # {path}"
    else:
        code = f"{name} = {obj!r}  # {path}"
    blocks = [_heading(4, name, name), _code(code)]
    comment = _data_comment(module, name)
    if comment:
        blocks.append(markdown(comment, "", symbols))
    return blocks


def render_symbol(module: ModuleType, name: str, symbols: Mapping[str, str]) -> str:
    """One symbol's heading, signature and documentation."""
    obj = getattr(module, name)
    path = f"{module.__name__}.{name}"
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        blocks = _render_enum(obj, name, path, symbols)
    elif inspect.isclass(obj):
        blocks = _render_class(obj, name, path, symbols)
    elif typing.get_origin(obj) is not typing.Literal and callable(obj):
        blocks = _render_function(obj, name, path, symbols)
    else:
        blocks = _render_data(module, obj, name, path, symbols)
    return "\n\n".join(blocks)


def _render_member(cls: type, owner: str, member: str, symbols: Mapping[str, str]) -> str:
    raw = vars(cls)[member]
    qualname = f"{owner}.{member}"
    if isinstance(raw, property):
        doc = parse_docstring(raw.__doc__)
        returns = _annotation(inspect.signature(raw.fget).return_annotation) if raw.fget else ""
        head = f"{owner}.{member}" + (f": {returns}" if returns else "") + "  # property"
        title = f"{owner}.{member}"
    else:
        function = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
        doc = parse_docstring(function.__doc__)
        skip = not isinstance(raw, staticmethod)
        head = signature(function, f"{owner}.{member}", skip_self=skip)
        if isinstance(raw, classmethod):
            head += "  # classmethod"
        title = f"{owner}.{member}()"
    blocks = [_heading(5, title, qualname), _code(head)]
    if doc.text:
        blocks.append(markdown(_paragraph(doc.text), owner, symbols))
    blocks += _sections(doc, owner, symbols)
    return "\n\n".join(blocks)


def render_api() -> str:
    """Every public symbol, grouped by module."""
    symbols = documented_symbols()
    out = []
    for group in groups():
        out.append(f"### {group.title}\n\n{group.intro}")
        for module, name in group.symbols:
            out.append(render_symbol(module, name, symbols))
    return "\n\n".join(out) + "\n"


# -- rendering the schemas --------------------------------------------------------

#: Meanings for the schema fields that carry no ``description``. A field the
#: schema describes takes its description; a new field the schema does not
#: describe fails //docs:test_reference_api_page until it has one (here, or in
#: the schema at its next version).
FIELD_MEANINGS: Mapping[str, Mapping[str, str]] = {
    "commit_record": {
        "pc_before": "The word's PC.",
        "pc_after": "The PC of the next word.",
        "slots": "The word's ops, one per non-`NOP` slot.",
        "slots[].slot": "The slot, as `ipu_isa.slots` names it.",
        "slots[].mnemonic": "The op's mnemonic.",
        "slots[].fields": "The encoded operand values, by operand name.",
        "reg_writes": "The registers the word wrote, at their end-of-cycle values.",
        "reg_writes[].register": "The bank, as `ipu_isa.registers` names it.",
        "reg_writes[].index": "A scalar register (`lr`, `cr`): the register within the bank.",
        "reg_writes[].value": "A scalar register: its unsigned 32-bit value.",
        "reg_writes[].offset": "A vector register: the first byte written, within the bank.",
        "reg_writes[].data": "A vector register: the bytes written.",
        "xmem_writes": "The XMEM ranges the word wrote.",
        "xmem_writes[].address": "The first XMEM byte written.",
        "xmem_writes[].data": "The bytes written.",
        "fault": "The fault the word raised, or `null`.",
        "fault.id": "The fault ID, as the [fault list](#faults) writes it.",
        "fault.pc": "The faulting word's PC.",
    },
    "trace": {
        "emulator_version": "The version of the model that wrote the trace "
        "(`ipu_emu.trace.EMULATOR_VERSION` for this emulator).",
    },
}


@dataclass(frozen=True)
class SchemaField:
    """One row of a schema table.

    Attributes:
        path: ``cycle``, or ``slots[].slot`` for a field of each array item.
        type: The JSON type, as text.
        required: Whether its object must have it.
        description: The schema's ``description``, or "".
        form: For a field of one ``oneOf`` alternative, ``(n, of)``; else ``None``.
    """

    path: str
    type: str
    required: bool
    description: str
    form: tuple[int, int] | None = None


def _resolve(node: Mapping[str, Any], schema: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return _resolve(schema["$defs"][ref.removeprefix("#/$defs/")], schema)
    return node


def _type(node: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    node = _resolve(node, schema)
    if "const" in node:
        return f"`{json.dumps(node['const'])}`"
    if "enum" in node:
        return "one of " + ", ".join(f"`{json.dumps(v)}`" for v in node["enum"])
    if "oneOf" in node:
        options = dict.fromkeys(_type(option, schema) for option in node["oneOf"])
        return " or ".join(options)
    kind = node.get("type", "any")
    if kind == "array":
        return f"array of {_type(node.get('items', {}), schema)}"
    if kind == "object" and "additionalProperties" in node and "properties" not in node:
        extra = node["additionalProperties"]
        if isinstance(extra, Mapping):
            return f"object of {_type(extra, schema)}"
    text = str(kind)
    limits = []
    if "minimum" in node and "maximum" in node:
        limits.append(f"{node['minimum']} to {node['maximum']}")
    elif "minimum" in node:
        limits.append(f"≥ {node['minimum']}")
    if "pattern" in node:
        limits.append(f"`{node['pattern']}`")
    return text + (f" ({', '.join(limits)})" if limits else "")


def _objects(node: Mapping[str, Any], schema: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The object schemas a value can be: itself, its array items, its ``oneOf`` options."""
    node = _resolve(node, schema)
    if node.get("type") == "array":
        return _objects(node.get("items", {}), schema)
    if "oneOf" in node:
        return [o for option in node["oneOf"] for o in _objects(option, schema)]
    if "properties" in node:
        return [node]
    return []


def schema_fields(schema: Mapping[str, Any], prefix: str = "") -> list[SchemaField]:
    """Every field of *schema*, nested ones as ``slots[].slot`` or ``fault.id``."""
    out: list[SchemaField] = []
    seen: set[str] = set()
    forms: dict[str, set[tuple[int, int] | None]] = {}

    def walk(node: Mapping[str, Any], prefix: str, form: tuple[int, int] | None) -> None:
        required = set(node.get("required", []))
        for name, sub in node.get("properties", {}).items():
            path = f"{prefix}{name}"
            resolved = _resolve(sub, schema)
            forms.setdefault(path, set()).add(form)
            if path not in seen:
                seen.add(path)
                description = resolved.get("description") or sub.get("description", "")
                field = SchemaField(path, _type(sub, schema), name in required, description, form)
                out.append(field)
            is_array = resolved.get("type") == "array"
            children = _objects(sub, schema)
            for number, child in enumerate(children, start=1):
                child_form = (number, len(children)) if len(children) > 1 else None
                walk(child, f"{path}[]." if is_array else f"{path}.", child_form)

    walk(schema, prefix, None)
    # A field of every alternative is not specific to one.
    return [
        f if f.form is None or len(forms[f.path]) < f.form[1] else dataclasses.replace(f, form=None)
        for f in out
    ]


def _condition_text(node: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
    parts = []
    for name, rule in node.get("properties", {}).items():
        if "const" in rule:
            parts.append(f"`{name}` is `{json.dumps(rule['const'])}`")
        elif rule.get("maxItems") == 0:
            parts.append(f"`{name}` is empty")
        elif "type" in rule:
            parts.append(
                f"`{name}` is {'`null`' if rule['type'] == 'null' else 'an ' + rule['type']}"
            )
        else:
            parts.append(f"`{name}`: `{json.dumps(rule)}`")
    return ", ".join(parts)


def schema_rules(schema: Mapping[str, Any]) -> list[str]:
    """The schema's ``allOf`` if/then/else rules, as sentences."""
    out = []
    for rule in schema.get("allOf", []):
        if "if" in rule:
            sentence = f"When {_condition_text(rule['if'], schema)}: {_condition_text(rule.get('then', {}), schema)}."
            if "else" in rule:
                sentence += f" Otherwise {_condition_text(rule['else'], schema)}."
            out.append(sentence)
        else:
            out.append(f"`{json.dumps(rule)}`")
    return out


def _meaning(name: str, field: SchemaField) -> str:
    return field.description or FIELD_MEANINGS.get(name, {}).get(field.path, "")


def render_schema(name: str, schema: Mapping[str, Any], download: str) -> str:
    """A schema's version, description, field table and rules."""
    version = str(schema["$id"]).rsplit("/", 1)[1]
    lines = [
        f"**{schema.get('title', name)}**, version `{version}` "
        f"([`{SCHEMA_FILES.get(name, name)}`]({download})): {schema.get('description', '')}",
        "",
        "| Field | Type | Required | Meaning |",
        "|---|---|---|---|",
    ]
    for field in schema_fields(schema):
        required = "yes" if field.required else "no"
        if field.form is not None:
            required += f", in form {field.form[0]} of {field.form[1]}"
        lines.append(
            f"| `{field.path}` | {_cell(field.type)} | {required} "
            f"| {_cell(_meaning(name, field))} |"
        )
    rules = schema_rules(schema)
    if rules:
        lines += ["", *(f"- {rule}" for rule in rules)]
    return "\n".join(lines)


def render_schemas(schemas: Mapping[str, Mapping[str, Any]], asset_dir: str) -> str:
    """The commit record and trace header schemas, and the arithmetic event names."""
    blocks = [
        render_schema(name, schema, f"{asset_dir}/{SCHEMA_FILES[name]}")
        for name, schema in schemas.items()
    ]
    events = ", ".join(f"`{event.value}`" for event in ArithmeticEvent)
    blocks.append(
        f"`arith_events` holds the names of `ipu_isa.faults.ArithmeticEvent`: {events}, "
        "sorted. An event never halts the model."
    )
    return "\n\n".join(blocks) + "\n"


# -- faults and version -----------------------------------------------------------


#: Why the model never raises a fault (faults.md, "In the emulator"): F-8 is
#: not modelled (``FaultInfo.modelled``), and F-7 has nothing to check.
NEVER_RAISED: Mapping[FaultId, str] = {
    FaultId.F7: "the emulator has no table configuration, so no table is read-only or write-only",
    FaultId.F8: "the emulator has no stall model "
    "([OQ-28](../requirements/open-questions.md#emulator-and-isa))",
}


def render_faults() -> str:
    """The fault IDs the API can report, from ``ipu_isa.faults.FAULTS``."""
    rows = ["| ID | `FaultId` | Fault | Detected in | In the model |", "|---|---|---|---|---|"]
    for fault_id, info in FAULTS.items():
        reason = NEVER_RAISED.get(fault_id)
        if not info.modelled:
            model = "**Not modelled**: never reported" + (f"; {reason}." if reason else ".")
        elif reason:
            model = f"Never raised: {reason}."
        else:
            model = "Reported."
        rows.append(
            f"| [{fault_id.label}]({FAULTS_PAGE}#fault-list) | `FaultId.{fault_id.name}` "
            f"| {_cell(info.summary)} | `{info.detected_in.value}` | {model} |"
        )
    return "\n".join(rows) + "\n"


def render_version() -> str:
    """The API version and the schema versions, as one sentence."""
    record = ipu_isa.trace_schema.schema_version("commit_record")
    trace = ipu_isa.trace_schema.schema_version("trace")
    return (
        f"API version **`{ipu_emu.reference.API_VERSION}`** "
        f"(`ipu_emu.reference.API_VERSION`); commit record schema `{record}`, "
        f"trace header schema `{trace}`.\n"
    )


def load_schemas() -> dict[str, dict[str, Any]]:
    """Both schemas, parsed, in page order."""
    return {name: ipu_isa.trace_schema.load_schema(name) for name in SCHEMA_FILES}  # type: ignore[arg-type]


def write_all(out_dir: Path, asset_dir: str = "../assets/schemas") -> list[Path]:
    """Write every snippet and the schema files into *out_dir*; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "version.md": render_version(),
        "api.md": render_api(),
        "schema.md": render_schemas(load_schemas(), asset_dir),
        "faults.md": render_faults(),
    }
    written = []
    for name, text in files.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    for file_name in SCHEMA_FILES.values():
        data = resources.files("ipu_isa").joinpath(f"schemas/{file_name}").read_bytes()
        path = out_dir / file_name
        path.write_bytes(data)
        written.append(path)
    return written


def main(argv: Iterable[str]) -> int:
    args = list(argv)
    if len(args) != 1:
        print("usage: reference_api_page OUT_DIR", file=sys.stderr)
        return 2
    for path in write_all(Path(args[0])):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
