"""The Jinja preprocessing layer of an assembly source, rendered safely.

Assembly text is rendered before it is parsed, so an `.asm` file is executable
input rather than inert data. `jinja2.Template` is not a sandbox: from any
expression, `{{ ''.__class__.__mro__ }}` walks to `os`, so

    {{ self.__init__.__globals__.__builtins__.__import__("os")
       .popen("…").read() }}

runs a shell command during assembly, and the source still assembles cleanly
afterwards because rendering succeeded. Assembling a kernel someone sent you is
then indistinguishable from running whatever they wanted to run.

So every render of assembly source goes through `render` here, the one place
in ipu_as that builds a Jinja environment for assembly source. A template that
reaches past what a kernel legitimately needs fails with a SecurityError,
which `render` reports, like every other template failure, as a located
`ipu_as.errors.TemplateError` whose `__cause__` is the jinja2 exception.

Template values come in as `defines`, checked against the declared parameters:
a `ParamSpec` the caller passes (the kernel harness hands over
`kernel.toml [params]`), or an `ipu:params` header comment in the source:

    {#- ipu:params
    cpr:        int = 3            # column chunks per row
    rows:       int = 8
    activation: str = "identity" in (identity, relu, silu)
    -#}

`{% include %}` and `{% import %}` resolve only inside the `search_path` roots.

This bounds reach, not cost: the sandbox does not limit how long a render runs
or how much it allocates. `{% for _ in range(10**9) %}` still hangs. It stops
the escape, not the loop.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import SandboxedEnvironment

from ipu_as.errors import SourceLocation, TemplateError
from ipu_isa.registers import REGISTER_DEFINITIONS

#: A marker means the parser sees different text than the file holds, so a
#: position in the parsed text need not correspond to the file it came from.
MARKERS = ("{{", "{%", "{#")

#: The default of a parameter that has none: the caller must supply it.
REQUIRED = object()

#: Parameter types a header can name, by their header spelling.
_TYPES: dict[str, type] = {"int": int, "bool": bool, "str": str, "tuple[int,...]": tuple}


@dataclass(frozen=True)
class Param:
    """One declared template parameter."""

    name: str
    #: int, bool, str or tuple (of ints) in an ``ipu:params`` header; a caller's
    #: ParamSpec may name any type (the kernel harness passes float and dict).
    type: type
    default: object = REQUIRED
    choices: tuple | None = None
    line: int | None = None  #: its line in the header, when it came from one


@dataclass(frozen=True)
class ParamSpec:
    """The parameters a template declares."""

    params: tuple[Param, ...]
    line: int = 1  #: the declaration's first line in its source

    def bind(self, defines: Mapping[str, object], source: str = "<asm>") -> dict[str, object]:
        """The render context for `defines`: checked, coerced, and filled from defaults.

        Raises:
            TemplateError: a define is undeclared, a required parameter is
                missing, or a value has the wrong type or is not a choice.
        """
        declared = {param.name: param for param in self.params}
        for name in defines:
            if name not in declared:
                raise TemplateError(
                    f"undeclared define {name!r}",
                    location=SourceLocation(self.line, 1, source=source),
                )
        context: dict[str, object] = {}
        for param in self.params:
            where = SourceLocation(param.line or self.line, 1, source=source)
            if param.name in defines:
                value = _coerce(param, defines[param.name], where)
            elif param.default is REQUIRED:
                raise TemplateError(f"missing required define {param.name!r}", location=where)
            else:
                value = param.default
            if param.choices is not None and value not in param.choices:
                raise TemplateError(
                    f"define {param.name!r} is {value!r}, not one of {param.choices}",
                    location=where,
                )
            context[param.name] = value
        return context


def _coerce(param: Param, value: object, where: SourceLocation) -> object:
    """`value` as `param.type`, converting a command-line string."""
    if isinstance(value, str) and param.type is not str:
        try:
            return _from_text(param.type, value)
        except ValueError:
            raise TemplateError(
                f"define {param.name!r} is {value!r}, not of type {_type_name(param.type)}",
                location=where,
            ) from None
    if param.type is tuple and isinstance(value, (list, tuple)):
        value = tuple(value)
    valid = isinstance(value, param.type) and not (param.type is int and isinstance(value, bool))
    if param.type is tuple:
        valid = valid and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    if not valid:
        raise TemplateError(
            f"define {param.name!r} is {value!r}, not of type {_type_name(param.type)}",
            location=where,
        )
    return value


def _from_text(kind: type, text: str) -> object:
    """Parse a command-line value: `3`, `true`, `1,2,3`. ValueError if it is not one."""
    text = text.strip()
    if kind is bool:
        lowered = text.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(text)
    if kind is int:
        return int(text, 0)
    if kind is tuple:
        return tuple(int(part, 0) for part in text.strip("()").split(",") if part.strip())
    return text


def _type_name(kind: type) -> str:
    return next((name for name, value in _TYPES.items() if value is kind), kind.__name__)


_HEADER = re.compile(r"\{#-?\s*ipu:params[ \t]*\n(?P<body>.*?)-?#\}", re.DOTALL)
_PARAM_LINE = re.compile(
    r"(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<type>tuple\[int,\s*\.\.\.\]|\w+)"
    r"(?:\s*=\s*(?P<default>.+?))?"
    r"(?:\s+in\s+\((?P<choices>(?:\"[^\"]*\"|'[^']*'|[^)\"'])*)\))?\s*$"
)
#: One choice in `in (...)`: a quoted string (which may hold commas) or a bare word.
_CHOICE = re.compile(r"\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^,\s]+)")


def _strip_comment(line: str) -> str:
    """`line` without its `# note`; a `#` inside quotes is part of a value."""
    quote = None
    for i, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return line[:i]
    # An unclosed quote is an apostrophe in prose (`don't`), not a quoted value.
    return line if quote is None else line.split("#", 1)[0]


def declared_params(text: str, source: str = "<asm>") -> ParamSpec | None:
    """The `ipu:params` header comment of `text`, or None if it has none.

    Raises:
        TemplateError: a header line is malformed; it names the line.
    """
    header = _HEADER.search(text)
    if header is None:
        return None
    first = text.count("\n", 0, header.start()) + 1
    body_line = first + 1
    params = []
    for offset, raw in enumerate(header.group("body").splitlines()):
        line = body_line + offset
        spec = _strip_comment(raw).strip()
        if not spec:
            continue
        where = SourceLocation(line, 1, source=source)
        param = _parse_param(spec, where)
        if any(seen.name == param.name for seen in params):
            raise TemplateError(f"ipu:params declares {param.name!r} twice", location=where)
        params.append(param)
    return ParamSpec(tuple(params), line=first)


def _parse_param(spec: str, where: SourceLocation) -> Param:
    match = _PARAM_LINE.fullmatch(spec)
    if match is None:
        raise TemplateError(f"malformed ipu:params line {spec!r}", location=where)
    type_name = re.sub(r"\s+", "", match["type"])
    if type_name not in _TYPES:
        raise TemplateError(f"unknown ipu:params type {match['type']!r}", location=where)
    kind = _TYPES[type_name]
    probe = Param(match["name"], kind, line=where.line)
    default: object = REQUIRED
    if match["default"] is not None:
        try:
            literal = ast.literal_eval(match["default"])
        except (ValueError, SyntaxError):
            literal = match["default"]
        default = _coerce(probe, literal, where)
    choices = None
    if match["choices"] is not None:
        choices = tuple(
            _coerce(probe, next(text for text in choice.groups() if text is not None), where)
            for choice in _CHOICE.finditer(match["choices"])
        )
    return Param(match["name"], kind, default, choices, where.line)


#: A transform applied to each template's text before it is compiled, given the
#: template's name: diagnostics uses one to tag every line with its origin.
SourceTransform = Callable[[str, str], str]


class _RootedLoader(jinja2.BaseLoader):
    """Loads `{% include %}` / `{% import %}` targets from inside the search roots only."""

    def __init__(self, roots: tuple[Path, ...], transform: SourceTransform | None) -> None:
        self.roots = tuple(root.resolve() for root in roots)
        self.transform = transform
        #: Filename -> template name of every template loaded, for error locations.
        self.loaded: dict[str, str] = {}

    def get_source(self, environment, name):
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts:
            raise jinja2.TemplateNotFound(
                name, f"{name!r}: absolute and '..' names are not allowed"
            )
        for root in self.roots:
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise jinja2.TemplateNotFound(name, f"{name!r} resolves outside its search root")
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                if self.transform is not None:
                    text = self.transform(name, text)
                mtime = path.stat().st_mtime
                self.loaded[str(path)] = name
                return text, str(path), lambda path=path, mtime=mtime: _unchanged(path, mtime)
        raise jinja2.TemplateNotFound(name, f"{name!r} is not in the search path")


def _unchanged(path: Path, mtime: float) -> bool:
    """jinja's uptodate check: False (reload) when the file changed or is gone."""
    try:
        return path.stat().st_mtime == mtime
    except OSError:
        return False


_ENVIRONMENTS: dict[tuple, SandboxedEnvironment] = {}


def _environment(
    search_path: Sequence[Path], transform: SourceTransform | None
) -> SandboxedEnvironment:
    """One sandboxed environment per (search path, transform)."""
    # Resolved, so a relative root means the directory it meant when the call was made.
    key = (tuple(Path(p).resolve() for p in search_path), transform)
    if key not in _ENVIRONMENTS:
        loader = _RootedLoader(key[0], transform) if key[0] else None
        # StrictUndefined: a misspelled or undeclared name fails at its line
        # instead of rendering as "" inside an instruction.
        _ENVIRONMENTS[key] = SandboxedEnvironment(loader=loader, undefined=jinja2.StrictUndefined)
    return _ENVIRONMENTS[key]


def lex(text: str) -> list[tuple[int, str, str]]:
    """Jinja's (line, token type, value) tokens of `text`, without rendering it."""
    return list(_environment((), None).lex(text))


#: The ABI namespaces a template may name registers through, and the register
#: file each one names: `{{ cr.data_base }}` -> "CR2", `{{ lr.data_ptr }}` -> "LR0".
ABI_NAMESPACES = ("cr", "lr")


class _AbiNamespace:
    """A read-only `name -> register token` mapping a template reads as `cr.name` or `cr["name"]`."""

    def __init__(self, namespace: str, registers: Mapping[str, str]) -> None:
        self._namespace = namespace
        self._registers = dict(registers)

    def __getattr__(self, name: str) -> str:
        # Only reached for names the object does not have; the sandbox refuses
        # `_`-names before this, and abi_namespaces() rejects them as ABI names.
        return self[name]

    def __getitem__(self, name: str) -> str:
        try:
            return self._registers[name]
        except KeyError:
            # Not AttributeError or KeyError: jinja turns those into Undefined,
            # and an unknown register name must fail whatever the Undefined policy.
            raise TemplateError(f"unknown ABI name {self._namespace}.{name}") from None

    def __iter__(self):
        raise TemplateError(f"the ABI namespace {self._namespace} cannot be iterated")

    def __str__(self) -> str:
        raise TemplateError(f"use {self._namespace}.<name>, not {self._namespace} alone")


def _abi_context(
    namespaces: Mapping[str, Mapping[str, str]], source_name: str
) -> dict[str, _AbiNamespace]:
    """The `cr` / `lr` render values; TemplateError for a namespace or register the ISA lacks."""
    where = SourceLocation(1, 1, source=source_name)
    context = {}
    for namespace, registers in namespaces.items():
        if namespace not in ABI_NAMESPACES:
            raise TemplateError(
                f"unknown ABI namespace {namespace!r} (have: {', '.join(ABI_NAMESPACES)})",
                location=where,
            )
        tokens = REGISTER_DEFINITIONS[namespace]["assembler_values"]
        checked = {}
        for name, register in registers.items():
            if not str(name).isidentifier() or str(name).startswith("_"):
                raise TemplateError(f"ABI name {namespace}.{name} is not a name", location=where)
            if str(register).lower() not in tokens:
                raise TemplateError(
                    f"ABI name {namespace}.{name} = {register!r} is not a register "
                    f"{tokens[0].upper()}-{tokens[-1].upper()}",
                    location=where,
                )
            checked[name] = str(register).upper()
        context[namespace] = _AbiNamespace(namespace, checked)
    return context


def abi_namespaces(table: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Split a flat `name -> register` table (`kernel.toml [abi]`) into the `cr` / `lr` namespaces.

    Raises:
        TemplateError: a value is not a CR or LR register.
    """
    namespaces: dict[str, dict[str, str]] = {namespace: {} for namespace in ABI_NAMESPACES}
    for name, register in table.items():
        namespace = str(register)[:2].lower()
        if not isinstance(register, str) or namespace not in namespaces:
            raise TemplateError(f"[abi] {name} = {register!r} is not a CR or LR register")
        namespaces[namespace][name] = register
    _abi_context(namespaces, "[abi]")  # the per-register checks
    return namespaces


def _check_reserved(names: Iterable[str], source_name: str, what: str) -> None:
    for name in names:
        if name in ABI_NAMESPACES:
            raise TemplateError(
                f"{what} {name!r} is reserved for the ABI namespace",
                location=SourceLocation(1, 1, source=source_name),
            )


def has_markers(text: str) -> bool:
    """True when the source carries a Jinja layer that must be rendered first."""
    return any(marker in text for marker in MARKERS)


@dataclass(frozen=True, eq=False)
class TemplateInputs:
    """What a template is rendered with, besides its text."""

    #: Template values, checked against the declared parameters.
    defines: Mapping[str, object] = field(default_factory=dict)
    #: The declared parameters; if None, the source's `ipu:params` header, if
    #: any. A source that declares nothing takes no defines.
    params: ParamSpec | None = None
    #: Roots that `{% include %}` and `{% import %}` resolve in.
    search_path: tuple[Path, ...] = ()
    #: ABI names by namespace, `{"cr": {"data_base": "CR2"}}`; the template
    #: reads them as `{{ cr.data_base }}`. An unknown name fails.
    namespaces: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


def _context(text: str, inputs: TemplateInputs, source_name: str) -> dict[str, object]:
    """The render values: the bound defines and the ABI namespaces."""
    defines = dict(inputs.defines)
    abi = _abi_context(inputs.namespaces, source_name)
    _check_reserved(defines, source_name, "define")
    params = inputs.params if inputs.params is not None else declared_params(text, source_name)
    if params is None:
        if defines:
            raise TemplateError(
                f"defines given ({', '.join(sorted(defines))}) but the source declares no "
                f"parameters (add an ipu:params header)",
                location=SourceLocation(1, 1, source=source_name),
            )
        return dict(abi)
    _check_reserved((param.name for param in params.params), source_name, "parameter")
    return {**params.bind(defines, source_name), **abi}


def render(
    text: str,
    inputs: TemplateInputs | None = None,
    *,
    source_name: str = "<asm>",
    transform: SourceTransform | None = None,
) -> str:
    """Render one assembly source through the sandbox.

    Args:
        text: The source.
        inputs: Its defines, declared parameters, search path and ABI names.
        source_name: The source's name in error locations.
        transform: Applied to the source and to every loaded template first.

    Returns:
        The rendered text; `text` itself when it has no Jinja markers.

    Raises:
        TemplateError: a bad define or ABI name, or any jinja failure (syntax,
            undefined name, sandbox escape, missing template, an expression
            raising), located in the template it happened in, with the jinja2
            exception as `__cause__`.
    """
    inputs = inputs or TemplateInputs()
    context = _context(text, inputs, source_name)
    if not has_markers(text):
        return text
    if transform is not None:
        text = transform(source_name, text)
    environment = _environment(inputs.search_path, transform)
    try:
        return environment.from_string(text).render(**context)
    except Exception as error:
        loaded = environment.loader.loaded if environment.loader is not None else {}
        location = _template_location(error, source_name, loaded)
        raise TemplateError(str(error), location=location) from error


def _template_location(
    error: BaseException, source_name: str, loaded: Mapping[str, str]
) -> SourceLocation | None:
    """Where a jinja failure happened: the template and line it names or ran at.

    A syntax error names its template and line. A runtime error's traceback has
    a frame per template line jinja ran (jinja rewrites it so); the innermost
    one is where it failed. `loaded` maps a loaded template's filename to its name.
    """
    if isinstance(error, jinja2.TemplateSyntaxError) and error.lineno:
        return SourceLocation(error.lineno, 1, source=error.name or source_name)
    found = None
    tb = error.__traceback__
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if filename in ("<template>", "<unknown>"):
            found = SourceLocation(tb.tb_lineno, 1, source=source_name)
        elif filename in loaded:
            found = SourceLocation(tb.tb_lineno, 1, source=loaded[filename])
        tb = tb.tb_next
    return found
