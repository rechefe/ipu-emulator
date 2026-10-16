"""The Jinja tags of an assembly source, as Jinja itself reads them.

Where each `{# #}`, `{% %}` and `{{ }}` tag is comes from Jinja's lexer, and
the names a tag defines and uses from Jinja's parser, so the editor's own
reading of the template layer (vscode-ipu-asm/language.js) can be checked
against Jinja's rather than against a copy of itself. Nothing is rendered.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import jinja2
import jinja2.nodes as nodes

import ipu_as.template as template
from ipu_as.errors import AssemblerError

__all__ = ["TagSpan", "header_names", "is_params_header", "tag_names", "tag_spans"]


@dataclass(frozen=True)
class TagSpan:
    """One Jinja tag, as offsets in its source.

    Attributes:
        kind: `comment`, `block` or `variable`.
        start: The offset of its opening delimiter.
        end: The offset just past its closing one (the end of the source if
            it is unclosed).
    """

    kind: str
    start: int
    end: int


#: Jinja's token types that open a region, by the region kind.
_OPENERS = {"comment_begin": "comment", "block_begin": "block", "variable_begin": "variable"}
#: Jinja's token types that close a region, by the delimiter their value holds.
_CLOSERS = {"comment_end": "#}", "block_end": "%}", "variable_end": "}}"}
#: `{% raw %}` and `{% endraw %}`: each one token, and each a block region.
_RAW = ("raw_begin", "raw_end")
#: Values Jinja rewrites (whitespace control, newline normalisation), and which
#: cannot hold a delimiter, so they are not looked for in the text.
_UNLOCATED = ("data", "comment")


def tag_spans(text: str) -> tuple[TagSpan, ...]:
    """The Jinja regions of `text`, from Jinja's own lexer.

    The lexer gives no offsets, and its values need not concatenate back to the
    text (whitespace control strips data, a closer carries the whitespace it
    removes). But each value inside a tag is a literal substring of the text,
    in order, so each is found by searching forward from the last.

    Args:
        text: An assembly source.

    Returns:
        Its tags in order. `{% raw %}` and `{% endraw %}` are block tags,
        the text between them none. An unclosed tag runs to the end of the
        text, and lexing stops at the first error Jinja reports.
    """
    spans: list[TagSpan] = []
    opened: TagSpan | None = None
    position = 0
    try:
        for _, kind, value in template.lex(text):
            if kind in _UNLOCATED:
                continue
            at = text.index(value, position)
            position = at + len(value)
            if kind in _RAW:
                spans.append(TagSpan("block", at, at + value.rindex("%}") + 2))
            elif kind in _OPENERS:
                opened = TagSpan(_OPENERS[kind], at, len(text))
            elif kind in _CLOSERS and opened is not None:
                spans.append(
                    TagSpan(opened.kind, opened.start, at + value.index(_CLOSERS[kind]) + 2)
                )
                opened = None
    except (jinja2.TemplateSyntaxError, ValueError):
        pass  # the text stops lexing here; what came before stands
    if opened is not None:
        spans.append(opened)
    return tuple(spans)


#: The keyword of a block tag: `{%- for` -> `for`.
_KEYWORD = re.compile(r"\{%[-+]?\s*(\w+)")
#: The start of the `ipu:params` header comment (`ipu_as.template.declared_params`).
_HEADER = re.compile(r"\{#-?\s*ipu:params[ \t]*\n")


def _names(tree: nodes.Node) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(Names bound, names read) in a Jinja syntax tree, each once, in order."""
    defines: dict[str, None] = {}
    uses: dict[str, None] = {}
    wanted = (nodes.Name, nodes.NSRef, nodes.Macro, nodes.Import, nodes.FromImport)
    for node in tree.find_all(wanted):
        if isinstance(node, nodes.Name):
            (uses if node.ctx == "load" else defines)[node.name] = None
        elif isinstance(node, nodes.NSRef):
            uses[node.name] = None
        elif isinstance(node, nodes.Macro):
            defines[node.name] = None
        elif isinstance(node, nodes.Import):
            defines[node.target] = None
        else:
            for entry in node.names:
                defines[entry if isinstance(entry, str) else entry[1]] = None
    return tuple(defines), tuple(uses)


@functools.lru_cache(maxsize=4096)
def tag_names(tag: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(Names bound, names read) by one `{% %}` or `{{ }}` tag, from Jinja's parser.

    Bound: `set` and `for` targets, a macro and its parameters, `import`
    names. Read: every name loaded, Jinja globals (`range`, `loop`) included;
    an attribute (`x.name`) or a string is not a name.

    A tag that opens a block (`for`, `macro`, `set x`, `if`) is parsed with its
    end tag added; one inside a block (`elif`, `else`) inside an `if`. An end
    tag, or a tag Jinja cannot parse, names nothing.

    Args:
        tag: The tag's text, delimiters included.

    Returns:
        (defines, uses), each name once, in order.
    """
    keyword = _KEYWORD.match(tag)
    candidates = [tag]
    if keyword is not None:
        end = f"{{% end{keyword.group(1)} %}}"
        candidates += [tag + end, "{% if 1 %}" + tag + "{% endif %}"]
    for candidate in candidates:
        try:
            return _names(template.parse(candidate))
        except jinja2.TemplateSyntaxError:
            continue
    return (), ()


def header_names(text: str) -> tuple[str, ...]:
    """The parameter names the `ipu:params` header of `text` declares.

    Args:
        text: An assembly source.

    Returns:
        The names in order; () without a header, or if it is malformed (the
        render reports that).
    """
    try:
        spec = template.declared_params(text)
    except AssemblerError:
        return ()
    return () if spec is None else tuple(param.name for param in spec.params)


def is_params_header(tag: str) -> bool:
    """True when a `{# #}` tag is the `ipu:params` header (`ipu_as.template.declared_params`)."""
    return _HEADER.match(tag) is not None
