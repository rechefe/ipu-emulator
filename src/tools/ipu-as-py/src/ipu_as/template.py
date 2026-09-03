"""The Jinja preprocessing layer of an assembly source, rendered safely.

Assembly text is rendered before it is parsed, so an `.asm` file is executable
input rather than inert data. `jinja2.Template` is not a sandbox: from any
expression, `{{ ''.__class__.__mro__ }}` walks to `os`, so

    {{ self.__init__.__globals__.__builtins__.__import__("os")
       .popen("…").read() }}

runs a shell command during assembly, and the source still assembles cleanly
afterwards because rendering succeeded. Assembling a kernel someone sent you is
then indistinguishable from running whatever they wanted to run.

So every render of assembly source goes through the sandbox here. A template
that reaches past what a kernel legitimately needs raises `SecurityError`,
which is a `TemplateRuntimeError` and therefore a `TemplateError`, so callers
that already handle template failures handle this one unchanged.

This bounds reach, not cost: the sandbox does not limit how long a render runs
or how much it allocates. `{% for _ in range(10**9) %}` still hangs. It stops
the escape, not the loop.
"""

from __future__ import annotations

from jinja2.sandbox import SandboxedEnvironment

#: A marker means the parser sees different text than the file holds, so a
#: position in the parsed text need not correspond to the file it came from.
MARKERS = ("{{", "{%", "{#")

#: One environment for every render: it carries no per-template state, and
#: constructing one per source is wasted work.
_ENVIRONMENT = SandboxedEnvironment()


def has_markers(text: str) -> bool:
    """True when the source carries a Jinja layer that must be rendered first."""
    return any(marker in text for marker in MARKERS)


def render(text: str) -> str:
    """Render one assembly source through the sandbox.

    Raises whatever the template raises: `TemplateSyntaxError` for a template
    that will not compile, `SecurityError` for one reaching outside the
    sandbox, and any Python exception the expressions themselves produce.
    """
    return _ENVIRONMENT.from_string(text).render()
