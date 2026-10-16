"""Assemble a kernel's program for one case (E06-07, Q80).

The program is rendered through the assembler's one render path (E05's
session and sandbox) with the case as template values: every parameter, the
reserved ``format`` (``"e4"``) and ``shapes`` (each tensor's shape), and the
``[abi]`` register names as ``{{ cr.x }}`` / ``{{ lr.i }}``. The declared
parameters are the kernel's: ``kernel.toml`` is the declaration, so the
program repeats none in an ``ipu:params`` header, and any other name is a
located error (the sandbox's strict undefined).

An :class:`AssemblyCache` encodes each distinct rendered program once: two
cases that render alike (a template that does not use ``rows`` renders alike
for every ``rows``) share one assembly, and an included file is part of what
renders. The harness imports no ``jinja2``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import get_args

from ipu_as.session import AssemblyResult, AssemblySession
from ipu_as.template import REQUIRED, Param, ParamSpec
from ipu_emu.errors import ConfigurationError
from ipu_harness.abi import template_names
from ipu_harness.manifest import ParamType
from ipu_harness.params import CasePlan

__all__ = ["AssemblyCache", "assemble", "search_path", "template_values"]

_PY_TYPES: Mapping[str, type] = {"int": int, "float": float, "bool": bool, "str": str}
assert set(_PY_TYPES) == set(get_args(ParamType)), (
    "a kernel.toml parameter type has no template type"
)


def template_values(plan: CasePlan) -> Mapping[str, object]:
    """What the program is rendered with, besides the register names.

    Args:
        plan: The checked case.

    Returns:
        Every parameter, ``format`` (the element format's name) and ``shapes``
        (each tensor's shape, as a list of ints).
    """
    shapes = {name: list(shape) for name, shape in plan.shapes.items()}
    return MappingProxyType({**plan.params, "format": plan.format.value, "shapes": shapes})


def _declared(plan: CasePlan) -> ParamSpec:
    """The parameters the program may read: the kernel's, ``format`` and ``shapes``."""
    params = [
        Param(name, _PY_TYPES[decl.type], REQUIRED) for name, decl in plan.manifest.params.items()
    ]
    params += [Param("format", str, REQUIRED), Param("shapes", dict, REQUIRED)]
    return ParamSpec(tuple(params))


def search_path(plan: CasePlan, asm_lib: Path | None = None) -> tuple[Path, ...]:
    """Where ``{% include %}`` and ``{% import %}`` look.

    Args:
        plan: The checked case.
        asm_lib: The shared macro library (``ipu_kernels/asm_lib``, ``*.asm.j2``)
            when the caller has it (Bazel data); none, only the kernel's folders.

    Returns:
        The kernel's folder, its family's folder, then ``asm_lib``.

    Raises:
        ConfigurationError: ``asm_lib`` is given but is no directory.
    """
    directory = plan.manifest.directory
    if asm_lib is not None and not asm_lib.is_dir():
        raise ConfigurationError(f"{asm_lib}: the macro library is not there")
    return (directory, directory.parent, *(() if asm_lib is None else (asm_lib,)))


class AssemblyCache:
    """Assembled programs by their rendered text: one per distinct render.

    It holds every program it has assembled, unbounded, so it lives as long as
    one run of a kernel's cases (the runner's, E06-11); without one every call
    assembles. A hit raises the program's encoding warnings again.
    """

    def __init__(self) -> None:
        self._results: dict[str, AssemblyResult] = {}
        self.assembled = 0  #: how many programs were encoded (not served from the cache)

    def get(self, session: AssemblySession) -> AssemblyResult:
        """The session's program, encoded only if its rendered text is new.

        Args:
            session: The program's assembly session (rendered here).

        Returns:
            The assembled program, cached or new.
        """
        digest = hashlib.sha256(session.render().encode("utf-8")).hexdigest()
        cached = self._results.get(digest)
        if cached is not None:
            cached.reissue_warnings()
            return cached
        result = session.assemble()
        self.assembled += 1
        self._results[digest] = result
        return result


def assemble(
    plan: CasePlan, cache: AssemblyCache | None = None, *, asm_lib: Path | None = None
) -> AssemblyResult:
    """The kernel's program for ``plan``: rendered with its values and assembled.

    Args:
        plan: The checked case; nothing is checked again.
        cache: Reuses an earlier assembly of the same rendered text.
        asm_lib: The shared macro library, when the caller has it.

    Returns:
        The instruction words, labels and rendered text (E05).

    Raises:
        ipu_as.errors.TemplateError: An undefined name or a bad ``cr.``/``lr.``
            name in the program, located in its file and line.
        ipu_as.errors.ParseError: The rendered program does not parse.
        ipu_as.errors.EncodeError: An instruction does not encode.
    """
    manifest = plan.manifest
    session = AssemblySession(
        manifest.asm.read_text(encoding="utf-8"),
        defines=template_values(plan),
        params=_declared(plan),
        search_path=search_path(plan, asm_lib),
        abi=template_names(manifest),
        name=manifest.asm.name,
    )
    return (cache or AssemblyCache()).get(session)
