"""What a kernel declares about itself, and what a query gets back.

A :class:`KernelSpec` lives in the app package it describes -- next to the
``.asm`` and the harness, not in a central routing file. Adding a kernel is
therefore a purely additive change: write the app, declare a ``SPEC`` beside
it, and discovery picks it up.

The single most important rule here is that a spec's :attr:`KernelSpec.supports`
is the *only* statement of a kernel's domain. App constructors are expected to
delegate their guards to it (see :meth:`KernelSpec.check`) rather than restate
the same bounds, because a rule written twice drifts: before this registry
existed, ``softmax_columns_packed`` told callers to "use softmax_columns for
width >= 128" long after the real boundary had moved to 65.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

from ipu_emu.ipu_math import DType

from ipu_apps.kernel_registry.shapes import ShapeBundle

if TYPE_CHECKING:
    from ipu_apps.base import IpuApp

# A query is an op name plus free-form parameters (config + shapes). Kernels
# for different operations need genuinely different parameters -- softmax has
# n/width, convolution has channels/stride -- so this is deliberately not a
# fixed signature.
Params = Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionConfig:
    """Arithmetic and storage requirements, independent of the debugger UI.

    Native mode uses encoded elements; fp32/int32 use four-byte vector
    elements. Defaults match IpuState(), including unquantized wide output.
    """

    mode: Literal["native", "fp32", "int32"] = "native"
    dtype: DType = DType.INT8
    quantize_output: bool = False

    def __post_init__(self):
        if self.mode not in ("native", "fp32", "int32"):
            raise ValueError(f"unknown execution mode {self.mode!r}")
        if not isinstance(self.dtype, DType):
            raise TypeError("execution dtype must be a DType")


@dataclass(frozen=True)
class Support:
    """Whether a kernel handles a query, and why (or why not).

    A bare bool would lose the reason, which is the part a user actually needs
    when the answer is "no".
    """

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def yes(reason: str = "") -> Support:
    return Support(True, reason)


def no(reason: str) -> Support:
    return Support(False, reason)


@dataclass(frozen=True)
class KernelSpec:
    """A kernel's self-declaration: what it computes, and for which shapes.

    Attributes:
        name:       Unique kernel id, conventionally the app package name.
        op:         Operation implemented, e.g. ``"softmax"``. Queries are
                    routed by this first.
        variant:    Short label distinguishing kernels of the same op
                    (``"rows"``, ``"columns_packed"``).
        app_class:  The :class:`~ipu_apps.base.IpuApp` subclass to instantiate.
        asm:        Path to the kernel's ``.asm``, relative to the app package.
        package:    Package containing cases.py and assembly. Defaults to the
                    harness module's containing package, including when the
                    harness is defined in a separate app.py module.
        supports:   ``params -> Support``. The single source of truth for this
                    kernel's domain.
        build:      ``params -> ctor kwargs`` (beyond inst/input/output paths).
        explain:    ``params -> str``, why this kernel suits these params.
        caveats:    ``params -> tuple[str, ...]``, non-blocking limits that
                    still apply. Computed per query so they can quantify the
                    actual cost (e.g. how many elements idle at this width).
        cost:       ``params -> float``, lower wins when several kernels claim
                    the same query. Making this explicit keeps resolution from
                    depending on discovery order.
        requires:   Parameter names this kernel's callbacks index. Checked
                    before any of them runs, so an omitted parameter is a
                    refusal naming what is missing rather than a ``KeyError``
                    escaping from inside ``supports``. Declaring it also keeps
                    a genuine ``KeyError`` *inside* a callback recognisable as
                    the kernel bug it is, instead of being guessed at and
                    reported as a routing miss.
        bundle:     Optional ``params -> ShapeBundle``, reporting the shapes as
                    the kernel actually understands them. Needed whenever a
                    kernel reinterprets the query -- flattening a rank>2 input,
                    deriving an output shape -- so the verdict can disclose it
                    rather than silently routing something the caller did not
                    literally ask for.
        tags:       Free-form markers for reporting (e.g. ``"fp32-wide"``).
        execution:  Fixed state configuration, or a selector receiving the
                    constructed harness with its normalized parameters.
    """

    name: str
    op: str
    app_class: type
    supports: Callable[..., Support]
    build: Callable[..., dict[str, Any]]
    explain: Callable[..., str]
    variant: str = ""
    asm: str | None = None
    requires: tuple[str, ...] = ()
    caveats: Callable[..., tuple[str, ...]] = lambda **_: ()
    cost: Callable[..., float] = lambda **_: 0.0
    bundle: Callable[..., ShapeBundle] | None = None
    tags: tuple[str, ...] = ()
    execution: ExecutionConfig | Callable[[IpuApp], ExecutionConfig] = field(
        default_factory=ExecutionConfig
    )
    package: str | None = None

    @property
    def resource_package(self) -> str:
        package = self.package or import_module(self.app_class.__module__).__package__
        if not package:
            raise ValueError(f"{self.name}: declare SPEC.package for cases and assembly")
        return package

    def missing_params(self, params: Params) -> tuple[str, ...]:
        """Names from :attr:`requires` that ``params`` does not carry."""
        return tuple(name for name in self.requires if name not in params)

    def check(self, **params: Any) -> Support:
        """Evaluate :attr:`supports`, normalising a bare bool to a Support."""
        absent = self.missing_params(params)
        if absent:
            return Support(
                False,
                f"needs parameter(s) {', '.join(repr(n) for n in absent)}, "
                f"which this query does not carry",
            )
        result = self.supports(**params)
        if isinstance(result, Support):
            return result
        return Support(bool(result), "" if result else "unsupported")

    def guard(self, **params: Any) -> None:
        """Raise ``ValueError`` unless this kernel supports ``params``.

        App constructors call this instead of hand-writing bounds checks, so
        the guard and the registry can never disagree about the domain.
        """
        verdict = self.check(**params)
        if not verdict.ok:
            raise ValueError(verdict.reason)


@dataclass(frozen=True)
class Verdict:
    """The answer to one registry query.

    Attributes:
        supported:  True if a kernel covers this query.
        reason:     Why this kernel was chosen, or why nothing covers it.
        kernel:     The chosen :class:`KernelSpec` (None if unsupported).
        kwargs:     Constructor keyword arguments for ``kernel.app_class``.
        shapes:     The (possibly reinterpreted) shape bundle actually routed.
        caveats:    Non-blocking limits that still apply.
        alternatives: Other kernels that also claimed the query, in cost order.
    """

    supported: bool
    reason: str
    kernel: KernelSpec | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    shapes: ShapeBundle | None = None
    caveats: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.supported

    @property
    def app_name(self) -> str | None:
        return self.kernel.name if self.kernel else None

    @property
    def app_class(self) -> str | None:
        if not self.kernel:
            return None
        cls = self.kernel.app_class
        return f"{cls.__module__}.{cls.__qualname__}"

    def describe(self) -> str:
        """Human-readable multi-line summary (what the CLI prints)."""
        head = "SUPPORTED" if self.supported else "NOT SUPPORTED"
        lines = [f"{head}: {self.reason}"]
        if self.kernel:
            args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
            lines.append(f"  app:  {self.kernel.name}")
            lines.append(
                f"  use:  {self.app_class}(inst_path=..., input_path=..., "
                f"output_path=..., {args})"
            )
        if self.shapes is not None:
            lines.append(f"  shapes: {self.shapes.describe()}   (* = derived)")
            for note in self.shapes.notes:
                lines.append(f"  note: {note}")
        for c in self.caveats:
            lines.append(f"  note: {c}")
        if self.alternatives:
            lines.append(f"  also:  {', '.join(self.alternatives)}")
        return "\n".join(lines)
