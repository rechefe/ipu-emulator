"""Routing as a value: which kernel computes an op with these parameters, and why (E06-16).

A :class:`Registry` is built from ``kernel.toml`` files (:meth:`Registry.load`)
or from manifests in hand (:meth:`Registry.of`, the test seam). It holds no
per-operation knowledge: :meth:`Registry.resolve` asks every kernel of the op
whether it accepts the parameters, with the same checks ``plan_case`` runs
(``check_params``, E06-03), and picks the cheapest that does. Two accepting
kernels at the same lowest cost are an error, never settled by name or
discovery order: the answer must not change because a folder was renamed.

Loading parses TOML only; no kernel Python is imported to route a query. Layer
adapters (``LAYER_ADAPTERS = {"Conv2d": fn, ...}`` in
``ipu_kernels.<family>.layers``) are imported by name, as plugins (Q62), only
when :meth:`from_layer` first needs one. Nothing here is module-level mutable
state; the one cached value is :func:`default_registry`.
"""

from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ipu_emu.errors import ConfigurationError
from ipu_harness.expressions import evaluate
from ipu_harness.family import REGISTRY_NAME, Entry, invalid_name, load_entry, read_registry
from ipu_harness.manifest import TOML_NAME, KernelManifest, load_manifest
from ipu_harness.params import ParamCheck, PlanError, Value, check_params
from ipu_isa.formats import ElementFormat

__all__ = [
    "LayerAdapter",
    "Registry",
    "UnsupportedLayer",
    "Verdict",
    "default_registry",
    "kernel_tree",
]

#: A layer adapter: ``(layer, input_shape) -> (op, params)``.
LayerAdapter = Callable[[Any, tuple[int, ...]], tuple[str, Mapping[str, object]]]

#: The name a family's ``layers.py`` declares its adapters under.
ADAPTERS_NAME = "LAYER_ADAPTERS"


class UnsupportedLayer(ConfigurationError):
    """No adapter translates this layer type into a query."""


@dataclass(frozen=True)
class Verdict:
    """The answer to one query.

    Attributes:
        supported: Whether a kernel accepts it.
        reason: Why that kernel, or why none.
        kernel: The chosen kernel, if any.
        params: Its parameters, defaults filled.
        formats: The formats it accepts them in.
        alternatives: The other kernels that accept it, cheapest first.
    """

    supported: bool
    reason: str
    kernel: KernelManifest | None = None
    params: Mapping[str, Value] = field(default_factory=lambda: MappingProxyType({}))
    formats: tuple[ElementFormat, ...] = ()
    alternatives: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.supported

    @property
    def kernel_name(self) -> str | None:
        """The chosen kernel's name."""
        return self.kernel.name if self.kernel else None

    @property
    def label(self) -> str | None:
        """The chosen kernel's Bazel label: ``//kernels/<family>/<name>``, or
        ``//kernels/<family>:<name>`` for a registry entry (E06-17)."""
        return None if self.kernel is None else self.kernel.label

    def _notes(self) -> list[str]:
        if self.kernel is None:
            return []
        notes = [f"runs in {', '.join(f.value for f in self.formats)}"]
        return [self.kernel.doc, *notes] if self.kernel.doc else notes

    def to_dict(self) -> dict[str, Any]:
        """The same answer as :meth:`describe`, for tools (``query --json``).

        Returns:
            ``app`` is the kernel's name (the VS Code extension reads it),
            ``use`` its Bazel label, and ``notes`` names the formats.
        """
        return {
            "supported": self.supported,
            "reason": self.reason,
            "app": self.kernel_name,
            "use": self.label,
            "notes": self._notes(),
            "alternatives": list(self.alternatives),
        }

    def describe(self) -> str:
        """A human-readable summary (what ``query`` prints)."""
        lines = [f"{'SUPPORTED' if self.supported else 'NOT SUPPORTED'}: {self.reason}"]
        if self.kernel:
            args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
            lines.append(f"  app:  {self.kernel_name}")
            lines.append(f"  use:  {self.label} ({args})")
        lines.extend(f"  note: {note}" for note in self._notes())
        if self.alternatives:
            lines.append(f"  also:  {', '.join(self.alternatives)}")
        return "\n".join(lines)


class _LayerAdapters(Mapping[str, LayerAdapter]):
    """The adapters of the families' ``layers`` modules, imported on first use.

    A failure is kept and raised again, so a broken ``layers.py`` is imported
    (and its side effects run) once.
    """

    def __init__(self, modules: Iterable[str]) -> None:
        self._modules = tuple(modules)
        self._table: dict[str, LayerAdapter] | None = None
        self._failure: ConfigurationError | None = None

    def _loaded(self) -> dict[str, LayerAdapter]:
        if self._failure is not None:
            raise self._failure
        if self._table is None:
            try:
                self._table = _collect(self._modules)
            except ConfigurationError as error:
                self._failure = error
                raise
        return self._table

    def __getitem__(self, name: str) -> LayerAdapter:
        return self._loaded()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._loaded())

    def __len__(self) -> int:
        return len(self._loaded())


def _collect(modules: Iterable[str]) -> dict[str, LayerAdapter]:
    """``LAYER_ADAPTERS`` of each ``layers`` module that exists, merged."""
    table: dict[str, LayerAdapter] = {}
    origin: dict[str, str] = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name is not None and (name + ".").startswith(error.name + "."):
                continue  # this family (or the package) declares no adapters
            raise ConfigurationError(f"{name}: cannot be imported: {error}") from error
        except Exception as error:
            raise ConfigurationError(
                f"{name}: cannot be imported: {type(error).__name__}: {error}"
            ) from error
        adapters = getattr(module, ADAPTERS_NAME, None)
        if not isinstance(adapters, Mapping) or not all(callable(a) for a in adapters.values()):
            raise ConfigurationError(f"{name}: {ADAPTERS_NAME} must map layer names to functions")
        for layer, adapter in adapters.items():
            if layer in table:
                raise ConfigurationError(
                    f"{layer} has an adapter in both {origin[layer]} and {name}"
                )
            table[layer], origin[layer] = adapter, name
    return table


def _cost(manifest: KernelManifest, check: ParamCheck) -> float:
    """The kernel's ``cost`` for these parameters. An expression that fails to
    evaluate is the kernel's bug, so it propagates rather than refusing."""
    if not isinstance(manifest.cost, str):
        return float(manifest.cost)
    names = {
        k: v for k, v in check.params.items() if isinstance(v, int) and not isinstance(v, bool)
    }
    return float(evaluate(manifest.cost, {**names, **check.derived}))


def _distinct(refusals: list[tuple[str, str]], candidates: int) -> list[str]:
    """Refusals, one line per distinct reason, naming the kernels that gave it."""
    voices: dict[str, list[str]] = {}
    for name, reason in refusals:
        voices.setdefault(reason, []).append(name)
    lines = []
    for reason, names in voices.items():
        who = (
            f"every kernel ({', '.join(names)})"
            if len(names) == candidates > 1
            else ", ".join(names)
        )
        lines.append(f"{who}: {reason}")
    return lines


def _left_out(tied: Sequence[KernelManifest], params: Mapping[str, object]) -> list[str]:
    """The parameters the query leaves out that every tied kernel, each a
    registry entry, fixes, to more than one value: what it must give to choose
    among them. Empty when a tied kernel fixes none of them (a tie a query
    cannot settle, so a configuration error)."""
    names = set.intersection(*({*m.fixed} for m in tied)) - set(params)
    return [n for n in sorted(names) if len({repr(m.fixed[n]) for m in tied}) > 1]


def _declared_at(manifest: KernelManifest) -> str:
    """Where a kernel is declared, for a name clash."""
    if manifest.registry is not None:
        return f"{manifest.registry} (entry {manifest.name})"
    return str(manifest.directory / TOML_NAME)


class _Found:
    """What :meth:`Registry.load` reads, family by family; a name is one kernel's."""

    def __init__(self) -> None:
        self.manifests: dict[str, KernelManifest] = {}
        self.templates: dict[str, KernelManifest] = {}
        self.invalid: dict[Path, ConfigurationError] = {}

    def _add(self, manifest: KernelManifest, into: dict[str, KernelManifest]) -> None:
        for known in (self.manifests, self.templates):
            if manifest.name in known:
                raise ConfigurationError(
                    f"{_declared_at(manifest)}: kernel {manifest.name} is also "
                    f"{_declared_at(known[manifest.name])}"
                )
        into[manifest.name] = manifest

    def family(self, family: Path) -> None:
        registry = family / REGISTRY_NAME
        entries: tuple[Entry, ...] = ()
        if registry.is_file():
            try:
                entries = read_registry(registry)
            except ConfigurationError as error:
                self.invalid[registry] = error
        templates = {e.template for e in entries}
        for toml in sorted(family.glob(f"*/{TOML_NAME}")):
            try:
                manifest = load_manifest(toml.parent)
            except ConfigurationError as error:
                self.invalid[toml] = error
                continue
            self._add(manifest, self.templates if toml.parent.name in templates else self.manifests)
        for entry in entries:
            try:
                manifest = load_entry(registry, entry)
            except ConfigurationError as error:
                self.invalid[family / entry.name] = error
                continue
            self._add(manifest, self.manifests)


@dataclass(frozen=True)
class Registry:
    """Every kernel, and the routing over them.

    Attributes:
        manifests: The kernels routing chooses among, by name: folder kernels
            and ``registry.toml`` entries (E06-17).
        invalid: The kernels that failed to load, with their errors: a
            ``kernel.toml`` by its path, a ``registry.toml`` by its path, an
            entry by ``<family folder>/<name>``.
        adapters: Layer adapters, by layer class name.
        templates: The folders entries are made from, by name. A template is
            run and tested itself but not routed: its free domain would tie
            with its own entries.
    """

    manifests: Mapping[str, KernelManifest]
    invalid: Mapping[Path, ConfigurationError] = field(default_factory=lambda: MappingProxyType({}))
    adapters: Mapping[str, LayerAdapter] = field(default_factory=lambda: MappingProxyType({}))
    templates: Mapping[str, KernelManifest] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        families: Iterable[str] | None = None,
        package: str = "ipu_kernels",
    ) -> Registry:
        """The kernels under ``root`` (``<root>/<family>/<name>/kernel.toml``).

        Args:
            root: The kernel tree.
            families: Only these families (a per-kernel test, E06-18).
            package: The package whose ``<family>.layers`` modules hold the
                layer adapters (imported on first use; a family without one
                has none).

        Returns:
            The registry; a ``kernel.toml`` that fails to load is kept in
            ``invalid``, not raised.

        Raises:
            ConfigurationError: ``root`` is no directory, or two kernels share a
                name.
        """
        if not root.is_dir():
            raise ConfigurationError(f"{root}: no kernel tree there")
        wanted = None if families is None else set(families)
        found = _Found()
        family_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        for family in family_dirs:
            if wanted is None or family.name in wanted:
                found.family(family)
        names = [f.name for f in family_dirs if wanted is None or f.name in wanted]
        adapters = _LayerAdapters(f"{package}.{n}.layers" for n in names)
        return cls(
            MappingProxyType(found.manifests),
            MappingProxyType(found.invalid),
            adapters,
            MappingProxyType(found.templates),
        )

    @classmethod
    def of(
        cls,
        manifests: Iterable[KernelManifest],
        adapters: Mapping[str, LayerAdapter] | None = None,
    ) -> Registry:
        """A registry of ``manifests`` and ``adapters`` (the test seam).

        Raises:
            ConfigurationError: Two kernels share a name.
        """
        by_name: dict[str, KernelManifest] = {}
        for m in manifests:
            if m.name in by_name:
                raise ConfigurationError(f"kernel {m.name} is given twice")
            by_name[m.name] = m
        return cls(MappingProxyType(by_name), adapters=MappingProxyType(dict(adapters or {})))

    def kernel(self, name: str) -> KernelManifest:
        """The kernel ``name``: a routed kernel or a template.

        Raises:
            ConfigurationError: It failed to load (that error), or there is no
                such kernel; the message then names any ``registry.toml`` that
                did not read, whose entries are unknown.
        """
        if name in self.manifests:
            return self.manifests[name]
        if name in self.templates:
            return self.templates[name]
        for path, error in self.invalid.items():
            if invalid_name(path) == name:
                raise error
        known = ", ".join(sorted(self.manifests)) or "none"
        unread = [str(path) for path in self.invalid if path.name == REGISTRY_NAME]
        also = f"; registries that did not read: {', '.join(unread)}" if unread else ""
        raise ConfigurationError(f"unknown kernel {name!r} (kernels: {known}{also})")

    def operations(self) -> tuple[str, ...]:
        """Every op some kernel computes."""
        return tuple(sorted({m.op for m in self.manifests.values()}))

    def kernels(self, op: str | None = None) -> tuple[KernelManifest, ...]:
        """The kernels, or those of ``op``, by name."""
        found = (m for m in self.manifests.values() if op is None or m.op == op)
        return tuple(sorted(found, key=lambda m: m.name))

    def resolve(
        self, op: str, /, *, format: ElementFormat | str | None = None, **params: object
    ) -> Verdict:
        """The cheapest kernel of ``op`` that accepts ``params``.

        Args:
            op: The operation.
            format: Accept only a kernel that takes the parameters in this
                format; none, in at least one of its formats.
            **params: The kernel parameters.

        Returns:
            The verdict, truthy when a kernel accepts; else its reason names
            every refusal (an unknown parameter by name).

        Raises:
            ConfigurationError: The cheapest accepting kernels tie, and the
                query gives every parameter they fix (a tie the query left
                open by leaving one out is a refusal that names it).
            ExpressionError: A kernel's cost expression does not evaluate.
        """
        fmt = None if format is None else ElementFormat(format)
        candidates = self.kernels(op)
        if not candidates:
            known = ", ".join(self.operations()) or "none"
            return Verdict(
                False, f"no kernels are registered for operation {op!r} (known operations: {known})"
            )
        accepted: list[tuple[float, KernelManifest, ParamCheck]] = []
        refusals: list[tuple[str, str]] = []
        for m in candidates:
            unnamed = [n for n, p in m.params.items() if p.query_required and n not in params]
            if unnamed:
                refusals.append((m.name, f"the query must give {', '.join(unnamed)}"))
                continue
            try:
                check = check_params(m, params)
            except PlanError as error:
                refusals.append((m.name, error.reason))
                continue
            if fmt is not None and fmt not in check.formats:
                refusals.append((m.name, f"does not take these parameters in {fmt.value}"))
            elif not check.formats:
                refusals.append((m.name, "no format of the kernel encodes its element scalars"))
            else:
                accepted.append((_cost(m, check), m, check))
        if not accepted:
            detail = "; ".join(_distinct(refusals, len(candidates)))
            return Verdict(False, f"no {op} kernel covers this query -- {detail}")
        accepted.sort(key=lambda item: item[0])
        if len(accepted) > 1 and accepted[0][0] == accepted[1][0]:
            tied = [m for c, m, _ in accepted if c == accepted[0][0]]
            missing = _left_out(tied, params)
            if missing:
                return Verdict(
                    False,
                    f"{', '.join(m.name for m in tied)} each fix {', '.join(missing)}, "
                    f"which the query leaves out; give {' and '.join(missing)}",
                )
            a, b = accepted[0][1].name, accepted[1][1].name
            raise ConfigurationError(
                f"{a} and {b} both accept this {op} query at cost {accepted[0][0]:g}; "
                f"their [kernel] cost must tell them apart"
            )
        cost, best, check = accepted[0]
        return Verdict(
            True,
            f"{best.name} accepts it, at cost {cost:g}",
            kernel=best,
            params=check.params,
            formats=check.formats if fmt is None else (fmt,),
            alternatives=tuple(m.name for _, m, _ in accepted[1:]),
        )

    def from_layer(self, layer: object, shape: Sequence[int]) -> Verdict:
        """Route ``layer`` applied to an input of ``shape``, through its adapter.

        Args:
            layer: A framework layer, matched by its class name.
            shape: The shape of the tensor it is applied to.

        Returns:
            The verdict for the query the adapter makes of them.

        Raises:
            UnsupportedLayer: No adapter takes the layer's type.
            ConfigurationError: The adapter returned something other than
                ``(op, params)``.
        """
        name = type(layer).__name__
        adapter = self.adapters.get(name)
        if adapter is None:
            known = ", ".join(sorted(self.adapters)) or "none"
            raise UnsupportedLayer(
                f"no adapter for layer type {name!r} (layers with one: {known}); add it to "
                f"{ADAPTERS_NAME} in the layers.py of the family that implements it"
            )
        made = adapter(layer, tuple(int(d) for d in shape))
        if not (
            isinstance(made, tuple)
            and len(made) == 2
            and isinstance(made[0], str)
            and isinstance(made[1], Mapping)
        ):
            raise ConfigurationError(f"the {name} adapter returned {made!r}, not (op, params)")
        op, params = made
        query: dict[str, Any] = dict(params)  # it may name a format, as any query may
        return self.resolve(op, **query)


def kernel_tree() -> Path:
    """Where the ``kernels/`` tree is.

    Returns:
        ``$RUNFILES_DIR/_main/kernels`` when run by Bazel (``//kernels:tree``
        data), else the ``kernels/`` of the checkout this package is in.
    """
    runfiles = os.environ.get("RUNFILES_DIR")
    if runfiles and (Path(runfiles) / "_main" / "kernels").is_dir():
        return Path(runfiles) / "_main" / "kernels"
    return Path(__file__).parents[5] / "kernels"


@functools.cache
def default_registry() -> Registry:
    """The registry of :func:`kernel_tree` with ``ipu_kernels``' adapters, built once.

    Raises:
        ConfigurationError: There is no kernel tree (a binary without the
            ``//kernels:tree`` data).
    """
    return Registry.load(kernel_tree())
