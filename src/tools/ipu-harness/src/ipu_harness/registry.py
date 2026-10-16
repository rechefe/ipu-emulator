"""Routing as a value: which kernel computes an op with these parameters, and why (E06-16).

A :class:`Registry` is built from ``kernel.toml`` files (:meth:`Registry.load`)
or from manifests in hand (:meth:`Registry.of`, the test seam). It holds no
per-operation knowledge: :meth:`Registry.resolve` asks every kernel of the op
whether it accepts the parameters, with the same checks ``plan_case`` runs
(``check_params``, E06-03), and picks the cheapest that does. Equal costs are
an error, never settled by discovery order: the answer must not change because
a folder was renamed.

Loading parses TOML only; no kernel Python is imported to route a query. Layer
adapters (``LAYER_ADAPTERS = {"Conv2d": fn, ...}`` in a family's
``layers.py``) are found at load and imported only when :meth:`from_layer`
first needs one. Nothing here is module-level mutable state; the one cached
value is :func:`default_registry`.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ipu_emu.errors import ConfigurationError
from ipu_harness.expressions import evaluate
from ipu_harness.manifest import TOML_NAME, KernelManifest, load_manifest
from ipu_harness.params import ParamCheck, PlanError, Value, check_params
from ipu_isa.formats import ElementFormat

__all__ = ["LayerAdapter", "Registry", "UnsupportedLayer", "Verdict", "default_registry"]

#: A layer adapter: ``(layer, input_shape) -> (op, params)``.
LayerAdapter = Callable[[Any, tuple[int, ...]], tuple[str, Mapping[str, object]]]

#: The name a family's ``layers.py`` declares its adapters under.
ADAPTERS_NAME = "LAYER_ADAPTERS"


class UnsupportedLayer(ValueError):
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
    def app_name(self) -> str | None:
        """The chosen kernel's name."""
        return self.kernel.name if self.kernel else None

    @property
    def label(self) -> str | None:
        """The chosen kernel's Bazel label, ``//kernels/<family>/<name>``."""
        if self.kernel is None:
            return None
        return f"//kernels/{self.kernel.directory.parent.name}/{self.kernel.directory.name}"

    def _notes(self) -> list[str]:
        return [f"runs in {', '.join(f.value for f in self.formats)}"] if self.kernel else []

    def to_dict(self) -> dict[str, Any]:
        """The same answer as :meth:`describe`, for tools (``query --json``); the
        keys of ``ipu_apps``' verdict, ``use`` naming the kernel's target."""
        return {
            "supported": self.supported,
            "reason": self.reason,
            "app": self.app_name,
            "use": self.label,
            "kwargs": {k: repr(v) for k, v in self.params.items()},
            "shapes": None,
            "notes": self._notes(),
            "alternatives": list(self.alternatives),
        }

    def describe(self) -> str:
        """A human-readable summary (what ``query`` prints)."""
        lines = [f"{'SUPPORTED' if self.supported else 'NOT SUPPORTED'}: {self.reason}"]
        if self.kernel:
            args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
            lines.append(f"  app:  {self.app_name}")
            lines.append(f"  use:  {self.label} ({args})")
        lines.extend(f"  note: {note}" for note in self._notes())
        if self.alternatives:
            lines.append(f"  also:  {', '.join(self.alternatives)}")
        return "\n".join(lines)


class _LayerAdapters(Mapping[str, LayerAdapter]):
    """The adapters of the families' ``layers.py`` files, imported on first use."""

    def __init__(self, files: Iterable[Path]) -> None:
        self._files = tuple(files)
        self._loaded: dict[str, LayerAdapter] | None = None

    def _table(self) -> dict[str, LayerAdapter]:
        if self._loaded is None:
            table: dict[str, LayerAdapter] = {}
            for path in self._files:
                for name, adapter in _adapters_in(path).items():
                    if name in table:
                        raise ConfigurationError(f"{path}: {name} has an adapter in two families")
                    table[name] = adapter
            self._loaded = table
        return self._loaded

    def __getitem__(self, name: str) -> LayerAdapter:
        return self._table()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._table())

    def __len__(self) -> int:
        return len(self._table())


def _adapters_in(path: Path) -> dict[str, LayerAdapter]:
    """``LAYER_ADAPTERS`` of one ``layers.py``, loaded from its file."""
    spec = importlib.util.spec_from_file_location(f"_ipu_layers_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"{path}: cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapters = getattr(module, ADAPTERS_NAME, None)
    if not isinstance(adapters, Mapping) or not all(callable(a) for a in adapters.values()):
        raise ConfigurationError(f"{path}: {ADAPTERS_NAME} must map layer names to functions")
    return dict(adapters)


def _cost(manifest: KernelManifest, check: ParamCheck) -> float:
    """The kernel's ``cost`` for these parameters. An expression that fails to
    evaluate is the kernel's bug, so it propagates rather than refusing."""
    if not isinstance(manifest.cost, str):
        return float(manifest.cost)
    names = {
        k: v for k, v in check.params.items() if isinstance(v, int) and not isinstance(v, bool)
    }
    return float(evaluate(manifest.cost, {**names, **check.derived}))


def _check_ties(manifests: Iterable[KernelManifest]) -> None:
    """Two kernels of one op with the same constant cost: no query could choose."""
    seen: dict[tuple[str, float], str] = {}
    for m in sorted(manifests, key=lambda m: m.name):
        if isinstance(m.cost, str):
            continue
        other = seen.setdefault((m.op, float(m.cost)), m.name)
        if other != m.name:
            raise ConfigurationError(
                f"{m.directory / TOML_NAME}: {other} and {m.name} both compute {m.op} at cost "
                f"{m.cost}; give one a different [kernel] cost (ties are never settled by name)"
            )


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


@dataclass(frozen=True)
class Registry:
    """Every kernel, and the routing over them.

    Attributes:
        manifests: The kernels, by name.
        invalid: The ``kernel.toml`` files that failed to load, by path, with
            their errors.
        adapters: Layer adapters, by layer class name.
    """

    manifests: Mapping[str, KernelManifest]
    invalid: Mapping[Path, ConfigurationError] = field(default_factory=lambda: MappingProxyType({}))
    adapters: Mapping[str, LayerAdapter] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        families: Iterable[str] | None = None,
        layers: Path | None = None,
    ) -> Registry:
        """The kernels under ``root`` (``<root>/<family>/<name>/kernel.toml``).

        Args:
            root: The kernel tree.
            families: Only these families (a per-kernel test, E06-18).
            layers: The directory holding ``<family>/layers.py`` (the
                ``ipu_kernels`` package); none, no adapters.

        Returns:
            The registry; a ``kernel.toml`` that fails to load is kept in
            ``invalid``, not raised.

        Raises:
            ConfigurationError: Two kernels share a name, or two of one op a
                constant cost.
        """
        wanted = None if families is None else set(families)
        manifests: dict[str, KernelManifest] = {}
        invalid: dict[Path, ConfigurationError] = {}
        family_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        for family in family_dirs:
            if wanted is not None and family.name not in wanted:
                continue
            for toml in sorted(family.glob(f"*/{TOML_NAME}")):
                try:
                    manifest = load_manifest(toml.parent)
                except ConfigurationError as error:
                    invalid[toml] = error
                    continue
                if manifest.name in manifests:
                    raise ConfigurationError(
                        f"{toml}: kernel {manifest.name} is also "
                        f"{manifests[manifest.name].directory / TOML_NAME}"
                    )
                manifests[manifest.name] = manifest
        _check_ties(manifests.values())
        files = []
        if layers is not None:
            names = [f.name for f in family_dirs if wanted is None or f.name in wanted]
            files = [
                layers / n / "layers.py" for n in names if (layers / n / "layers.py").is_file()
            ]
        return cls(MappingProxyType(manifests), MappingProxyType(invalid), _LayerAdapters(files))

    @classmethod
    def of(
        cls,
        manifests: Iterable[KernelManifest],
        adapters: Mapping[str, LayerAdapter] | None = None,
    ) -> Registry:
        """A registry of ``manifests`` and ``adapters`` (the test seam).

        Raises:
            ConfigurationError: Two kernels share a name, or two of one op a
                constant cost.
        """
        by_name: dict[str, KernelManifest] = {}
        for m in manifests:
            if m.name in by_name:
                raise ConfigurationError(f"kernel {m.name} is given twice")
            by_name[m.name] = m
        _check_ties(by_name.values())
        return cls(MappingProxyType(by_name), adapters=MappingProxyType(dict(adapters or {})))

    def kernel(self, name: str) -> KernelManifest:
        """The kernel ``name``.

        Raises:
            ConfigurationError: Its ``kernel.toml`` failed to load (that error),
                or there is no such kernel.
        """
        if name in self.manifests:
            return self.manifests[name]
        for toml, error in self.invalid.items():
            if toml.parent.name == name:
                raise error
        known = ", ".join(sorted(self.manifests)) or "none"
        raise ConfigurationError(f"unknown kernel {name!r} (kernels: {known})")

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
            ConfigurationError: The cheapest accepting kernels tie.
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

    def from_layer(self, layer: object, shape: Iterable[int]) -> Verdict:
        """Route ``layer`` applied to an input of ``shape``, through its adapter.

        Raises:
            UnsupportedLayer: No adapter takes the layer's type.
        """
        name = type(layer).__name__
        adapter = self.adapters.get(name)
        if adapter is None:
            known = ", ".join(sorted(self.adapters)) or "none"
            raise UnsupportedLayer(
                f"no adapter for layer type {name!r} (layers with one: {known}); add it to "
                f"{ADAPTERS_NAME} in the layers.py of the family that implements it"
            )
        op, params = adapter(layer, tuple(int(d) for d in shape))
        query: dict[str, Any] = dict(params)  # may name a format, like any query
        return self.resolve(op, **query)


#: The checkout or runfiles root: ``src/tools/ipu-harness/src/ipu_harness/`` is five down.
_ROOT = Path(__file__).parents[5]


@functools.cache
def default_registry() -> Registry:
    """The registry of the ``kernels/`` tree beside this package (a checkout or
    Bazel's runfiles), with the ``ipu_kernels`` package's layer adapters if it is
    importable. Built once."""
    spec = importlib.util.find_spec("ipu_kernels")
    locations = list(spec.submodule_search_locations or []) if spec else []
    layers = Path(locations[0]) if locations else None
    return Registry.load(_ROOT / "kernels", layers=layers)
