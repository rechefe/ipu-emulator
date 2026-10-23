"""The checked declarations of a ``kernel.toml`` (for :mod:`ipu_harness.manifest`).

The frozen dataclasses :func:`ipu_harness.manifest.load_manifest` returns, and
the keys each table of the schema takes. Import them from
:mod:`ipu_harness.manifest`.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from ipu_harness.conditions import Condition
from ipu_harness.expressions import placed_reads
from ipu_harness.tolerance import Tier as ToleranceTier
from ipu_harness.variants import Variant

ParamType = Literal["int", "float", "bool", "str"]
Unit = Literal["count", "rows", "bytes"]
Role = Literal["input", "output", "scratch"]
Store = Literal["element"]
Tier = Literal["default", "nightly"]

ROLES: Mapping[str, Role] = {"inputs": "input", "outputs": "output", "scratch": "scratch"}
#: The section each role is declared in.
SECTION_OF: Mapping[Role, str] = {role: section for section, role in ROLES.items()}

#: The keys each table takes (``manifest.SCHEMA_KEYS`` lists them for the
#: reference page).
BENCHMARK_KEYS = ("points", "per", "max_cycles")
KERNEL_KEYS = ("op", "formats", "asm", "stages", "cost")
PARAM_KEYS = ("type", "min", "max", "multiple_of", "choices", "default", "unit", "query_required")
TENSOR_KEYS: Mapping[Role, tuple[str, ...]] = {
    "input": ("shape", "row", "after", "when", "pack", "range"),
    "output": ("shape", "row", "after", "when", "pack", "store"),
    "scratch": ("shape", "row", "after", "when", "fill", "bytes"),
}
CASE_KEYS = ("formats", "inputs", "expected", "generator", "tier")
GOLDEN_KEYS = ("reference", "op", "tolerance")
TIER_KEYS = ("tier", "rtol", "atol", "ulps", "why")
BINDING_KEYS = ("register", "offset", "when")


@dataclass(frozen=True)
class ParamDecl:
    """A parameter and its domain.

    Attributes:
        name: Its name in ``[params]``.
        type: ``int``, ``float``, ``bool`` or ``str``.
        min: Smallest value (int and float).
        max: Largest value (int and float).
        multiple_of: Values are multiples of this (int).
        choices: The only values allowed, if given.
        default: The value when a case or query gives none (None: required).
        unit: What an int counts: ``count``, ``rows`` or ``bytes`` (L16).
        query_required: A routing query must give it (``Registry.resolve``).
    """

    name: str
    type: ParamType
    min: int | float | None = None
    max: int | float | None = None
    multiple_of: int | None = None
    choices: tuple[object, ...] | None = None
    default: object | None = None
    unit: Unit = "count"
    query_required: bool = False


@dataclass(frozen=True)
class TensorDecl:
    """An input, output or scratch region.

    Attributes:
        name: Its name in its section.
        role: ``input``, ``output`` or ``scratch``.
        shape: Ints, or expressions over parameter and derived names (E06-03).
        row: The first XMEM row, if pinned (Q81): an int, or an expression over
            parameter and derived names.
        after: The tensor this region is placed right after, if it says.
        when: The condition it exists under (``when = "bias"``), if any.
        pack: The named pack transform of an input or output (E06-05).
        store: How an output is stored: ``element`` (``accumulator`` left with
            ST.ACC, ADR 0007).
        fill: The value a scratch region is filled with: a number, or an
            expression over parameters and derived values, evaluated exactly
            (``"1 / channels"``) and rounded to the format when stored.
        content: ``"file.py:fn"`` giving a scratch region's bytes, if it says.
        value_range: ``[lo, hi]`` per format for an input's generated values
            (``range``, E06-08); a format not listed takes the default.
    """

    name: str
    role: Role
    shape: tuple[int | str, ...]
    row: int | str | None = None
    after: str | None = None
    when: Condition | None = None
    pack: str | None = None
    store: Store = "element"
    fill: int | float | str = 0
    content: str | None = None
    value_range: Mapping[Variant, tuple[float, float]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def section(self) -> str:
        """Its section in ``kernel.toml``: ``inputs``, ``outputs`` or ``scratch``."""
        return SECTION_OF[self.role]


@dataclass(frozen=True)
class CaseDecl:
    """A case from ``[cases.<name>]``.

    Attributes:
        name: The case's name.
        values: Parameter values; a tuple is a sweep axis.
        formats: The formats it runs in, if it limits them.
        inputs: User input files, by tensor name (Q82).
        expected: User expected outputs, by tensor name: a ``.npy`` file, or a
            ``"file.py:fn"`` whose ``fn(plan)`` returns the values.
        generator: ``"file.py:fn"`` that makes inputs instead.
        tier: ``default`` or ``nightly`` (Q73).
    """

    name: str
    values: Mapping[str, object]
    formats: tuple[Variant, ...] | None = None
    inputs: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    expected: Mapping[str, Path | str] = field(default_factory=lambda: MappingProxyType({}))
    generator: str | None = None
    tier: Tier = "default"


@dataclass(frozen=True)
class BenchmarkDecl:
    """The ``[benchmark]`` table (E06-11): sizes to time the kernel at.

    Attributes:
        points: Parameter values, each over the default case's: one run per
            point and format (paired values, not a sweep).
        per: An int parameter cycles are divided by in the report, if any.
        max_cycles: Each run's cycle limit, if not the model's.
    """

    points: tuple[Mapping[str, object], ...]
    per: str | None = None
    max_cycles: int | None = None


@dataclass(frozen=True)
class GoldenDecl:
    """How outputs are checked.

    Attributes:
        reference: ``"file.py:fn"`` in the kernel directory.
        op: A shared reference in ``ipu_kernels.reference``.
        tolerance: The tier per format it names (E06-10).
    """

    reference: str | None = None
    op: str | None = None
    tolerance: Mapping[Variant, ToleranceTier] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, eq=False)
class KernelManifest:
    """The checked, frozen view of one ``kernel.toml`` (compared and hashed by identity).

    Attributes:
        name: The folder's name, or the registry entry's (E06-17).
        directory: The kernel's folder.
        asm: The program.
        op: The operation.
        formats: The element formats it runs in.
        params: The parameters, by name.
        tensors: Inputs, outputs and scratch regions, by name.
        abi: Name to ``"CR2"``..``"CR15"`` or ``"LR0"``..``"LR15"`` (``"CR0"`` for a
            tensor pinned at row 0).
        abi_offsets: A tensor's offset from its base row, when ``[abi]`` gives one.
        abi_when: A binding's condition, when it has one (a conditional tensor's
            included).
        derived: Name to expression; a placed one (:attr:`placed`) reads a
            region's base.
        cases: The cases, ``default`` among them.
        golden: How outputs are checked.
        stages: ``"stages.py:run"`` for a multi-program pipeline (E06-14).
        cost: Routing cost: a number, or an expression over integer
            parameters and derived values (E06-16).
        fixed: Parameters a registry entry fixed (E06-17), with their values.
        doc: A registry entry's one-line description, if any.
        registry: The ``registry.toml`` that declares this kernel, for an entry.
    """

    name: str
    directory: Path
    asm: Path
    op: str
    formats: tuple[Variant, ...]
    params: Mapping[str, ParamDecl]
    tensors: Mapping[str, TensorDecl]
    abi: Mapping[str, str]
    derived: Mapping[str, str]
    cases: Mapping[str, CaseDecl]
    golden: GoldenDecl
    stages: str | None = None
    fixed: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    cost: int | float | str = 0
    abi_offsets: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    abi_when: Mapping[str, Condition] = field(default_factory=lambda: MappingProxyType({}))
    doc: str | None = None
    registry: Path | None = None
    benchmark: BenchmarkDecl | None = None

    @property
    def label(self) -> str:
        """Its Bazel label: ``//kernels/<family>/<name>`` for a folder kernel,
        ``//kernels/<family>:<name>`` for a registry entry."""
        if self.registry is not None:
            return f"//kernels/{self.registry.parent.name}:{self.name}"
        return f"//kernels/{self.directory.parent.name}/{self.directory.name}"

    @functools.cached_property
    def placed(self) -> Mapping[str, frozenset[str]]:
        """The ``[derived]`` values that read ``base(<tensor>)`` or ``row_bytes``,
        directly or through another, with the tensors whose bases each reads:
        known once the case's regions (and format) are, so only ``[abi]`` reads
        them."""
        return MappingProxyType(placed_reads(self.derived))

    @property
    def byte_units(self) -> bool:
        """Whether a parameter counts bytes (a legacy kernel, L16)."""
        return any(p.unit == "bytes" for p in self.params.values())
