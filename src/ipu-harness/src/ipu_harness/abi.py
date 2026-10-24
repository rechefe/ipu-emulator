"""Registers from ``[abi]``: region bases, integers, element scalars (E06-06).

``[abi]`` is the one register map. :func:`register_writes` gives the value
each bound register starts with; :func:`template_names` gives the program the
same names (``{{ cr.x }}`` renders ``CR2``), so the two cannot drift. After
assembly, :func:`check_cr_reads` fails a program that reads a CR nothing
binds. A kernel writes no register itself.

What a key binds decides the value:

=====================  =====================================================
a tensor               its base row (``Placement``); in a ``unit = "bytes"``
                       kernel (L16), ``base_row x row_bytes(format)``; plus
                       the binding's ``offset``, if it gives one. A tensor
                       bound to ``CR0`` (pinned at row 0) is written nowhere:
                       the program reads the hardwired 0.
an ``int`` parameter   the integer
a ``bool`` parameter   1 or 0
a ``float`` parameter  the CR word a MULT reads as it in the case's mode
                       (FP32: its float32 bits; NX8: the MULT constant
                       layout, ADR 0007; E06-03 checked it is exact)
a ``[derived]`` value  the integer (``-1``, ``"k - 128 * (ceil(k / 128) - 1) - 2"``);
                       a placed one (``"base(w) + 1"``) over the regions'
                       bases, in rows (bytes in a ``unit = "bytes"`` kernel),
                       ``row_bytes`` and ``case_qf`` in the case's variant
=====================  =====================================================

Values are raw register bits: a negative integer is its two's complement in
the register's width. Unless ``[abi]`` binds ``CR15``, it holds the default
DS in ``CR15.L`` (``ds(128)``: every element, one partition; ADR 0007).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ipu_as.session import AssemblySession
from ipu_emu.errors import ConfigurationError

# The emulator's default DS convention (CR15.L, CLAUDE.md).
from ipu_emu.ipu_config import DEFAULT_DS_CR
from ipu_emu.reference import ReferenceModel
from ipu_harness._codec import scalar_word
from ipu_harness.expressions import (
    CASE_QF_NAME,
    ROW_BYTES_NAME,
    ExpressionError,
    base_tensor,
    evaluate,
    names_in,
)
from ipu_harness.manifest import ZERO_REGISTER, KernelManifest
from ipu_harness.params import CasePlan, PlanError
from ipu_harness.placement import Placement
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES, LR_CR_SCALAR_VALUE_MASK
from ipu_isa.cr_halves import DEFAULT_DS
from ipu_isa.operand_types import OperandKind
from ipu_isa.view import Isa, isa

__all__ = ["RegisterWrite", "apply", "check_cr_reads", "register_writes", "template_names"]

_CR = re.compile(r"CR([0-9]+)(?:\.[LH])?")

#: Raw register bits hold a value in [-2^31, 2^32): a signed or an unsigned word.
_LOWEST, _PAST_HIGHEST = -(1 << 31), LR_CR_SCALAR_VALUE_MASK + 1


def _bank(register: str) -> str:
    """``"cr"`` or ``"lr"``: the register file's name for ``register``'s bank."""
    return register[:2].lower()


@dataclass(frozen=True)
class RegisterWrite:
    """One register's starting value.

    Attributes:
        register: ``"CR2"`` .. ``"CR15"`` or ``"LR0"`` .. ``"LR15"``.
        value: Its raw bits.
        source: What the value is, for a report: ``"x (region base)"``,
            ``"rows (int param)"``, ``"s (element scalar, e4)"``.
    """

    register: str
    value: int
    source: str

    @property
    def bank(self) -> str:
        """``"cr"`` or ``"lr"``: the register file's name for the bank."""
        return _bank(self.register)

    @property
    def index(self) -> int:
        """The register's number in its bank."""
        return int(self.register[2:])


def _element_bits(value: float, plan: CasePlan) -> int:
    """The CR word a MULT reads as ``value`` in ``plan.format``'s mode."""
    word = scalar_word(value, plan.format)
    if word is None:  # plan_case checked it encodes
        raise PlanError(plan.manifest, "params", f"{value!r} has no {plan.format.value} encoding")
    return word


def _placed_value(name: str, plan: CasePlan, placement: Placement) -> int:
    """A placed ``[derived]`` value: its expression over the integer parameters,
    the other derived values (placed ones computed as it reads them), the
    bases it reads (bytes in a ``unit = "bytes"`` kernel, as a tensor's
    binding), ``row_bytes`` (the bytes of one row in the case's variant) and
    ``case_qf`` (its QF half)."""
    manifest = plan.manifest
    expression = manifest.derived[name]
    scale = plan.format.row_bytes if manifest.byte_units else 1
    ints = {k: v for k, v in plan.params.items() if isinstance(v, int) and not isinstance(v, bool)}
    names: dict[str, int] = {**ints, **plan.derived}
    for read in names_in(expression):
        tensor = base_tensor(read)
        if tensor is not None:
            if tensor not in placement.regions:
                raise PlanError(
                    manifest,
                    f"derived.{name}",
                    f"{expression!r}: {tensor} is not placed in this case (its when is false)",
                )
            names[read] = placement.regions[tensor].base_row * scale
        elif read == ROW_BYTES_NAME:
            names[read] = plan.format.row_bytes
        elif read == CASE_QF_NAME:
            names[read] = plan.format.qf_half
        elif read in manifest.placed:
            names[read] = _placed_value(read, plan, placement)
    try:
        return evaluate(expression, names)
    except ExpressionError as error:
        raise PlanError(manifest, f"derived.{name}", f"{expression!r}: {error}") from None


def _value(
    name: str, plan: CasePlan, placement: Placement, offsets: Mapping[str, int]
) -> tuple[int, str]:
    """The bits ``name`` puts in its register, and what they are.

    ``offsets`` are the binding table's (the kernel's ``[abi]`` offsets, or
    none for a stage's table): a region's base row plus its offset.
    """
    manifest = plan.manifest
    if name in manifest.tensors:
        base = placement.region(name).base_row
        offset = offsets.get(name, 0)
        shown = f" {offset:+d}" if offset else ""
        if manifest.byte_units:
            bytes_base = base * plan.format.row_bytes
            return bytes_base + offset, f"{name} (region base, bytes{shown})"
        return base + offset, f"{name} (region base{shown})"
    if name in manifest.placed:
        return _placed_value(name, plan, placement), f"{name} (derived, placed)"
    if name in manifest.derived:
        return plan.derived[name], f"{name} (derived)"
    decl = manifest.params[name]
    value = plan.params[name]
    if decl.type == "float":
        return _element_bits(float(value), plan), f"{name} (element scalar, {plan.format.value})"
    # The loader binds no str parameter: what is left is an int or a bool.
    return int(value), f"{name} ({decl.type} param)"


def register_writes(
    plan: CasePlan,
    placement: Placement,
    *,
    abi: Mapping[str, str] | None = None,
    ds: int | None = None,
) -> tuple[RegisterWrite, ...]:
    """Every register's starting value, in ``[abi]`` order, then the default
    DS in ``CR15.L`` unless ``[abi]`` binds ``CR15``. A ``CR0`` binding writes
    nothing.

    Args:
        plan: The checked case.
        placement: Its regions.
        abi: Names to registers in place of the kernel's ``[abi]`` (a stage's,
            E06-14), already checked by ``manifest.check_abi``. The kernel's
            ``[abi]`` offsets do not apply to it: each region is its base.
        ds: ``CR15.L``'s DS half; ``None``: the default (``ds(128)``).

    Returns:
        The writes, values as raw register bits.

    Raises:
        PlanError: A value does not fit a 32-bit register.
        ConfigurationError: ``ds`` is given and ``abi`` binds ``CR15``.
    """
    bindings = plan.manifest.abi if abi is None else abi
    # An [abi] offset belongs to the kernel's own binding: a stage's table
    # binds each region at its base.
    offsets = plan.manifest.abi_offsets if abi is None else {}
    cr15 = f"CR{DEFAULT_DS_CR}"
    if ds is not None and cr15 in bindings.values():
        raise ConfigurationError(
            f"{plan.manifest.name}: the abi binds {cr15} and a DS is given for it"
        )
    writes: list[RegisterWrite] = []
    for name, register in bindings.items():
        value, source = _value(name, plan, placement, offsets)
        if register == ZERO_REGISTER:
            if value:  # the loader pins it at row 0; a byte-unit base is 0 too
                raise PlanError(
                    plan.manifest, f"abi.{name}", f"{source} = {value}, but CR0 always reads 0"
                )
            continue
        if not _LOWEST <= value < _PAST_HIGHEST:
            raise PlanError(
                plan.manifest, f"abi.{name}", f"{source} = {value} does not fit a 32-bit register"
            )
        bits = value & LR_CR_SCALAR_VALUE_MASK
        clash = next((w for w in writes if w.register == register and w.value != bits), None)
        if clash is not None:  # the loader lets two bindings share only an equal value
            raise PlanError(
                plan.manifest,
                f"abi.{name}",
                f"{source} = {value}, but {clash.source} put {clash.value} in {register}",
            )
        writes.append(RegisterWrite(register, bits, source))
    if cr15 not in bindings.values():
        if ds is None:
            bits, source = DEFAULT_DS, "DS in CR15.L (default, ds(128))"
        else:
            bits, source = ds, f"DS in CR15.L ({ds:#06x})"
        writes.append(RegisterWrite(cr15, bits, source))
    return tuple(writes)


def template_names(
    manifest: KernelManifest, abi: Mapping[str, str] | None = None
) -> Mapping[str, Mapping[str, str]]:
    """The program's register names (E06-07).

    Args:
        manifest: The kernel.
        abi: Names to registers in place of its ``[abi]`` (a stage's, E06-14).

    Returns:
        ``{"cr": {"x": "CR2", ...}, "lr": {...}}``, read as ``{{ cr.x }}`` and
        ``{{ lr.i }}``.
    """
    banks: dict[str, dict[str, str]] = {"cr": {}, "lr": {}}
    for name, register in (manifest.abi if abi is None else abi).items():
        banks[_bank(register)][name] = register
    return MappingProxyType({bank: MappingProxyType(names) for bank, names in banks.items()})


def apply(model: ReferenceModel, writes: Iterable[RegisterWrite]) -> None:
    """Write registers through the reference-model API (E04-30), by bank name.

    Args:
        model: The model, halted (after a reset or a program load).
        writes: From :func:`register_writes`.

    Raises:
        ConfigurationError: The model refuses a write (it is running, or the
            register is CR0 or CR1).
    """
    for write in writes:
        model.write_register(write.bank, write.value, write.index)


def _cr_types(view: Isa) -> frozenset[str]:
    """Register operand types whose values include a CR (an LR-or-CR type reads a
    CR when it is spelled ``CRn``), from the ISA's operand-type table."""
    return frozenset(
        name
        for name, t in view.operand_types.items()
        if t.kind is OperandKind.REGISTER
        and any(v.lower().startswith("cr") for v in t.domain.members)
    )


def _operand_types(view: Isa) -> dict[str, list[str]]:
    """Each mnemonic's operand types, by name (NOP, in every slot, has none)."""
    types: dict[str, list[str]] = {}
    for slot in view.spec_slot_order:
        for instruction in view.instructions(slot):
            types.setdefault(
                instruction.mnemonic.lower(), [op.type.name for op in instruction.operands]
            )
    return types


def check_cr_reads(
    manifest: KernelManifest,
    rendered: str,
    *,
    abi: Mapping[str, str] | None = None,
    name: str | None = None,
) -> None:
    """Fail a rendered program that reads a CR neither hardwired nor bound.

    ``CR0`` and ``CR1`` are hardwired; ``CR15`` holds the default DS unless
    bound; every other CR a program reads, whole or a half (``CR13.L``), must
    be in ``[abi]``.

    Args:
        manifest: The kernel.
        rendered: Its program with the template rendered (E06-07's
            ``AssemblyResult.rendered``); it is parsed, not rendered again.
        abi: The names bound in place of its ``[abi]`` (a stage's, E06-14).
        name: The program's file name for messages; its ``.asm`` by default.

    Raises:
        ConfigurationError: A read of an unbound CR, naming the line and its
            text. The line is the rendered program's: the program file's own
            line when it has no template layer.
    """
    bindings = manifest.abi if abi is None else abi
    program = manifest.asm.name if name is None else name
    bound = {int(r[2:]) for r in bindings.values() if _bank(r) == "cr"}
    bound |= {*CR_READ_ONLY_INITIAL_VALUES, DEFAULT_DS_CR}
    view = isa()
    types, cr_types = _operand_types(view), _cr_types(view)
    lines = rendered.splitlines()
    for word in AssemblySession("", name=program).parse_rendered(rendered):
        for instruction in word["instructions"]:
            mnemonic = instruction["opcode"].token.value
            for kind, operand in zip(types[mnemonic.lower()], instruction["operands"], strict=True):
                m = _CR.fullmatch(operand.token.value.upper())
                if kind in cr_types and m and int(m.group(1)) not in bound:
                    where = operand.location()
                    raise ConfigurationError(
                        f"{program}, rendered line {where.line} "
                        f"({lines[where.line - 1].strip()!r}): reads CR{m.group(1)}, which "
                        f"[abi] does not bind (bound: {', '.join(f'CR{i}' for i in sorted(bound))})"
                    )
