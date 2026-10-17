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
a ``float`` parameter  its element encoding in ``plan.format``, in the low
                       element-width bits (Q49; E06-03 checked it is exact)
a ``[derived]`` value  the integer (``-1``, ``"k - 128 * (ceil(k / 128) - 1) - 2"``)
=====================  =====================================================

Values are raw register bits: a negative integer is its two's complement in
the register's width. Unless ``[abi]`` binds ``CR15``, it holds the default
dstructure (``valid_elements = 128``), as ``MemoryApp`` set it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ipu_as.session import AssemblySession
from ipu_emu.errors import ConfigurationError

# The emulator's dstructure convention (CR15, CLAUDE.md); ipu_isa has no constant for it.
from ipu_emu.ipu_config import DEFAULT_DSTRUCTURE_CR
from ipu_emu.reference import ReferenceModel
from ipu_harness._codec import encode
from ipu_harness.expressions import ExpressionError, base_name, evaluate
from ipu_harness.manifest import ZERO_REGISTER, KernelManifest
from ipu_harness.params import CasePlan, PlanError
from ipu_harness.placement import Placement
from ipu_isa.constants import (
    CR_READ_ONLY_INITIAL_VALUES,
    LR_CR_SCALAR_VALUE_MASK,
    DStructureConfig,
    encode_dstructure,
)
from ipu_isa.formats import FORMATS
from ipu_isa.geometry import LANES
from ipu_isa.operand_types import OperandKind
from ipu_isa.view import Isa, isa

__all__ = ["RegisterWrite", "apply", "check_cr_reads", "register_writes", "template_names"]

_CR = re.compile(r"CR([0-9]+)")

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
    """``value`` encoded in ``plan.format``, as the low bits of a register."""
    return int.from_bytes(encode(np.array([value]), plan.format), "little")


def _placed_value(name: str, plan: CasePlan, placement: Placement) -> int:
    """A placed ``[derived]`` value: its expression over the integer parameters,
    the other derived values and each region's base (bytes in a ``unit =
    "bytes"`` kernel, as a tensor's binding)."""
    manifest = plan.manifest
    scale = FORMATS[plan.format].row_bytes if manifest.byte_units else 1
    names: dict[str, int] = {
        **{k: v for k, v in plan.params.items() if isinstance(v, int) and not isinstance(v, bool)},
        **plan.derived,
        **{base_name(r.name): r.base_row * scale for r in placement.regions.values()},
    }
    for derived, expression in manifest.derived.items():
        if derived in manifest.placed:
            try:
                names[derived] = evaluate(expression, names)
            except ExpressionError as error:
                raise PlanError(
                    manifest, f"derived.{derived}", f"{expression!r}: {error}"
                ) from None
        if derived == name:
            return names[name]
    raise KeyError(name)  # the caller passes a placed name


def _value(name: str, plan: CasePlan, placement: Placement) -> tuple[int, str]:
    """The bits ``name`` puts in its register, and what they are."""
    manifest = plan.manifest
    if name in manifest.tensors:
        base = placement.region(name).base_row
        offset = manifest.abi_offsets.get(name, 0)
        shown = f" {offset:+d}" if offset else ""
        if manifest.byte_units:
            bytes_base = base * FORMATS[plan.format].row_bytes
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
    dstructure: DStructureConfig | None = None,
) -> tuple[RegisterWrite, ...]:
    """Every register's starting value, in ``[abi]`` order, then the
    dstructure in ``CR15`` unless ``[abi]`` binds it. A ``CR0`` binding writes
    nothing.

    Args:
        plan: The checked case.
        placement: Its regions.
        abi: Names to registers in place of the kernel's ``[abi]`` (a stage's,
            E06-14), already checked by ``manifest.check_abi``.
        dstructure: ``CR15``'s dstructure; ``None``: the default (128 valid
            elements).

    Returns:
        The writes, values as raw register bits.

    Raises:
        PlanError: A value does not fit a 32-bit register.
        ConfigurationError: ``dstructure`` is given and ``abi`` binds ``CR15``.
    """
    bindings = plan.manifest.abi if abi is None else abi
    cr15 = f"CR{DEFAULT_DSTRUCTURE_CR}"
    if dstructure is not None and cr15 in bindings.values():
        raise ConfigurationError(
            f"{plan.manifest.name}: the abi binds {cr15} and a dstructure is given for it"
        )
    writes = []
    for name, register in bindings.items():
        value, source = _value(name, plan, placement)
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
        writes.append(RegisterWrite(register, value & LR_CR_SCALAR_VALUE_MASK, source))
    if cr15 not in bindings.values():
        if dstructure is None:
            bits = encode_dstructure(valid_elements=LANES)
            source = "dstructure (default, 128 valid elements)"
        else:
            bits = dstructure.to_register_value()
            source = f"dstructure ({dstructure.valid_elements} valid elements)"
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

    ``CR0`` and ``CR1`` are hardwired; ``CR15`` holds the default dstructure
    unless bound; every other CR a program reads must be in ``[abi]``.

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
    bound |= {*CR_READ_ONLY_INITIAL_VALUES, DEFAULT_DSTRUCTURE_CR}
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
