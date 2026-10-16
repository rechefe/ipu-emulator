"""Registers from ``[abi]``: region bases, integers, element scalars (E06-06).

``[abi]`` is the one register map. :func:`register_writes` gives the value
each bound register starts with; :func:`template_names` gives the program the
same names (``{{ cr.x }}`` renders ``CR2``), so the two cannot drift. After
assembly, :func:`check_cr_reads` fails a program that reads a CR nothing
binds. A kernel writes no register itself.

What a key binds decides the value:

=====================  =====================================================
a tensor               its base row (``Placement``); in a ``unit = "bytes"``
                       kernel (L16), ``base_row x row_bytes(format)``
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
from typing import Protocol

import numpy as np

from ipu_as.session import AssemblySession
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_config import DEFAULT_DSTRUCTURE_CR
from ipu_harness._codec import encode
from ipu_harness.manifest import KernelManifest
from ipu_harness.params import CasePlan
from ipu_harness.placement import Placement
from ipu_isa.constants import (
    CR_READ_ONLY_INITIAL_VALUES,
    LR_CR_SCALAR_VALUE_MASK,
    encode_dstructure,
)
from ipu_isa.formats import FORMATS
from ipu_isa.geometry import LANES
from ipu_isa.operand_types import OperandKind
from ipu_isa.view import Isa, isa

__all__ = ["RegisterWrite", "apply", "check_cr_reads", "register_writes", "template_names"]

_REGISTER = re.compile(r"(CR|LR)([0-9]+)")


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
        return self.register[:2].lower()

    @property
    def index(self) -> int:
        """The register's number in its bank."""
        return int(self.register[2:])


def _element_bits(value: float, plan: CasePlan) -> int:
    """``value`` encoded in ``plan.format``, as the low bits of a register."""
    return int.from_bytes(encode(np.array([value]), plan.format), "little")


def _value(name: str, plan: CasePlan, placement: Placement) -> tuple[int, str]:
    """The bits ``name`` puts in its register, and what they are."""
    manifest = plan.manifest
    if name in manifest.tensors:
        base = placement.region(name).base_row
        if manifest.byte_units:
            return base * FORMATS[plan.format].row_bytes, f"{name} (region base, bytes)"
        return base, f"{name} (region base)"
    if name in manifest.derived:
        return plan.derived[name], f"{name} (derived)"
    decl = manifest.params[name]
    value = plan.params[name]
    if decl.type == "float":
        return _element_bits(float(value), plan), f"{name} (element scalar, {plan.format.value})"
    # The loader binds no str parameter: what is left is an int or a bool.
    return int(value), f"{name} ({decl.type} param)"


def register_writes(plan: CasePlan, placement: Placement) -> tuple[RegisterWrite, ...]:
    """Every register's starting value, in ``[abi]`` order, then the default
    dstructure in ``CR15`` unless ``[abi]`` binds it.

    Args:
        plan: The checked case.
        placement: Its regions.

    Returns:
        The writes, values as raw register bits.
    """
    writes = []
    for name, register in plan.manifest.abi.items():
        value, source = _value(name, plan, placement)
        writes.append(RegisterWrite(register, value & LR_CR_SCALAR_VALUE_MASK, source))
    dstructure = f"CR{DEFAULT_DSTRUCTURE_CR}"
    if dstructure not in plan.manifest.abi.values():
        bits = encode_dstructure(valid_elements=LANES)
        writes.append(RegisterWrite(dstructure, bits, "dstructure (default, 128 valid elements)"))
    return tuple(writes)


def template_names(manifest: KernelManifest) -> Mapping[str, Mapping[str, str]]:
    """The program's register names: ``{"cr": {"x": "CR2", ...}, "lr": {...}}``
    (``{{ cr.x }}``, ``{{ lr.i }}``; E06-07)."""
    banks: dict[str, dict[str, str]] = {"cr": {}, "lr": {}}
    for name, register in manifest.abi.items():
        banks[register[:2].lower()][name] = register
    return MappingProxyType({bank: MappingProxyType(names) for bank, names in banks.items()})


class _Registers(Protocol):
    def set_scalar(self, name: str, index: int, value: int) -> None: ...


class _State(Protocol):
    @property
    def regfile(self) -> _Registers: ...


def apply(state: _State, writes: Iterable[RegisterWrite]) -> None:
    """Write ``writes`` through the reference model's register file, by bank name."""
    for write in writes:
        state.regfile.set_scalar(write.bank, write.index, write.value)


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


def check_cr_reads(manifest: KernelManifest, rendered: str) -> None:
    """Fail a rendered program that reads a CR neither hardwired nor bound.

    ``CR0`` and ``CR1`` are hardwired; ``CR15`` holds the default dstructure
    unless bound; every other CR a program reads must be in ``[abi]``.

    Args:
        manifest: The kernel.
        rendered: Its program, Jinja rendered (pseudo-instructions are expanded
            by the parse).

    Raises:
        ConfigurationError: A read of an unbound CR, naming its line.
    """
    bound = {int(r[2:]) for r in manifest.abi.values() if r.startswith("CR")}
    bound |= {*CR_READ_ONLY_INITIAL_VALUES, DEFAULT_DSTRUCTURE_CR}
    view = isa()
    types, cr_types = _operand_types(view), _cr_types(view)
    for word in AssemblySession(rendered, name=str(manifest.asm)).parse():
        for instruction in word["instructions"]:
            mnemonic = instruction["opcode"].token.value
            operand_types = types.get(mnemonic.lower(), [])
            for kind, operand in zip(operand_types, instruction["operands"], strict=False):
                m = _REGISTER.fullmatch(operand.token.value.upper())
                if kind in cr_types and m and m.group(1) == "CR" and int(m.group(2)) not in bound:
                    raise ConfigurationError(
                        f"{manifest.asm.name}: {operand.location()}: {mnemonic} reads CR{m.group(2)}, which [abi] "
                        f"does not bind (bound: {', '.join(f'CR{i}' for i in sorted(bound))})"
                    )
