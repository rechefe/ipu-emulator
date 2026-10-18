"""The ``[abi]`` table of a ``kernel.toml``: which register each name binds,
and when two bindings may share one (for :mod:`ipu_harness.manifest`)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ipu_emu.errors import ConfigurationError
from ipu_harness._manifest_decls import BINDING_KEYS, KernelManifest, ParamDecl, TensorDecl
from ipu_harness._manifest_params import read_condition
from ipu_harness._toml_source import Reader, where
from ipu_harness.conditions import Condition
from ipu_isa.registers import REGISTER_DEFINITIONS


def _binding_problem(
    name: str,
    register: str,
    params: Mapping[str, ParamDecl],
    used: Mapping[str, str],
    *,
    label: str,
    table: str,
) -> str | None:
    """What is wrong with binding ``name`` to ``register`` (already upper case),
    or ``None``; ``label`` names the key in a message, ``table`` the table.
    The caller has checked that ``name`` is the kernel's."""
    m = re.fullmatch(r"(CR|LR)(0|[1-9][0-9]?)", register, re.ASCII)
    banks = {"CR": REGISTER_DEFINITIONS["cr"]["count"], "LR": REGISTER_DEFINITIONS["lr"]["count"]}
    if not m or int(m.group(2)) >= banks[m.group(1)]:
        return f"{label} = {register!r} is no CR or LR register"
    if m.group(1) == "CR" and int(m.group(2)) < 2:
        return f"{table} binds {register}, a hardwired constant"
    if register in used:
        return f"{table} {name} and {used[register]} both bind {register}"
    if name in params and params[name].type == "str":
        return f"{table} {name} is a str parameter; a register holds a number"
    if register.startswith("LR") and name in params and params[name].type == "float":
        return f"{table} {name} is an element scalar (a float); it binds a CR, not {register}"
    return None


#: The hardwired zero a tensor pinned at row 0 may be read through.
ZERO_REGISTER = "CR0"


def _register(
    r: Reader, name: str, raw: object, tensor: TensorDecl | None, params: Mapping[str, ParamDecl]
) -> str:
    """The register of one binding, in upper case: any but CR0 and CR1 by
    :func:`_binding_problem`'s rules, or CR0 for a tensor pinned at row 0.
    Sharing a register is :func:`read_abi`'s check (conditions may exclude)."""
    path = ("abi", name)
    text = str(r.typed(raw, str, path))
    register = text.upper()
    if register == ZERO_REGISTER and tensor is not None:
        if tensor.row != 0:
            raise r.fail(path, f"[abi] {name} is read through CR0 (always 0); pin it with row = 0")
        return register
    problem = _binding_problem(name, register, params, {}, label=where(path), table="[abi]")
    if problem is not None:
        # The kernel.toml spelling, not the upper-case one.
        raise r.fail(path, problem.replace(repr(register), repr(text), 1))
    return register


@dataclass(frozen=True)
class _Binding:
    register: str
    offset: int
    when: Condition | None


def _binding(
    r: Reader,
    name: str,
    raw: object,
    tensor: TensorDecl | None,
    params: Mapping[str, ParamDecl],
) -> _Binding:
    """A register, or ``{ register = ..., offset = n, when = ... }``."""
    path = ("abi", name)
    when = tensor.when if tensor is not None else None
    if not isinstance(raw, dict):
        return _Binding(_register(r, name, raw, tensor, params), 0, when)
    r.known(raw, BINDING_KEYS, path)
    if "register" not in raw:
        raise r.fail(path, f"{where(path)} needs 'register'")
    register = _register(r, name, raw["register"], tensor, params)
    if "offset" in raw and tensor is None:
        raise r.fail((*path, "offset"), f"[abi] {name}: only a tensor's binding takes an offset")
    offset = r.typed(raw.get("offset", 0), int, (*path, "offset"))
    if register == ZERO_REGISTER and offset:
        raise r.fail((*path, "offset"), f"[abi] {name} is read through CR0; it takes no offset")
    if "when" in raw:
        bools = [n for n, p in params.items() if p.type == "bool"]
        own = read_condition(r, raw["when"], bools, (*path, "when"))
        if tensor is not None and own != when:
            # An always-present tensor must always be bound, and a conditional
            # one exactly when it exists.
            raise r.fail(
                (*path, "when"),
                f"[abi] {name} takes its tensor's condition ({when or 'always'}), not {own}",
            )
        when = own
    return _Binding(register, offset, when)


def _same_pinned_value(
    tensors: Mapping[str, TensorDecl],
    offsets: Mapping[str, int],
    name: str,
    offset: int,
    other: str,
) -> bool:
    """Whether ``name`` (at ``offset``) and ``other`` are tensors pinned at literal
    rows whose bindings give their register the same value (row plus offset):
    one register then reaches both regions (vector_tour's relu, stored at
    ``acc`` plus three accumulator stores)."""
    mine, theirs = tensors.get(name), tensors.get(other)
    if mine is None or theirs is None:
        return False
    if not isinstance(mine.row, int) or not isinstance(theirs.row, int):
        return False
    return mine.row + offset == theirs.row + offsets.get(other, 0)


def _sharing_clash(
    name: str,
    binding: _Binding,
    others: list[str],
    conditions: Mapping[str, Condition],
    tensors: Mapping[str, TensorDecl],
    offsets: Mapping[str, int],
) -> str | None:
    """The first earlier binding of the same register that ``name`` may not
    share it with: unless their conditions exclude each other, or both are
    tensors pinned so the register holds one value (:func:`_same_pinned_value`)."""
    for other in others:
        if binding.when is not None and binding.when.excludes(conditions.get(other)):
            continue
        if not _same_pinned_value(tensors, offsets, name, binding.offset, other):
            return other
    return None


def read_abi(
    r: Reader,
    raw: object,
    params: Mapping[str, ParamDecl],
    names: set[str],
    tensors: Mapping[str, TensorDecl],
    placed: Mapping[str, frozenset[str]],
) -> tuple[Mapping[str, str], Mapping[str, int], Mapping[str, Condition]]:
    table = r.table(raw, ("abi",))
    abi: dict[str, str] = {}
    offsets: dict[str, int] = {}
    conditions: dict[str, Condition] = {}
    used: dict[str, list[str]] = {}
    for name, value in table.items():
        path = ("abi", name)
        if name not in names:
            raise r.fail(path, f"{where(path)} names no parameter, tensor or derived value")
        binding = _binding(r, name, value, tensors.get(name), params)
        register = binding.register
        if binding.offset:
            offsets[name] = binding.offset
        if binding.when is not None:
            conditions[name] = binding.when
        clash = _sharing_clash(name, binding, used.get(register, []), conditions, tensors, offsets)
        if clash is not None:
            raise r.fail(path, f"[abi] {name} and {clash} both bind {register}")
        used.setdefault(register, []).append(name)
        abi[name] = register

    # A bound placed value that reads base(x) carries x's base to the program
    # too (a region bound at a parameter-dependent offset, E07-15), when it is
    # bound in every case x exists: unconditionally, or under x's condition.
    def reaches(tensor: TensorDecl) -> bool:
        return any(
            tensor.name in reads and conditions.get(name) in (None, tensor.when)
            for name, reads in placed.items()
            if name in abi
        )

    for tensor in tensors.values():
        if tensor.name not in abi and not reaches(tensor):
            raise r.fail(
                (tensor.section, tensor.name),
                f"[abi] does not bind tensor {tensor.name!r}; its base row must reach the program",
            )
    return MappingProxyType(abi), MappingProxyType(offsets), MappingProxyType(conditions)


def check_placed_conditions(
    r: Reader,
    placed: Mapping[str, frozenset[str]],
    tensors: Mapping[str, TensorDecl],
    abi_when: Mapping[str, Condition],
    abi: Mapping[str, str],
) -> None:
    """A placed value's binding takes the condition of the conditional tensors
    it reads, as a tensor's binding takes its tensor's: it is bound exactly
    when they exist."""
    for name, read in placed.items():
        if name not in abi:
            continue
        conditions = {tensors[t].when for t in read if tensors[t].when is not None}
        if len(conditions) > 1:
            raise r.fail(("abi", name), f"[abi] {name} reads tensors under different conditions")
        if conditions and abi_when.get(name) not in conditions:
            want = next(iter(conditions))
            raise r.fail(
                ("abi", name),
                f"[abi] {name} reads the base of a tensor that exists when {want}; "
                f"bind it with when = {str(want)!r}",
            )


def check_abi(manifest: KernelManifest, abi: Mapping[str, str], *, where: str) -> Mapping[str, str]:
    """``abi`` checked by ``[abi]``'s rules, for a table that is not the
    kernel's own (a stage's, E06-14): each name is the kernel's parameter,
    tensor or derived value, each register a CR (not CR0 or CR1) or LR bound
    once, no ``str`` parameter, no float on an LR. Unlike ``[abi]``, it need
    not bind every tensor.

    Args:
        manifest: The kernel whose names ``abi`` binds.
        abi: Names to registers (``{"x": "CR2"}``; any case).
        where: What the table is, for the error (``"stage copy.asm"``).

    Returns:
        The table, registers in upper case.

    Raises:
        ConfigurationError: A binding breaks a rule.
    """
    names = {*manifest.params, *manifest.tensors, *manifest.derived}
    checked: dict[str, str] = {}
    for name, value in abi.items():
        if name not in names:
            raise ConfigurationError(
                f"{manifest.name}: {where} abi {name!r} names no parameter, tensor or derived value"
            )
        register = value.upper() if isinstance(value, str) else str(value)
        problem = _binding_problem(
            name,
            register,
            manifest.params,
            {v: k for k, v in checked.items()},
            label=f"{where} abi {name!r}",
            table=f"{where}'s abi",
        )
        if problem is not None:
            # The value as the stage wrote it, not the upper-case one.
            shown = problem.replace(repr(register), repr(value), 1)
            raise ConfigurationError(f"{manifest.name}: {shown}")
        checked[name] = register
    return MappingProxyType(checked)
