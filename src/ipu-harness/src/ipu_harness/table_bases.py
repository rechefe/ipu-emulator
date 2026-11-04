"""One CR names one table: the rule a kernel's base registers must keep.

A CR that an XMEM instruction uses as its ``base`` (``LD.MULT``,
``LD.MULT.CYCLIC``, ``LD.MASK``) or ``table_id`` (``LD.LUT``,
``ACTIVATE.QUANTIZE``, ``ACTIVATE.EXACT``) will hold a bare table ID, never a
row: the row is the table's storage plus the offset LR. So a kernel reaches
each tensor through exactly one CR, and any offset into it is in an LR.
:func:`check_table_bases` holds a kernel to that:

* every base CR is bound by ``[abi]`` to a tensor (``CR0`` only when the
  kernel pins a tensor at row 0 and binds it to ``CR0``);
* ``[abi]`` gives no tensor an ``offset``, and binds no ``[derived]`` value
  over ``base(<tensor>)`` (``"base(a) + 1"``) to a CR: those would be a
  second CR into the tensor, or a CR holding more than its base.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from ipu_as.session import AssemblySession
from ipu_emu.errors import ConfigurationError
from ipu_emu.xmem_access import ADDRESS_OPERANDS
from ipu_harness.manifest import ZERO_REGISTER, KernelManifest
from ipu_isa.view import Isa, isa

__all__ = ["check_table_bases"]

_CR = re.compile(r"CR([0-9]+)")


def _base_operands(view: Isa) -> dict[str, int]:
    """For each XMEM mnemonic (lower case), the index of its base-CR operand."""
    wanted = {base for base, _ in ADDRESS_OPERANDS} | {table for table, _ in ADDRESS_OPERANDS}
    out: dict[str, int] = {}
    for slot in view.spec_slot_order:
        for instruction in view.instructions(slot):
            for index, operand in enumerate(instruction.operands):
                if operand.name in wanted:
                    out.setdefault(instruction.mnemonic.lower(), index)
    return out


def _bound_tensors(manifest: KernelManifest, bindings: Mapping[str, str]) -> set[int]:
    """The CRs bound to a tensor."""
    out: set[int] = set()
    for name, register in bindings.items():
        match = _CR.fullmatch(register.upper())
        if match and name in manifest.tensors:
            out.add(int(match.group(1)))
    return out


def check_table_bases(
    manifest: KernelManifest,
    rendered: str,
    *,
    abi: Mapping[str, str] | None = None,
    name: str | None = None,
) -> None:
    """Fail a kernel that reaches a tensor through more than its one CR.

    Args:
        manifest: The kernel.
        rendered: Its program with the template rendered
            (``AssemblyResult.rendered``); it is parsed, not rendered again.
        abi: The names bound in place of its ``[abi]`` (a stage's). A stage
            has no ``[abi]`` offsets of its own: ``None`` checks the kernel's.
        name: The program's file name for messages; its ``.asm`` by default.

    Raises:
        ConfigurationError: An ``[abi]`` offset or placed ``base(...)`` value
            bound to a CR, or an XMEM instruction whose base CR is bound to no tensor. The message
            names the tensor or the line.
    """
    program = manifest.asm.name if name is None else name
    bindings = manifest.abi if abi is None else abi
    offsets = manifest.abi_offsets if abi is None else {}
    for tensor, offset in offsets.items():
        if offset and tensor in bindings:
            raise ConfigurationError(
                f"{manifest.name}: [abi] binds {tensor} at offset {offset}: a CR names a "
                "table, so its offset belongs in an LR (SET it from a CR holding the offset)"
            )
    for value, bases in manifest.placed.items():
        if bases and bindings.get(value, "").upper().startswith("CR"):
            reads = ", ".join(sorted(bases))
            raise ConfigurationError(
                f"{manifest.name}: [abi] binds {value} to {bindings[value]}, a value over "
                f"base({reads}): bind {reads} itself and keep the offset in an LR"
            )
    tensors = _bound_tensors(manifest, bindings)
    operands = _base_operands(isa())
    lines = rendered.splitlines()
    zero = int(ZERO_REGISTER[2:])
    for word in AssemblySession("", name=program).parse_rendered(rendered):
        for instruction in word["instructions"]:
            mnemonic = instruction["opcode"].token.value.lower()
            if mnemonic not in operands:
                continue
            operand = instruction["operands"][operands[mnemonic]]
            match = _CR.fullmatch(operand.token.value.upper())
            if match is None or int(match.group(1)) in tensors:
                continue
            where = operand.location()
            raise ConfigurationError(
                f"{program}, rendered line {where.line} ({lines[where.line - 1].strip()!r}): "
                f"{mnemonic.upper()} addresses a table through {match.group(0)}, which [abi] "
                "binds to no tensor"
                + (" (CR0 needs a tensor pinned at row 0)" if int(match.group(1)) == zero else "")
            )
