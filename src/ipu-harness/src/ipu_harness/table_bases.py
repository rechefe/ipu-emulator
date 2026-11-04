"""One CR names one table: the rule a kernel's table CRs must keep.

A CR that an XMEM instruction uses as its ``base`` (``LD.MULT``,
``LD.MULT.CYCLIC``, ``LD.MASK``) or ``table_id`` (``LD.LUT``,
``ACTIVATE.QUANTIZE``, ``ACTIVATE.EXACT``) holds a bare table ID
(:mod:`ipu_harness.table_ids`), never a row: the row is the table's storage
plus the offset LR. So a kernel reaches each tensor through exactly one CR,
any offset into it is in an LR, and the program does not depend on the ID's
value. :func:`check_table_bases` holds a kernel to that:

* every base CR is bound by ``[abi]`` to a tensor (``CR0`` only when a tensor
  is bound to it: table 0);
* a CR bound to a free table is not read as a number (``SET LR1 CR5``,
  ``ADD ... CR5``): its ID is the binder's to choose. The pinned ``CR0`` reads
  0 and may be read.

(That ``[abi]`` gives no tensor an ``offset``, and binds no ``[derived]`` value
over ``base(<tensor>)`` to a CR, is the loader's check.)
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

_CR = re.compile(r"CR([0-9]+)(?:\.[LH])?")


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
    """Fail a program that reaches a tensor other than through its table CR.

    Args:
        manifest: The kernel.
        rendered: Its program with the template rendered
            (``AssemblyResult.rendered``); it is parsed, not rendered again.
        abi: The names bound in place of its ``[abi]`` (a stage's).
        name: The program's file name for messages; its ``.asm`` by default.

    Raises:
        ConfigurationError: An XMEM instruction whose base CR is bound to no
            tensor, or a free table's CR read as a number. The message names
            the line.
    """
    program = manifest.asm.name if name is None else name
    bindings = manifest.abi if abi is None else abi
    tensors = _bound_tensors(manifest, bindings)
    operands = _base_operands(isa())
    lines = rendered.splitlines()
    zero = int(ZERO_REGISTER[2:])
    for word in AssemblySession("", name=program).parse_rendered(rendered):
        for instruction in word["instructions"]:
            mnemonic = instruction["opcode"].token.value.lower()
            base = operands.get(mnemonic)
            for index, operand in enumerate(instruction["operands"]):
                match = _CR.fullmatch(operand.token.value.upper())
                if match is None:
                    continue
                number = int(match.group(1))
                where = operand.location()
                line = f"{program}, rendered line {where.line} ({lines[where.line - 1].strip()!r})"
                if index == base and number not in tensors:
                    raise ConfigurationError(
                        f"{line}: {mnemonic.upper()} addresses a table through {match.group(0)}, "
                        "which [abi] binds to no tensor"
                        + (" (CR0 is table 0: bind a tensor to it)" if number == zero else "")
                    )
                if index != base and number in tensors and number != zero:
                    raise ConfigurationError(
                        f"{line}: {mnemonic.upper()} reads {match.group(0)}, which holds a "
                        "table ID that the harness chooses: a program may not depend on its value"
                    )
