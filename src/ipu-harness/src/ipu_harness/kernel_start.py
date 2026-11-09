"""What a kernel starts with: the contract the harness holds every program to.

A kernel is launched, not reset into. At every kernel start (a single-program
run, each stage of a multi-stage kernel, each step of a chain) the harness
puts the registers in the state the hardware leaves unspecified, so a program
that reads a register it never wrote fails here, not on the chip:

* ``R_MASK`` is all ones (its reset value, every lane on in all eight slots),
  unless the kernel declares ``[kernel] keeps_mask = true``: it then keeps what
  the kernel before it left (the first kernel on a model has none, so it gets
  garbage).
* The loop registers, the vector registers, the accumulator and the other data
  registers (:data:`GARBAGE`) hold seeded random bytes: every bit pattern,
  NaNs included. The seed names the kernel, the case and the start, so a
  failure reproduces and two starts differ.
* The CRs are the host's: ``[abi]`` and the default DS (:mod:`ipu_harness.abi`).
  Unbound CRs are not touched (``check_cr_reads`` refuses a program reading one).

XMEM, the fault state, the arithmetic flags and the PC are not registers of the
kernel and stay as they are.
"""

from __future__ import annotations

import numpy as np

from ipu_emu.reference import ReferenceModel
from ipu_harness.cases import rng
from ipu_isa.registers import REGISTER_DEFINITIONS

#: Banks that hold seeded garbage at kernel start, in the order they are written.
GARBAGE: tuple[str, ...] = (
    "lr",
    "r",
    "r_wide_debug",
    "r_cyclic",
    "r_cyclic_wide_debug",
    "r_meta",
    "r_cyclic_meta",
    "r_acc",
    "mult_res",
    "mem_bypass",
    "aaq_lut",
)

#: Banks set to all ones at kernel start (unless the kernel keeps them).
RESET_TO_ONES: tuple[str, ...] = ("r_mask",)

#: Banks the host writes at launch (``[abi]`` and the default DS).
HOST: tuple[str, ...] = ("cr",)


def start_rng(kernel: str, case: str, *, step: int = 0, stage: int = 0) -> np.random.Generator:
    """The garbage stream of one kernel start.

    Args:
        kernel: The kernel's name.
        case: The case's name.
        step: The chain step (0 outside a chain).
        stage: The stage of a multi-stage kernel (0 for a single program).

    Returns:
        A generator seeded from CRC-32 of the four, through
        :func:`ipu_harness.cases.rng`: the same in every process, different for
        every start.
    """
    return rng(kernel, case, f"kernel start {step}.{stage}")


def start_kernel(
    model: ReferenceModel, rng: np.random.Generator, *, keep_mask: bool, fresh: bool
) -> None:
    """Put ``model``'s registers in the state a kernel starts with.

    Args:
        model: The model, halted, its program loaded.
        rng: The garbage's source; the same generator state gives the same
            registers.
        keep_mask: The kernel keeps ``R_MASK`` from the kernel before it.
        fresh: There is no kernel before it (a run on a new model, the first
            step of a chain): a kept ``R_MASK`` is garbage.
    """
    for bank in GARBAGE:
        definition = REGISTER_DEFINITIONS[bank]
        size = definition["size_bytes"]
        for index in range(definition["count"]):
            if definition["vector"]:
                model.write_register(bank, rng.bytes(size), index)
            else:
                model.write_register(bank, int(rng.integers(0, 1 << (8 * size))), index)
    for bank in RESET_TO_ONES:
        definition = REGISTER_DEFINITIONS[bank]
        size = definition["size_bytes"]
        for index in range(definition["count"]):
            if not keep_mask:
                model.write_register(bank, b"\xff" * size, index)
            elif fresh:
                model.write_register(bank, rng.bytes(size), index)
