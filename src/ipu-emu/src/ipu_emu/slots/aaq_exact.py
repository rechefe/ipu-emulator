"""AAQ_EXACT slot handlers: the emulator-only ``ACTIVATE.EXACT`` (ADR 0007).

``ACTIVATE.EXACT`` applies one of the thirteen exact Python activations
(``ipu_emu.activations.apply_activation``, ``rsqrt(x <= 0) = 0``) to the active
elements of the forwarded R_ACC (the DS half) and stores the row as 128 FP32
values (512 bytes) at ``CR[table_id] + LR[table_offset]``; inactive lanes are
written 0. It is the FP32 mode's store: a program that uses it is refused when
it is loaded into an NX8-mode state (``ipu_emu.emulator.load_program``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.activations import apply_activation
from ipu_emu.errors import EmulatorError
from ipu_emu.geometry import fp32_mode, xmem_row_addr, xmem_row_check
from ipu_isa.activations import ACTIVATION_EXP2, ACTIVATION_IDENTITY, ACTIVATION_RECIPROCAL
from ipu_isa.faults import FaultId
from ipu_isa.geometry import LANES

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu
    from ipu_emu.ipu_state import IpuState

Fault = tuple[FaultId, str]

# Exactly the double used by ``apply_activation`` for EXP2, computed once.
_LOG_2 = math.log(2.0)


def execute_aaq_exact_nop(ipu: Ipu) -> None:
    """Execute NOP in the aaq_exact slot: No operation."""


def check_activate_exact(
    ipu: Ipu, *, activation_fn: int, ds: int, table_id: int, table_offset: int
) -> Fault | None:
    """The DS must be valid; F-6 for a row outside XMEM."""
    ipu.view.ds(ds)
    return xmem_row_check(ipu.state, table_id + table_offset)


def _activate(state: IpuState, fn_id: int, raw: float) -> float:
    """The selected activation of one lane, with the state's activation parameters."""
    params = state.activation
    return apply_activation(
        fn_id,
        float(raw),
        elu_alpha=params.elu_alpha,
        window_a=params.window_a,
        window_b=params.window_b,
    )


def _prototype_full_row(fn_id: int, lanes: np.ndarray) -> list[float] | None:
    """The activated row of the functions FP32 softmax uses, batched (prototyping).

    The physical pipeline activates each lane; the arithmetic here is the same
    (Python ``math.exp`` and double division), so the stored row is
    bit-identical. ``None`` for any other function, or when scalar arithmetic
    raises: the caller then runs the lane loop, whose exception is authoritative.
    """
    try:
        if fn_id == ACTIVATION_IDENTITY:
            return lanes.tolist()
        if fn_id == ACTIVATION_EXP2:
            return [math.exp(float(value) * _LOG_2) for value in lanes]
        if fn_id == ACTIVATION_RECIPROCAL:
            return [1.0 / float(v) if float(v) != 0.0 else 0.0 for v in lanes]
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return None


def execute_activate_exact(
    ipu: Ipu, *, activation_fn: int, ds: int, table_id: int, table_offset: int
) -> None:
    """Execute ACTIVATE.EXACT: exact activation of the active lanes, stored as one FP32 row.

    Raises:
        EmulatorError: The state is in NX8 mode (a program loaded through
            ``load_program`` never gets here: it is refused at load).
    """
    state = ipu.state
    if not fp32_mode(state):
        raise EmulatorError("ACTIVATE.EXACT is valid in FP32 mode only (ADR 0007)")
    codec = ipu.codec
    lanes = codec.load_acc(ipu.view.read("r_acc"))
    active = ipu.view.ds(ds).active()
    values: list[float] | None = None
    if ipu.prototyping and all(active):
        values = _prototype_full_row(activation_fn, lanes)
    if values is None:
        values = [
            _activate(state, activation_fn, x) if on else 0.0
            for x, on in zip(lanes.tolist(), active, strict=True)
        ]
    row = bytearray(4 * LANES)
    codec.store_acc(row, np.asarray(values, dtype=np.float64))
    state.xmem.write_address(xmem_row_addr(state, table_id + table_offset), bytes(row))
