"""AAQ slot handlers: ACTIVATE.QUANTIZE (activation, then INT8 quantization)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.activations import apply_activation
from ipu_emu.errors import EmulatorError
from ipu_emu.geometry import agg_lane_count, lane_format
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import WideVectorArithmetic
from ipu_isa.activations import (
    ACTIVATION_EXP2,
    ACTIVATION_IDENTITY,
    ACTIVATION_RECIPROCAL,
)
from ipu_isa.constants import REGISTER_WORD_VALUE_MASK
from ipu_isa.geometry import LANES

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu
    from ipu_emu.ipu_state import IpuState
    from ipu_emu.numerics import LaneCodec

# Exactly the double used by ``apply_activation`` for EXP2, computed once.
_LOG_2 = math.log(2.0)


def execute_aaq_nop(ipu: Ipu) -> None:
    """Execute NOP in aaq slot: No operation."""


def _execute_prototype_wide_fp32_activation(
    codec: LaneCodec, fn_id: int, acc_buf: memoryview, post_buf: memoryview
) -> bool:
    """Batch the full-row activations used by FP32 softmax.

    This is deliberately prototype-only: the physical pipeline still
    activates each lane. The arithmetic is nevertheless the parent's
    arithmetic -- Python ``math.exp`` and double division -- with the same
    float32 store boundary. Computing the row first removes 128 pairs of
    ``struct`` calls and repeated activation dispatch. If scalar
    arithmetic raises, the caller replays the original lane loop so its
    exception and partial-write state remain authoritative.
    """
    if fn_id not in (
        ACTIVATION_IDENTITY,
        ACTIVATION_EXP2,
        ACTIVATION_RECIPROCAL,
    ):
        return False

    lanes = codec.load_acc(acc_buf)
    try:
        if fn_id == ACTIVATION_IDENTITY:
            activated = lanes
        elif fn_id == ACTIVATION_EXP2:
            activated = np.asarray(
                [math.exp(float(value) * _LOG_2) for value in lanes],
                dtype=np.float64,
            )
        else:
            values = []
            for value in lanes:
                scalar = float(value)
                values.append(1.0 / scalar if scalar != 0.0 else 0.0)
            activated = np.asarray(values, dtype=np.float64)
    except (OverflowError, ValueError, ZeroDivisionError):
        return False

    codec.store_acc(post_buf, activated)
    return True


def _activate(state: IpuState, fn_id: int, raw: float | int) -> float:
    """The selected activation of one lane, with the state's activation parameters."""
    return apply_activation(
        fn_id,
        float(raw),
        elu_alpha=state.elu_alpha,
        window_a=state.window_a,
        window_b=state.window_b,
    )


def _activate_wide(
    state: IpuState,
    codec: LaneCodec,
    fn_id: int,
    active: int,
    acc_buf: memoryview,
    post_buf: memoryview,
) -> None:
    """Wide-vector ACTIVATE.QUANTIZE: activate the active lanes, then quantize if asked."""
    fmt = lane_format(state)
    activated: list[float | int] = []
    try:
        for raw in codec.load_acc(acc_buf, active).tolist():
            y = _activate(state, fn_id, raw)
            if fmt == "<i":
                yi = int(round(y))
                if yi < -2147483648:
                    yi = -2147483648
                elif yi > 2147483647:
                    yi = 2147483647
                activated.append(yi)
            else:
                activated.append(float(y))
    finally:
        # The lanes activated before a failing one are written, as the
        # lane-by-lane store left them; a store failure wins, being earlier.
        codec.store_acc(post_buf, activated)

    if not state.wide_vector_quantize_output:
        return

    result = bytearray(128)
    post = codec.load_acc(post_buf, active).tolist()
    if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
        for i, val in enumerate(post):
            result[i] = max(-128, min(127, int(round(val)))) & 0xFF
    else:
        for i, val in enumerate(post):
            result[i] = max(-128, min(127, val)) & 0xFF
    state.regfile.set_post_aaq_reg(result + bytearray(384))


def execute_activate_quantize(ipu: Ipu, *, activation_fn: int, cr_idx: int) -> None:
    """Apply element-wise activation then quantize to INT8.

    Reads each active element from the live ``r_acc`` register file, applies
    the selected activation, clamps to the INT8 range ``[-128, 127]``, and stores
    the resulting bytes in the leading active-element positions of ``post_aaq_reg``;
    all remaining bytes are zeroed. ``r_acc`` is not modified.

    The active element count comes from ``cr_idx``'s decoded dstructure
    ``valid_elements`` field; the caller must name a CR register explicitly.

    Requires INT8 mode in normal operation. In wide-vector debug mode, activation
    is always applied; quantization only occurs when ``state.wide_vector_quantize_output``
    is set (enabling comparison with the real INT8 path).
    """
    state = ipu.state
    fn_id = int(activation_fn) & REGISTER_WORD_VALUE_MASK
    valid_elements = ipu._view.dstructure(cr_idx, "cr_idx").valid_elements
    active = agg_lane_count(valid_elements)
    fmt = lane_format(state)
    codec = ipu._codec
    acc_buf = ipu._view.read_forgetting_provenance("r_acc")
    post_buf = state.regfile.mutable("post_aaq_reg")

    if state.wide_vector_debug:
        if (
            ipu._prototyping
            and active == LANES
            and fmt == "<f"
            and not state.wide_vector_quantize_output
            and _execute_prototype_wide_fp32_activation(codec, fn_id, acc_buf, post_buf)
        ):
            return
        _activate_wide(state, codec, fn_id, active, acc_buf, post_buf)
        return

    if state.dtype != DType.INT8:
        raise EmulatorError("ACTIVATE.QUANTIZE instruction requires INT8 mode")

    result = bytearray(128)
    for i, raw in enumerate(codec.load_acc(acc_buf, active).tolist()):
        y = _activate(state, fn_id, raw)
        result[i] = max(-128, min(127, int(round(y)))) & 0xFF
    state.regfile.set_post_aaq_reg(result + bytearray(384))
