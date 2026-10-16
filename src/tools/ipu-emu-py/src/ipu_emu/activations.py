"""Element-wise activation math the emulator applies in the AAQ stage.

The activation ids and names are the ISA's ``ActivationFn`` operand
(:mod:`ipu_isa.activations`). α for ``elu`` defaults to ``DEFAULT_ELU_ALPHA``
below, and the ``window`` bounds ``[a, b)`` default to ``DEFAULT_WINDOW_A`` /
``DEFAULT_WINDOW_B``; override per run via the
:class:`ipu_emu.ipu_state.IpuState` constructor or
:meth:`IpuState.set_activation_alphas` (not CR-visible). The emulator writes
activated elements into ``POST_AAQ_REG``. See
``docs/content/building-applications.md#activations-emulator`` for calibration,
``ST.AAQ`` (store that register to XMEM), and pipeline notes.
"""

from __future__ import annotations

import math

from ipu_isa.activations import (
    ACTIVATION_COUNT,
    ACTIVATION_ELU,
    ACTIVATION_EXP2,
    ACTIVATION_GELU,
    ACTIVATION_IDENTITY,
    ACTIVATION_RECIPROCAL,
    ACTIVATION_RELU,
    ACTIVATION_RELU6,
    ACTIVATION_RSQRT,
    ACTIVATION_SIGMOID,
    ACTIVATION_SILU,
    ACTIVATION_SOFTPLUS,
    ACTIVATION_TANH,
    ACTIVATION_WINDOW,
)

# Default α value — virtual configuration outside the ISA (issue #77).
# Mutable so tests can monkeypatch; ``IpuState`` normally snapshots this at init.
DEFAULT_ELU_ALPHA: float = 1.0

# Default window bounds [a, b) for ``window`` — virtual configuration outside the
# ISA, same mechanism as α. Provisional values pending calibration.
DEFAULT_WINDOW_A: float = 0.0
DEFAULT_WINDOW_B: float = 0.1

# Legacy private name (same object) for older monkeypatch patterns.
_ELU_ALPHA = DEFAULT_ELU_ALPHA


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _softplus(x: float) -> float:
    # log(1+exp(x)); stable for large |x|
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def apply_activation(
    fn_id: int,
    x: float,
    *,
    elu_alpha: float | None = None,
    window_a: float | None = None,
    window_b: float | None = None,
) -> float:
    """Apply activation ``fn_id`` (0–12) to scalar ``x``. Unknown ids → identity.

    If ``elu_alpha`` / ``window_a`` / ``window_b`` are omitted, the values come
    from the module ``DEFAULT_ELU_ALPHA`` / ``DEFAULT_WINDOW_A`` /
    ``DEFAULT_WINDOW_B`` constants (snapshotted onto
    :class:`ipu_emu.ipu_state.IpuState` at construction for ``ACTIVATE``).
    Passing explicit values overrides those defaults for this call.
    """
    k = int(fn_id) & 0xFFFFFFFF
    if k >= ACTIVATION_COUNT:
        return x

    ea = _ELU_ALPHA if elu_alpha is None else float(elu_alpha)
    wa = DEFAULT_WINDOW_A if window_a is None else float(window_a)
    wb = DEFAULT_WINDOW_B if window_b is None else float(window_b)

    if k == ACTIVATION_IDENTITY:
        return x
    if k == ACTIVATION_RELU:
        return x if x > 0.0 else 0.0
    if k == ACTIVATION_RELU6:
        return min(max(x, 0.0), 6.0)
    if k == ACTIVATION_SIGMOID:
        return _sigmoid(x)
    if k == ACTIVATION_TANH:
        return math.tanh(x)
    if k == ACTIVATION_GELU:
        return x * _norm_cdf(x)
    if k == ACTIVATION_SOFTPLUS:
        return _softplus(x)
    if k == ACTIVATION_ELU:
        return x if x >= 0.0 else ea * (math.exp(x) - 1.0)
    if k == ACTIVATION_EXP2:
        return math.exp(x * math.log(2.0))
    if k == ACTIVATION_RECIPROCAL:
        return 1.0 / x if x != 0.0 else 0.0
    if k == ACTIVATION_RSQRT:
        return 1.0 / math.sqrt(x) if x > 0.0 else 0.0
    if k == ACTIVATION_SILU:
        return x * _sigmoid(x)
    if k == ACTIVATION_WINDOW:
        # Rectangular window: 1 inside the half-open range [a, b), 0 elsewhere.
        return 1.0 if wa <= x < wb else 0.0
    return x
