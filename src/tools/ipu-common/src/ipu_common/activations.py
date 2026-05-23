"""Element-wise activation functions for IPU accumulator lanes.

Encodings match ``docs/content/specs/stage-aaq.md`` section 7.0. α for
``leaky_relu``, ``elu``, and ``prelu`` defaults to ``DEFAULT_*`` constants below;
override per run via :class:`ipu_emu.ipu_state.IpuState` constructor or
:meth:`IpuState.set_activation_alphas` (not CR-visible). Assembly uses
``ACTIVATE … <name>`` where ``<name>`` is one of the strings in
``ACTIVATION_FN_NAMES`` (same order as ids **0**–**11**). See
``docs/content/building-applications.md#activations-emulator`` for calibration,
``POST_AAQ_REG`` (interim **512 B**, same width as ``R_ACC``), and
``STR_POST_AAQ_REG`` (store that register to XMEM).
"""

from __future__ import annotations

import math

ACTIVATION_IDENTITY = 0
ACTIVATION_RELU = 1
ACTIVATION_RELU6 = 2
ACTIVATION_LEAKY_RELU = 3
ACTIVATION_SIGMOID = 4
ACTIVATION_TANH = 5
ACTIVATION_GELU = 6
ACTIVATION_SILU = 7
ACTIVATION_SOFTPLUS = 8
ACTIVATION_ELU = 9
ACTIVATION_PRELU = 10
ACTIVATION_EXP2 = 11

ACTIVATION_COUNT = 12

# Assembly / encoding order (id = index); must match ACTIVATION_* constants above.
ACTIVATION_FN_NAMES: tuple[str, ...] = (
    "identity",
    "relu",
    "relu6",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "softplus",
    "elu",
    "prelu",
    "exp2",
)

# Default α values — virtual configuration outside the ISA (issue #77).
# Mutable so tests can monkeypatch; ``IpuState`` normally snapshots these at init.
DEFAULT_LEAKY_ALPHA: float = 0.01
DEFAULT_ELU_ALPHA: float = 1.0
DEFAULT_PRELU_ALPHA: float = 0.25

# Legacy private names (same objects) for older monkeypatch patterns.
_LEAKY_ALPHA = DEFAULT_LEAKY_ALPHA
_ELU_ALPHA = DEFAULT_ELU_ALPHA
_PRELU_ALPHA = DEFAULT_PRELU_ALPHA


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
    leaky_relu_alpha: float | None = None,
    elu_alpha: float | None = None,
    prelu_alpha: float | None = None,
) -> float:
    """Apply activation ``fn_id`` (0–11) to scalar ``x``. Unknown ids → identity.

    If an α keyword is omitted, the value comes from the module ``DEFAULT_*``
    constants (snapshotted onto :class:`ipu_emu.ipu_state.IpuState` at construction
    for ``ACTIVATE``). Passing explicit α overrides those defaults for this call.
    """
    k = int(fn_id) & 0xFFFFFFFF
    if k >= ACTIVATION_COUNT:
        return x

    la = _LEAKY_ALPHA if leaky_relu_alpha is None else float(leaky_relu_alpha)
    ea = _ELU_ALPHA if elu_alpha is None else float(elu_alpha)
    pa = _PRELU_ALPHA if prelu_alpha is None else float(prelu_alpha)

    if k == ACTIVATION_IDENTITY:
        return x
    if k == ACTIVATION_RELU:
        return x if x > 0.0 else 0.0
    if k == ACTIVATION_RELU6:
        return min(max(x, 0.0), 6.0)
    if k == ACTIVATION_LEAKY_RELU:
        return x if x >= 0.0 else la * x
    if k == ACTIVATION_SIGMOID:
        return _sigmoid(x)
    if k == ACTIVATION_TANH:
        return math.tanh(x)
    if k == ACTIVATION_GELU:
        return x * _norm_cdf(x)
    if k == ACTIVATION_SILU:
        return x * _sigmoid(x)
    if k == ACTIVATION_SOFTPLUS:
        return _softplus(x)
    if k == ACTIVATION_ELU:
        return x if x >= 0.0 else ea * (math.exp(x) - 1.0)
    if k == ACTIVATION_PRELU:
        return x if x >= 0.0 else pa * x
    if k == ACTIVATION_EXP2:
        return math.exp(x * math.log(2.0))
    return x
