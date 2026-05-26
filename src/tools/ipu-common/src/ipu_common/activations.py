"""Element-wise activation functions for IPU accumulator lanes.

Encodings match ``docs/content/specs/stage-aaq.md`` section 7.0. α for
``elu`` defaults to ``DEFAULT_ELU_ALPHA`` below; override per run via
:class:`ipu_emu.ipu_state.IpuState` constructor or
:meth:`IpuState.set_activation_alphas` (not CR-visible). Assembly uses
``ACTIVATE … <name>`` where ``<name>`` is one of the strings in
``ACTIVATION_FN_NAMES`` (same order as ids **0**–**8**); the emulator writes
activated lanes into ``POST_AAQ_REG``. See
``docs/content/building-applications.md#activations-emulator`` for calibration,
``STR_POST_AAQ_REG`` (store that register to XMEM), and pipeline notes.
"""

from __future__ import annotations

import math

ACTIVATION_IDENTITY = 0
ACTIVATION_RELU = 1
ACTIVATION_RELU6 = 2
ACTIVATION_SIGMOID = 3
ACTIVATION_TANH = 4
ACTIVATION_GELU = 5
ACTIVATION_SOFTPLUS = 6
ACTIVATION_ELU = 7
ACTIVATION_EXP2 = 8

ACTIVATION_COUNT = 9

# Assembly / encoding order (id = index); must match ACTIVATION_* constants above.
ACTIVATION_FN_NAMES: tuple[str, ...] = (
    "identity",
    "relu",
    "relu6",
    "sigmoid",
    "tanh",
    "gelu",
    "softplus",
    "elu",
    "exp2",
)

# Default α value — virtual configuration outside the ISA (issue #77).
# Mutable so tests can monkeypatch; ``IpuState`` normally snapshots this at init.
DEFAULT_ELU_ALPHA: float = 1.0

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
) -> float:
    """Apply activation ``fn_id`` (0–8) to scalar ``x``. Unknown ids → identity.

    If ``elu_alpha`` is omitted, the value comes from the module
    ``DEFAULT_ELU_ALPHA`` constant (snapshotted onto
    :class:`ipu_emu.ipu_state.IpuState` at construction for ``ACTIVATE``).
    Passing explicit α overrides those defaults for this call.
    """
    k = int(fn_id) & 0xFFFFFFFF
    if k >= ACTIVATION_COUNT:
        return x

    ea = _ELU_ALPHA if elu_alpha is None else float(elu_alpha)

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
    return x
