"""The activation operands: ids and their assembly names.

Two operand types (ADR 0007):

* ``AaqActivationFn`` -- the 3-bit selector of the hardware
  ``ACTIVATE.QUANTIZE``, in the ``ARCH-stage-aaq-str.md`` encoding:
  ``AAQ_ACTIVATION_FN_NAMES`` (ids 0-6; 7 is reserved).
* ``ActivationFn`` -- the 4-bit selector of the emulator-only
  ``ACTIVATE.EXACT``: ``ACTIVATION_FN_NAMES`` (ids 0-12, the
  ``MAS-stage-aaq-str.md`` section 5.0 list).

The activation math and its emulator-only defaults (alpha, window bounds) live
in the emulator package.
"""

from __future__ import annotations

ACTIVATION_IDENTITY = 0
ACTIVATION_RELU = 1
ACTIVATION_RELU6 = 2
ACTIVATION_SIGMOID = 3
ACTIVATION_TANH = 4
ACTIVATION_GELU = 5
ACTIVATION_SOFTPLUS = 6
ACTIVATION_ELU = 7
ACTIVATION_EXP2 = 8
ACTIVATION_RECIPROCAL = 9
ACTIVATION_RSQRT = 10
ACTIVATION_SILU = 11
ACTIVATION_WINDOW = 12

ACTIVATION_COUNT = 13

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
    "reciprocal",
    "rsqrt",
    "silu",
    "window",
)

# ACTIVATE.QUANTIZE (ARCH encoding, ADR 0007): id = index.
AAQ_IDENTITY = 0
AAQ_RELU = 1
AAQ_RELU6 = 2
AAQ_RECIPROCAL = 3
AAQ_RSQRT = 4
AAQ_EXP2 = 5
AAQ_GENERIC = 6

AAQ_ACTIVATION_FN_NAMES: tuple[str, ...] = (
    "identity",
    "relu",
    "relu6",
    "reciprocal",
    "rsqrt",
    "exp2",
    "generic",
)
