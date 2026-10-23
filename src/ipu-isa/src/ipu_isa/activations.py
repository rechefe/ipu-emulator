"""The ``ActivationFn`` operand: activation ids and their assembly names.

Encodings match ``docs/content/specs/MAS-stage-aaq-str.md`` section 5.0. Assembly
uses ``ACTIVATE … <name>`` where ``<name>`` is one of the strings in
``ACTIVATION_FN_NAMES`` (same order as ids **0**–**12**). The activation math
and its emulator-only defaults (α, window bounds) live in the emulator
package.
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
