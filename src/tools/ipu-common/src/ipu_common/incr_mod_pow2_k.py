"""incr_mod_pow2 k operand — shared by assembler and emulator.

The LR slot encodes k in a narrow field: semantic k ∈ [LR_MOD_POW2_K_MIN, LR_MOD_POW2_K_MAX]
is stored as (k - 1) in LR_MOD_POW2_K_FIELD_BITS bits (4 bits → nine distinct codes 0..8).
"""

LR_MOD_POW2_K_MIN = 1
LR_MOD_POW2_K_MAX = 9
LR_MOD_POW2_K_FIELD_BITS = 4

# Valid encoded values are 0 .. (LR_MOD_POW2_K_MAX - LR_MOD_POW2_K_MIN)
LR_MOD_POW2_K_ENCODED_MAX = LR_MOD_POW2_K_MAX - LR_MOD_POW2_K_MIN
