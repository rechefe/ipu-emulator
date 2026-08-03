# Residual add: C[r] = A[r] + B[r]  for r = 0..287
#
# A, B: interleaved channel-major [256 tokens, 144 channels] = 288 rows × 128 bytes
#   Row r at A_BASE + r*128  (resp. B_BASE + r*128)
# C: FP32 accumulator output, 288 rows × 512 bytes
#   Row r at OUTPUT_BASE + r*512
#
# There is no add instruction in the vector path, so each addend is passed
# through the multiplier against a constant 1.0 and summed in the accumulator.
# The scalar 1.0 comes from a CR (DTYPE_ONE), so no register holds ones.
#   Per row:
#     Cycle 1: LDR_CYCLIC_MULT_REG → r_cyclic = A[r]; MULT.RC.VE × 1.0; ACC.ADD.FIRST
#     Cycle 2: LDR_CYCLIC_MULT_REG → r_cyclic = B[r]; MULT.RC.VE × 1.0; ACC.ADD
#     Cycle 3: STR_ACC_REG → store A[r]+B[r]; advance row counter
#     Cycle 4: advance output ptr; BLT → loop
#   Total: 4 cycles × 288 rows = 1152 cycles.
#
# IMPORTANT — live-LR semantics:
#   ADD fires before XMEM; ptrs init at -128 so first live = 0 after ADD.
#   STR_ACC_REG reads out_ptr live; do NOT ADD out_ptr in the same cycle as it.
#   BLT reads snapshot; ADD row_index must happen the cycle before BLT.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.
#
# Register assignments and their meanings are listed in the register-name
# block below -- it is the single source of truth for this kernel.

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0   = "lr0"  %}  {# const 0: r_cyclic slot-0 base / mask_shift #}
{% set mask_off   = "lr1"  %}  {# const 0: mask_offset (kept separate for clarity) #}
{% set a_ptr      = "lr2"  %}  {# byte offset into A (startup -128) #}
{% set b_ptr      = "lr3"  %}  {# byte offset into B (startup -128) #}
{% set out_ptr    = "lr4"  %}  {# byte offset into C, += 512 per row #}
{% set row_index  = "lr5"  %}  {# row counter 0..287 #}
{% set row_limit  = "lr6"  %}  {# 288 = N_ROWS #}
{% set row_stride = "lr7"  %}  {# 128 = row stride for A and B #}
{% set out_stride = "lr8"  %}  {# 512 = output row stride #}

{% set A_BASE     = "cr0"  %}  {# base of A #}
{% set ONE        = "cr1"  %}  {# hardwired read-only 1 #}
{% set ONES_BASE  = "cr2"  %}  {# 128 bytes of dtype-1.0 #}
{% set OUT_BASE   = "cr3"  %}  {# output base #}
{% set ZERO       = "cr4"  %}  {# const 0 #}
{% set PTR_START  = "cr5"  %}  {# -128 startup init for the A/B row pointers #}
{% set ROW_COUNT  = "cr6"  %}  {# 288 = N_ROWS #}
{% set ROW_STRIDE = "cr7"  %}  {# 128 = A/B row stride #}
{% set OUT_STRIDE = "cr8"  %}  {# 512 = output row stride #}
{% set B_BASE     = "cr9"  %}  {# base of B #}
{% set DTYPE_ONE  = "cr10" %}  {# dtype-encoded 1.0 scalar for the pass-through MULT #}
{% set DSTRUCT    = "cr15" %}  {# reserved dstructure register #}

    SET                 {{ rc_slot0 }} {{ ZERO }};;
    SET                 {{ mask_off }} {{ ZERO }};;
    SET                 {{ a_ptr }} {{ PTR_START }};;
    SET                 {{ b_ptr }} {{ PTR_START }};;
    SET                 {{ out_ptr }} {{ ZERO }};;
    SET                 {{ row_index }} {{ ZERO }};;
    SET                 {{ row_limit }} {{ ROW_COUNT }};;
    SET                 {{ row_stride }} {{ ROW_STRIDE }};;
    SET                 {{ out_stride }} {{ OUT_STRIDE }};;
    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ ONES_BASE }};;  # r0 = ONES_BASE[0..127] = dtype-1.0 × 128

row_loop:
    # Cycle 1: r_acc = A[r] × 1.0  (live ADD a_ptr fires first → live a_ptr = r*128)
    #   MULT.RC.VE r_cyclic[0] × DTYPE_ONE(=dtype 1.0) → A[r] passed through.
    LDR_CYCLIC_MULT_REG {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }}; ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }}; MULT.RC.VE {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;;
    # Cycle 2: r_acc += B[r] × 1.0
    LDR_CYCLIC_MULT_REG {{ b_ptr }} {{ B_BASE }} {{ rc_slot0 }}; ADD {{ b_ptr }} {{ b_ptr }} {{ row_stride }}; MULT.RC.VE {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;;
    # Cycle 3: store (do NOT ADD out_ptr here: STR_ACC_REG reads out_ptr live)
    STR_ACC_REG         {{ out_ptr }} {{ OUT_BASE }}; ADD {{ row_index }} {{ row_index }} {{ ONE }};;
    # Cycle 4: advance output ptr; BLT reads snap row_index = already-incremented
    ADD                 {{ out_ptr }} {{ out_ptr }} {{ out_stride }}; BLT {{ row_index }} {{ row_limit }} row_loop;;

end:
    BKPT;;
