# Packed residual add, L4 shape:
#   C[ch] = A[ch] + B[ch]  for ch = 0..191, PACKED 2 channels/row.
#
# Layer:   L4
# Scope:   single-stream
# Layout:  packed
# Shape:   192ch x 64tok, packing factor 2
# Status:  validated
# Related: asm_packed_residual_add_240x16.asm (L5 counterpart, same algorithm shape)
# Tests:   test_packed_residual_add_192x64.py
#
# A, B, C: 192 channels x 64 tokens, PACKED 2 channels per 128-lane row
#   (channel p's 64 tokens at lanes [64p, 64p+64)), 96 rows total
#   (192 / 2 = 96, exact). Row r holds channels [2r, 2r+2).
#
# L4 port of asm_packed_residual_add_240x16.asm: no width-dependent
# constants beyond the packed row count (96 vs L5's 30) -- elementwise add
# is partition-local regardless of partition width, so the construction is
# unchanged except the loop trip count. Same MULT-snapshot-contract /
# 4-cycles-per-row structure as residual_add_64x192.asm (unpacked
#
# CROSS-KERNEL R_MASK HAZARD: this kernel never calls LDR_MULT_MASK_REG --
# it relies entirely on R_MASK's regfile-init all-ones default, which is
# correct standalone but breaks if a prior kernel in the same IpuState left
# R_MASK non-default (e.g. asm_packed_output_linear_generic_p4.asm's
# one-hot scatter mask). Callers chaining this kernel after a packed-output
# linear kernel must explicitly reload an all-ones R_MASK first. See
# docs/isa_friction_log.md's "Cross-kernel R_MASK state bleed" entry.
# baseline).

{% set rc_slot0   = "lr0"  %}  {# const 0: r_cyclic slot-0 base / mask_shift #}
{% set mask_off   = "lr1"  %}  {# const 0: mask_offset (kept separate for clarity) #}
{% set a_ptr      = "lr2"  %}  {# row offset into A (startup -1) #}
{% set b_ptr      = "lr3"  %}  {# row offset into B (startup -1) #}
{% set out_ptr    = "lr4"  %}  {# row offset into C, += 1 per row #}
{% set row_index  = "lr5"  %}  {# row counter 0..95 #}
{% set row_limit  = "lr6"  %}  {# 96 = N_PACKED_ROWS #}
{% set row_stride = "lr7"  %}  {# 1 = row stride for A and B #}
{% set out_stride = "lr8"  %}  {# 1 = output row stride #}

{% set A_BASE     = "cr0"  %}  {# base of A #}
{% set ONE        = "cr1"  %}  {# hardwired read-only 1 #}
{% set OUT_BASE   = "cr3"  %}  {# output base #}
{% set ZERO       = "cr4"  %}  {# const 0 #}
{% set PTR_START  = "cr5"  %}  {# -1 row startup init for the A/B row pointers #}
{% set ROW_COUNT  = "cr6"  %}  {# 96 = N_PACKED_ROWS #}
{% set ROW_STRIDE = "cr7"  %}  {# 1 = A/B row stride #}
{% set OUT_STRIDE = "cr8"  %}  {# 1 = output row stride #}
{% set B_BASE     = "cr9"  %}  {# base of B #}
{% set DTYPE_ONE  = "cr10" %}  {# dtype-encoded 1.0 scalar for the pass-through MULT #}
{% set DSTRUCT    = "cr15" %}  {# reserved dstructure register: valid_elements=128 (whole packed row) #}

    SET                 {{ rc_slot0 }} {{ ZERO }};;
    SET                 {{ mask_off }} {{ ZERO }};;
    SET                 {{ a_ptr }} {{ PTR_START }};;
    SET                 {{ b_ptr }} {{ PTR_START }};;
    SET                 {{ out_ptr }} {{ ZERO }};;
    SET                 {{ row_index }} {{ ZERO }};;
    SET                 {{ row_limit }} {{ ROW_COUNT }};;
    SET                 {{ row_stride }} {{ ROW_STRIDE }};;
    SET                 {{ out_stride }} {{ OUT_STRIDE }};;
    LDR_CYCLIC_MULT_REG {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }};
    ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }};;

row_loop:
    LDR_CYCLIC_MULT_REG {{ b_ptr }} {{ B_BASE }} {{ rc_slot0 }};
    ADD {{ b_ptr }} {{ b_ptr }} {{ row_stride }};
    MULT.RC.VE {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;;
    MULT.RC.VE          {{ rc_slot0 }} {{ DTYPE_ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;;
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_BASE }};
    ADD {{ row_index }} {{ row_index }} {{ ONE }};
    LDR_CYCLIC_MULT_REG {{ a_ptr }} {{ A_BASE }} {{ rc_slot0 }};
    ADD {{ a_ptr }} {{ a_ptr }} {{ row_stride }};;
    ADD                 {{ out_ptr }} {{ out_ptr }} {{ out_stride }};
    BLT {{ row_index }} {{ row_limit }} row_loop;;

end:
    BKPT;;
