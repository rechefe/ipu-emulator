# Packed residual add (L5 packed-layout viability experiment):
#   C[ch] = A[ch] + B[ch]  for ch = 0..239, PACKED 8 channels/row.
#
# Layer:   L5
# Scope:   single-stream
# Layout:  packed
# Shape:   240ch x 16tok, packing factor 8
# Status:  validated
# Related: asm_packed_residual_add_192x64.asm (L4 counterpart, same algorithm shape)
# Tests:   test_packed_residual_add_240x16.py
#
# A, B, C: 240 channels x 16 tokens, PACKED 8 channels per 128-lane row
#   (channel p's 16 tokens at lanes [16p, 16p+16)), 30 rows total
#   (240 / 8 = 30, exact). Row r holds channels [8r, 8r+8).
#
# No cross-partition combine needed: elementwise add is partition-local,
# each of the 8 packed channels in a row is independent of its neighbours.
# valid_elements=128 (the WHOLE row, all 8 packed channels) throughout --
# this is the "B: easy case" of the packed-viability task (see docs and the
# session report); contrast with the packed LINEAR kernel (C), which does
# need the cross-partition combine (primitive A) because contraction mixes
# packed input channels together.
#
# Same MULT-snapshot-contract / 4-cycles-per-row structure as
# residual_add_16x240.asm (unpacked baseline) -- see that file's header for
# the full pipelining rationale, unchanged here. The only difference is the
# loop trip count: 30 packed rows instead of 240 unpacked rows.
#
# CROSS-KERNEL R_MASK HAZARD: this kernel never calls LDR_MULT_MASK_REG --
# it relies entirely on R_MASK's regfile-init all-ones default, which is
# correct standalone but breaks if a prior kernel in the same IpuState left
# R_MASK non-default (e.g. asm_packed_output_linear_generic.asm's one-hot
# scatter mask). Encountered directly when chaining out-proj into residual
# add for the full-L5-layer test: partitions 1-7 read back as zero.
# Callers must explicitly reload an all-ones R_MASK between such kernels.
# See docs/isa_friction_log.md's "Cross-kernel R_MASK state bleed" entry.

{% set rc_slot0   = "lr0"  %}  {# const 0: r_cyclic slot-0 base / mask_shift #}
{% set mask_off   = "lr1"  %}  {# const 0: mask_offset (kept separate for clarity) #}
{% set a_ptr      = "lr2"  %}  {# row offset into A (startup -1) #}
{% set b_ptr      = "lr3"  %}  {# row offset into B (startup -1) #}
{% set out_ptr    = "lr4"  %}  {# row offset into C, += 1 per row #}
{% set row_index  = "lr5"  %}  {# row counter 0..29 #}
{% set row_limit  = "lr6"  %}  {# 30 = N_PACKED_ROWS #}
{% set row_stride = "lr7"  %}  {# 1 = row stride for A and B #}
{% set out_stride = "lr8"  %}  {# 1 = output row stride #}

{% set A_BASE     = "cr0"  %}  {# base of A #}
{% set ONE        = "cr1"  %}  {# hardwired read-only 1 #}
{% set OUT_BASE   = "cr3"  %}  {# output base #}
{% set ZERO       = "cr4"  %}  {# const 0 #}
{% set PTR_START  = "cr5"  %}  {# -1 row startup init for the A/B row pointers #}
{% set ROW_COUNT  = "cr6"  %}  {# 30 = N_PACKED_ROWS #}
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
