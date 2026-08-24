# Primitive A -- cross-partition combine, in isolation.
#
# Layer:   L5
# Scope:   single-stream
# Layout:  n/a (register-state microbenchmark, not a full kernel)
# Shape:   16ch equivalent, packing factor 8
# Status:  validated
# Related: asm_primitive_a_combine2x64.asm (L4 counterpart)
# Tests:   test_primitive_a_combine8x16.py
#
# Input:  r_acc holds 8 partitions of 16 lanes (128 lanes total), pre-loaded
#         directly by the test harness (host-side r_acc write -- there is no
#         XMEM data for this microbenchmark, only the register state).
# Output: 16 FP32 values = elementwise sum across the 8 partitions, stored
#         to XMEM row OUT_BASE (first 16 lanes valid; teardown crops).
#
# Method (the only cross-partition move the ISA offers -- see AGG.SUM's
# collapse-to-scalar semantics and RESHAPE's R_ACC-only/8-lane-per-call
# limits, neither of which does a partition-segmented reduce):
#   1. ACTIVATE.QUANTIZE identity (valid_elements=128) + STR_POST_AAQ_REG
#      dumps r_acc's raw 128 FP32 lanes to XMEM (wide-vector mode: identity
#      + no wide_vector_quantize_output = byte-exact float copy, same path
#      residual_add_16x240 uses).
#   2. LDR_CYCLIC_MULT_REG reloads those 128 elements into R_CYCLIC (index=0,
#      the only wide-mode-legal slot).
#   3. Eight MULT.RC.VE x 1.0 calls, rc_idx = 16*p for p=0..7 (ELEMENT
#      offset, matches LDR_CYCLIC_MULT_REG's index unit -- issue #182/PR
#      #196), each landing partition p's 16 values in mult_res lanes 0..15
#      (lanes 16..127 read stale/wrapped R_CYCLIC data past the loaded
#      region, but valid_elements=16 on the final store crops them out).
#      ACC.ADD.FIRST on p=0, ACC.ADD on p=1..7 sums them into r_acc[0..15].
#      rc_idx must be an LR (live-read operand); each step SETs it from a CR
#      holding the constant 16*p (cheaper than ADD-ing 16 seven times would
#      be to reason about, and each SET/MULT pair still co-issues in one
#      cycle since SET's dest LR is read live by the same bundle's MULT).
#   4. ACTIVATE.QUANTIZE identity (valid_elements=16) + STR_POST_AAQ_REG
#      stores the 16-lane combined result.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing.

{% set rc_idx_reg  = "lr0"  %}  {# r_cyclic base ELEMENT offset, reset per partition #}
{% set mask_off    = "lr1"  %}  {# const 0: mask_offset #}
{% set scratch_ptr = "lr2"  %}  {# row offset for the scratch dump/reload row #}
{% set out_ptr     = "lr3"  %}  {# row offset for the combined-result row #}

{% set SCRATCH_BASE = "cr0"  %}  {# base row: r_acc dump / reload scratch #}
{% set ONE          = "cr1"  %}  {# hardwired read-only 1 #}
{% set OUT_BASE     = "cr2"  %}  {# base row: combined 16-lane result #}
{% set OFF16        = "cr3"  %}  {# 16  = rc_idx for partition 1 #}
{% set ZERO         = "cr4"  %}  {# const 0 = rc_idx for partition 0 #}
{% set OFF32        = "cr5"  %}  {# 32  = rc_idx for partition 2 #}
{% set OFF48        = "cr6"  %}  {# 48  = rc_idx for partition 3 #}
{% set OFF64        = "cr7"  %}  {# 64  = rc_idx for partition 4 #}
{% set OFF80        = "cr8"  %}  {# 80  = rc_idx for partition 5 #}
{% set OFF96        = "cr9"  %}  {# 96  = rc_idx for partition 6 #}
{% set DTYPE_ONE    = "cr10" %}  {# dtype-encoded 1.0 scalar for the pass-through MULT #}
{% set OFF112       = "cr11" %}  {# 112 = rc_idx for partition 7 #}
{% set DSTRUCT_WIDE = "cr14" %}  {# valid_elements=128, for the reload multiplies #}
{% set DSTRUCT_NARR = "cr15" %}  {# valid_elements=16, for both AAQ stores #}

    SET                 {{ mask_off }} {{ ZERO }};;
    SET                 {{ scratch_ptr }} {{ ZERO }};;
    SET                 {{ out_ptr }} {{ ZERO }};;

    # Step 1: dump r_acc's 128 raw FP32 lanes to the scratch row.
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT_WIDE }};
    STR_POST_AAQ_REG {{ scratch_ptr }} {{ SCRATCH_BASE }};;

    # Step 2: reload into R_CYCLIC. index must be an LR holding {0,128,256,384}.
    LDR_CYCLIC_MULT_REG {{ scratch_ptr }} {{ SCRATCH_BASE }} {{ mask_off }};;

    # Step 3: eight reload-at-offset multiplies, accumulating partitions 0..7.
    SET                 {{ rc_idx_reg }} {{ ZERO }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD.FIRST;;
    SET                 {{ rc_idx_reg }} {{ OFF16 }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD;;
    SET                 {{ rc_idx_reg }} {{ OFF32 }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD;;
    SET                 {{ rc_idx_reg }} {{ OFF48 }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD;;
    SET                 {{ rc_idx_reg }} {{ OFF64 }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD;;
    SET                 {{ rc_idx_reg }} {{ OFF80 }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD;;
    SET                 {{ rc_idx_reg }} {{ OFF96 }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD;;
    SET                 {{ rc_idx_reg }} {{ OFF112 }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD;;

    # Step 4: store the combined 16-lane result.
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT_NARR }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_BASE }};;

end:
    BKPT;;
