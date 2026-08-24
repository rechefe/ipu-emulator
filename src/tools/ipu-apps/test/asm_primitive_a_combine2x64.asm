# Primitive A -- cross-partition combine, L4 shape (2 partitions x 64 lanes).
#
# Layer:   L4
# Scope:   single-stream
# Layout:  n/a (register-state microbenchmark, not a full kernel)
# Shape:   64ch equivalent, packing factor 2
# Status:  validated
# Related: asm_primitive_a_combine8x16.asm (L5 counterpart)
# Tests:   test_primitive_a_combine2x64.py
#
# L4 port of asm_primitive_a_combine8x16.asm: same mechanism, only the
# partition count/width change (8x16 -> 2x64, per partition_size(64)=64 ->
# parts_per_chunk=128/64=2 -- see docs/isa_friction_log.md's L4 entry).
#
# Input:  r_acc holds 2 partitions of 64 lanes (128 lanes total), pre-loaded
#         directly by the test harness (host-side r_acc write -- there is no
#         XMEM data for this microbenchmark, only the register state).
# Output: 64 FP32 values = elementwise sum across the 2 partitions, stored
#         to XMEM row OUT_BASE (first 64 lanes valid; teardown crops).
#
# Method (unchanged from the L5 version -- AGG.SUM's collapse-to-scalar and
# RESHAPE's R_ACC-only/8-lane-per-call limits still block a direct segmented
# reduce at this shape too):
#   1. ACTIVATE.QUANTIZE identity (valid_elements=128) + STR_POST_AAQ_REG
#      dumps r_acc's raw 128 FP32 lanes to XMEM.
#   2. LDR_CYCLIC_MULT_REG reloads those 128 elements into R_CYCLIC (index=0).
#   3. Two MULT.RC.VE x 1.0 calls, rc_idx = 64*p for p=0..1 (ELEMENT offset),
#      each landing partition p's 64 values in mult_res lanes 0..63.
#      ACC.ADD.FIRST on p=0, ACC.ADD on p=1 sums them into r_acc[0..63].
#   4. ACTIVATE.QUANTIZE identity (valid_elements=64) + STR_POST_AAQ_REG
#      stores the 64-lane combined result.

{% set rc_idx_reg  = "lr0"  %}  {# r_cyclic base ELEMENT offset, reset per partition #}
{% set mask_off    = "lr1"  %}  {# const 0: mask_offset #}
{% set scratch_ptr = "lr2"  %}  {# row offset for the scratch dump/reload row #}
{% set out_ptr     = "lr3"  %}  {# row offset for the combined-result row #}

{% set SCRATCH_BASE = "cr0"  %}  {# base row: r_acc dump / reload scratch #}
{% set ONE          = "cr1"  %}  {# hardwired read-only 1 #}
{% set OUT_BASE     = "cr2"  %}  {# base row: combined 64-lane result #}
{% set OFF64        = "cr3"  %}  {# 64  = rc_idx for partition 1 #}
{% set ZERO         = "cr4"  %}  {# const 0 = rc_idx for partition 0 #}
{% set DTYPE_ONE    = "cr10" %}  {# dtype-encoded 1.0 scalar for the pass-through MULT #}
{% set DSTRUCT_WIDE = "cr14" %}  {# valid_elements=128, for the reload multiplies #}
{% set DSTRUCT_NARR = "cr15" %}  {# valid_elements=64, for both AAQ stores #}

    SET                 {{ mask_off }} {{ ZERO }};;
    SET                 {{ scratch_ptr }} {{ ZERO }};;
    SET                 {{ out_ptr }} {{ ZERO }};;

    # Step 1: dump r_acc's 128 raw FP32 lanes to the scratch row.
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT_WIDE }};
    STR_POST_AAQ_REG {{ scratch_ptr }} {{ SCRATCH_BASE }};;

    # Step 2: reload into R_CYCLIC. index must be an LR holding {0,128,256,384}.
    LDR_CYCLIC_MULT_REG {{ scratch_ptr }} {{ SCRATCH_BASE }} {{ mask_off }};;

    # Step 3: two reload-at-offset multiplies, accumulating partitions 0..1.
    SET                 {{ rc_idx_reg }} {{ ZERO }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD.FIRST;;
    SET                 {{ rc_idx_reg }} {{ OFF64 }};;
    MULT.RC.VE          {{ rc_idx_reg }} {{ DTYPE_ONE }} 0 {{ mask_off }} {{ DSTRUCT_WIDE }};
    ACC.ADD;;

    # Step 4: store the combined 64-lane result.
    ACTIVATE.QUANTIZE   identity {{ DSTRUCT_NARR }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_BASE }};;

end:
    BKPT;;
