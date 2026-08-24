# Packed linear layer, path (a): PRE-REPLICATED WEIGHTS (compute-optimal).
# 240 input channels -> 8 output channels, 16 tokens (L5 packed-viability
# task, kernel C, the "hard case").
#
# Layer:   L5
# Scope:   single-stream
# Layout:  packed (input) -> unpacked (output)
# Shape:   240ch->8ch x 16tok, packing factor 8
# Status:  superseded -- ruled out for memory blowup (a full per-token weight
#          replica costs too much XMEM); see asm_packed_linear_240to8_masked.asm
#          (path (b), masked pass over unpacked weights) for the adopted
#          memory-optimal approach. Kept for the compute-optimal comparison
#          point; the prose below still describes the pre-replicated
#          construction as-implemented, which remains correct on its own terms.
# Related: asm_packed_linear_240to8_masked.asm (superseding path (b))
# Tests:   test_packed_linear_240to8.py
#
#   OUT[o, t] = sum_k W[o, k] * X[k, t]   for o = 0..7, k = 0..239
#
# Packed input: 240 channels PACKED 8/row (16 lanes/channel), 30 rows
# (chunks) total. Chunk c holds channels [8c, 8c+8) at lanes [16p, 16p+16)
# for p = 0..7.
#
# Per output o, per chunk c: multiply the WHOLE packed chunk by an ra
# VECTOR (not a scalar) where ra[16p+j] = W[o, 8c+p] for all j in 0..15
# (the weight replicated across each 16-lane partition) -- MULT.RC.VV
# (vector x vector), not MULT.RC.VE (vector x scalar); this is the only
# mechanism that applies 8 DIFFERENT per-partition scalars in one
# instruction (MULT.RC.VE broadcasts ONE scalar to all 128 lanes).
# ACC.ADD accumulates ALL 30 chunks' contributions into r_acc's 128 lanes
# WITHOUT combining partitions yet (addition is associative -- see the
# session's math check): after 30 chunks, r_acc[16p+t] holds
#   sum_{c=0..29} W[o, 8c+p] * X[8c+p, t]
# i.e. partition p's running sum over the 30 channels that landed in that
# partition. ONE combine (primitive A: store, reload at rc_idx=16p for
# p=0..7, accumulate) per OUTPUT CHANNEL -- not per chunk -- then finishes
# the reduction: out[o,t] = sum_p r_acc[16p+t].
#
# Weight rows: pre-replicated by the HOST (this is a data-layout choice for
# a static per-invocation input, same class of host-side staging
# qk_scores_16x60 already uses for its query-major Q layout) -- one row
# per (o, chunk) pair, holding W[o,8c+p] replicated 16x per partition.
# WEIGHTS_BASE + o*N_CHUNKS + c.
#
# MULT SNAPSHOT CONTRACT (issue #157): applies to BOTH R_CYCLIC data (via
# rc_idx) and R0 (the `ra` operand, also a "read: snapshot" field) -- see
# LDR_MULT_REG's execute_fn, which writes r0 live, and MULT.RC.VV's ra
# read=snapshot. LDR_MULT_REG (weight) and LDR_CYCLIC_MULT_REG (packed
# data) are BOTH "load" slot instructions -- only one load per VLIW bundle
# -- so unlike single-load-stream kernels elsewhere, both loads need their
# own cycle, one full chunk-pair ahead of the MULT that consumes them.
#
# CR budget: cr_idx for MULT.RC.VV/VE only feeds _mult_mask_and_shift,
# which with mask_shift=0 (used for every MULT.RC.* call below) never reads
# valid_elements or partition -- that branch is skipped entirely when
# shift==0. So every MULT.RC.* call below passes DSTRUCT_NARR (16) for
# cr_idx, wide or not; it is inert there. valid_elements genuinely matters
# only for ACTIVATE.QUANTIZE (crops the store): the scratch dump uses
# DSTRUCT_WIDE (128), the final result uses DSTRUCT_NARR (16).
#
# Two more CR-budget tricks (16 CRs is a hard limit and this kernel wants
# more named constants than that):
#   - partition 7's combine offset (112) is computed at runtime as
#     OFF96 + OFF16 via one extra ADD instead of costing a 7th offset CR.
#   - the do-while inner bound (N_CHUNKS - 2 = 28) is computed at runtime
#     via two SUBs into n_chunks_m2 (an LR) instead of costing its own CR.
#   - the o-loop exit test compares weight_row_off (already walking in
#     steps of N_CHUNKS) against N_TOTAL_ROWS = 8*N_CHUNKS = 240, instead
#     of maintaining a separate o_idx-vs-8 comparison that would need an
#     "8" CR.

{% set data_ptr       = "lr0"  %}  {# packed-chunk row pointer, walks 0..29, RESET per output (X reused per o) #}
{% set rc_slot0       = "lr1"  %}  {# const 0: r_cyclic write-index #}
{% set chunk_idx      = "lr2"  %}  {# runtime chunk counter 0..29 #}
{% set w_ptr          = "lr3"  %}  {# weight row pointer: weight_row_off + chunk_idx #}
{% set out_ptr        = "lr4"  %}  {# output row pointer, += 1 per o #}
{% set n_chunks_m2    = "lr5"  %}  {# 28 = N_CHUNKS - 2, the do-while inner bound (computed once at setup) #}
{% set weight_row_off = "lr6"  %}  {# o*N_CHUNKS, += N_CHUNKS per o; also the o-loop progress counter #}
{% set scratch_ptr    = "lr7"  %}  {# combine scratch row pointer (primitive A step 1/2) #}
{% set combine_rc     = "lr8"  %}  {# combine reload rc_idx, varies per partition #}

{% set ZERO          = "cr0"  %}  {# const 0 #}
{% set ONE           = "cr1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE     = "cr2"  %}  {# packed X base row MINUS ONE (harness pre-biases it -- see the pre-increment note above) #}
{% set WEIGHTS_BASE  = "cr3"  %}  {# weight base row #}
{% set OUTPUT_BASE   = "cr4"  %}  {# OUT base row #}
{% set N_CHUNKS      = "cr5"  %}  {# 30 #}
{% set SCRATCH_BASE  = "cr6"  %}  {# combine scratch base row #}
{% set OFF16         = "cr7"  %}  {# 16: combine reload offset, partition 1 (also the +16 step to reach partition 7) #}
{% set OFF32         = "cr8"  %}  {# 32: combine reload offset, partition 2 #}
{% set OFF48         = "cr9"  %}  {# 48: combine reload offset, partition 3 #}
{% set OFF64         = "cr10" %}  {# 64: combine reload offset, partition 4 #}
{% set OFF80         = "cr11" %}  {# 80: combine reload offset, partition 5 #}
{% set OFF96         = "cr12" %}  {# 96: combine reload offset, partition 6 (base for partition 7 = 96+16) #}
{% set N_TOTAL_ROWS  = "cr13" %}  {# 240 = N_OUT * N_CHUNKS: o-loop exit bound, compared against weight_row_off #}
{% set DSTRUCT_WIDE  = "cr14" %}  {# valid_elements=128, for the scratch dump only #}
{% set DSTRUCT_NARR  = "cr15" %}  {# valid_elements=16, for the final result store and (inertly) every MULT.RC.* call #}

    SET {{ n_chunks_m2 }} {{ N_CHUNKS }};;
    SUB {{ n_chunks_m2 }} {{ n_chunks_m2 }} {{ ONE }};;
    SUB {{ n_chunks_m2 }} {{ n_chunks_m2 }} {{ ONE }};;
    SET {{ weight_row_off }} {{ ZERO }};;
    SET {{ out_ptr }} {{ OUTPUT_BASE }};;

o_loop:
    # ---- chunk 0: prime + peel first chunk (ACC.ADD.FIRST) ----
    SET {{ data_ptr }} {{ DATA_BASE }};;
    ADD {{ w_ptr }} {{ weight_row_off }} {{ ZERO }};;
    LDR_MULT_REG r0 {{ w_ptr }} {{ WEIGHTS_BASE }};;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};;
    SET {{ chunk_idx }} {{ ZERO }};;

    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD.FIRST;
    ADD {{ w_ptr }} {{ w_ptr }} {{ ONE }};
    ADD {{ chunk_idx }} {{ chunk_idx }} {{ ONE }};
    BLT {{ chunk_idx }} {{ n_chunks_m2 }} chunk_loop;;
    B after_chunks;;

chunk_loop:
    LDR_MULT_REG r0 {{ w_ptr }} {{ WEIGHTS_BASE }};;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};;
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;
    ADD {{ w_ptr }} {{ w_ptr }} {{ ONE }};
    ADD {{ chunk_idx }} {{ chunk_idx }} {{ ONE }};
    BLT {{ chunk_idx }} {{ n_chunks_m2 }} chunk_loop;;

after_chunks:
    # Last chunk (chunk_idx == N_CHUNKS-2): weight/data still need loading.
    LDR_MULT_REG r0 {{ w_ptr }} {{ WEIGHTS_BASE }};;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};;
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;;

    # ---- primitive A: combine 8 partitions of r_acc into 16 lanes ----
    ACTIVATE.QUANTIZE identity {{ DSTRUCT_WIDE }};
    STR_POST_AAQ_REG {{ scratch_ptr }} {{ SCRATCH_BASE }};;
    LDR_CYCLIC_MULT_REG {{ scratch_ptr }} {{ SCRATCH_BASE }} {{ rc_slot0 }};;

    SET {{ combine_rc }} {{ ZERO }};;
    MULT.RC.VE {{ combine_rc }} {{ ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD.FIRST;;
    SET {{ combine_rc }} {{ OFF16 }};;
    MULT.RC.VE {{ combine_rc }} {{ ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;;
    SET {{ combine_rc }} {{ OFF32 }};;
    MULT.RC.VE {{ combine_rc }} {{ ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;;
    SET {{ combine_rc }} {{ OFF48 }};;
    MULT.RC.VE {{ combine_rc }} {{ ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;;
    SET {{ combine_rc }} {{ OFF64 }};;
    MULT.RC.VE {{ combine_rc }} {{ ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;;
    SET {{ combine_rc }} {{ OFF80 }};;
    MULT.RC.VE {{ combine_rc }} {{ ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;;
    SET {{ combine_rc }} {{ OFF96 }};;
    MULT.RC.VE {{ combine_rc }} {{ ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;;
    ADD {{ combine_rc }} {{ combine_rc }} {{ OFF16 }};;
    MULT.RC.VE {{ combine_rc }} {{ ONE }} 0 {{ rc_slot0 }} {{ DSTRUCT_NARR }};
    ACC.ADD;;

    ACTIVATE.QUANTIZE identity {{ DSTRUCT_NARR }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;
    ADD {{ out_ptr }} {{ out_ptr }} {{ ONE }};;

    ADD {{ weight_row_off }} {{ weight_row_off }} {{ N_CHUNKS }};;
    BLT {{ weight_row_off }} {{ N_TOTAL_ROWS }} o_loop;;

end:
    BKPT;;
