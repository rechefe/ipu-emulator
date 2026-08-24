# Unpacked linear layer baseline (L5 packed-viability task, kernel C
# comparison): 240 input channels -> 8 output channels, 16 tokens.
#   OUT[o, t] = sum_k W[o, k] * X[k, t]   for o = 0..7, k = 0..239
#
# Layer:   L5
# Scope:   single-stream
# Layout:  unpacked
# Shape:   240ch->8ch x 16tok, packing factor n/a
# Status:  validated
# Related: asm_packed_linear_240to8_masked.asm, asm_packed_linear_240to8_replicated.asm
#          (packed comparison points against this baseline)
# Tests:   test_linear_240to8_unpacked_baseline.py
#
# ONE CHANNEL PER ROW throughout (matches every other L5 kernel's
# convention): X has 240 rows (one input channel/row, 16 live lanes),
# OUT has 8 rows (one output channel/row, 16 live lanes).
#
# Weight layout: output-major [8, 240], same mechanism proj_qkv_240_p4 uses --
# each weight-chunk row holds up to 128 scalars, LDR_MULT_REG loads the whole
# row into R0, and MULT.RC.VE's `src` (an LR) selects one scalar from R0 by
# index per k-step.
#
# Weight rows: WEIGHTS_BASE + o*W_CHUNKS + c holds up to 128 weights for
# output o, chunk c (W_CHUNKS = ceil(240/128) = 2: chunk0 width 128,
# chunk1 (tail) width 112).
#
# MULT SNAPSHOT CONTRACT (issue #157): same rule as every other matmul-style
# kernel here -- MULT.RC.VE reads r_cyclic DATA from the start-of-cycle
# snapshot, so a same-bundle LDR_CYCLIC_MULT_REG load is not visible until
# the NEXT cycle. Data pointer therefore primes one row ahead of chunk 0's
# first MULT, exactly like proj_qkv_240_p4 / qk_scores_16x60.

{% set data_ptr    = "lr0"  %}  {# data (X) row pointer, walks 0..239 continuously #}
{% set rc_slot0    = "lr1"  %}  {# const 0: r_cyclic write-index #}
{% set k_idx       = "lr2"  %}  {# fixed_idx (MULT.RC.VE rc_idx into R0), reset per chunk, biased chunk0 #}
{% set bound       = "lr3"  %}  {# selected inner bound for the current chunk #}
{% set chunk_w_ptr = "lr4"  %}  {# weight row pointer: o*W_CHUNKS + chunk_idx #}
{% set chunk_idx   = "lr5"  %}  {# runtime chunk counter 0..1 #}
{% set out_ptr     = "lr6"  %}  {# output row pointer, += 1 per o #}
{% set o_idx       = "lr7"  %}  {# output-channel counter 0..7 #}
{% set weight_row_off = "lr8" %} {# o*W_CHUNKS, += W_CHUNKS per o #}

{% set ZERO         = "cr0"  %}  {# const 0 #}
{% set ONE          = "cr1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE    = "cr2"  %}  {# X base row #}
{% set WEIGHTS_BASE = "cr3"  %}  {# W base row #}
{% set OUTPUT_BASE  = "cr4"  %}  {# OUT base row #}
{% set NEG_ONE      = "cr5"  %}  {# -1 #}
{% set FULL_BOUND   = "cr6"  %}  {# 126: inner bound for a width-128 chunk #}
{% set TAIL_BOUND   = "cr7"  %}  {# 110: inner bound for the width-112 tail chunk (112-2) #}
{% set W_CHUNKS     = "cr8"  %}  {# 2 = ceil(240/128) #}
{% set N_OUT        = "cr9"  %}  {# 8 #}
{% set LAST_CHUNK_IDX = "cr10" %} {# W_CHUNKS - 1: only this chunk_idx uses TAIL_BOUND #}
{% set DSTRUCT      = "cr15" %}  {# valid_elements=16 (one channel's live tokens) #}

    SET {{ o_idx }} {{ ZERO }};;
    SET {{ weight_row_off }} {{ ZERO }};;
    SET {{ out_ptr }} {{ OUTPUT_BASE }};;

o_loop:
    # ---- chunk 0: prime + peel first k-iter ----
    SET {{ data_ptr }} {{ DATA_BASE }};;
    ADD {{ data_ptr }} {{ data_ptr }} {{ NEG_ONE }};;
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ ZERO }};;
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;
    SET {{ k_idx }} {{ NEG_ONE }};;
    SUB {{ k_idx }} {{ k_idx }} {{ ONE }};;
    SET {{ chunk_idx }} {{ ZERO }};;

    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;

    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ FULL_BOUND }} k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ FULL_BOUND }} k_chunk0;;

after_chunk0:
    INC {{ chunk_idx }} 1;;
    BLT {{ chunk_idx }} {{ W_CHUNKS }} chunk_loop;;
    B store_out;;

chunk_loop:
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ chunk_idx }};;
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};
    SET {{ k_idx }} {{ NEG_ONE }};;

    {# bound_sel: only the LAST chunk (chunk_idx == W_CHUNKS-1) uses
       TAIL_BOUND;
       every other chunk is a full width-128 chunk and must use;
       FULL_BOUND. Missing this check (using TAIL_BOUND unconditionally) is
       invisible at W_CHUNKS<=2 -- the only case this kernel was originally
       tested at -- because chunk_loop then runs at most once and that one
       iteration IS the last chunk. It produces silently wrong sums for any
       K needing >2 weight chunks (W_CHUNKS>=3, K>256): found via the L5
       packed-viability FFN2 extrapolation (K=480, W_CHUNKS=4). #}
    BLT {{ chunk_idx }} {{ LAST_CHUNK_IDX }} use_full_bound;;
    SET {{ bound }} {{ TAIL_BOUND }};;
    B chunk_body;;
use_full_bound:
    SET {{ bound }} {{ FULL_BOUND }};;

chunk_body:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ bound }} chunk_body;;

    INC {{ chunk_idx }} 1;;
    BLT {{ chunk_idx }} {{ W_CHUNKS }} chunk_loop;;

store_out:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;
    ADD {{ out_ptr }} {{ out_ptr }} {{ ONE }};;

    ADD {{ weight_row_off }} {{ weight_row_off }} {{ W_CHUNKS }};
    INC {{ o_idx }} 1;;
    BLT {{ o_idx }} {{ N_OUT }} o_loop;;

end:
    BKPT;;
