# Packed linear layer, path (b) -- GENERIC (K, N_OUT) parameterization of
# asm_packed_linear_240to8_masked.asm, for the L5 real-size measurement
# (QKV 240->720, outproj 240->240, FFN1 240->480, FFN2 480->240).
#
# Layer:   L5
# Scope:   single-stream
# Layout:  packed (input) -> unpacked (output)
# Shape:   Kch->N_OUTch x 16tok, packing factor 8 (K, N_OUT parameterized)
# Status:  template (generic Jinja template, not directly tested standalone)
# Related: asm_packed_linear_240to8_masked.asm (fixed 240->8 instance this generalizes)
# Tests:   exercised via test_l5_real_size_packed_b.py
#
#   OUT[o, t] = sum_k W[o, k] * X[k, t]   for o = 0..N_OUT-1, k = 0..K-1
#
# Requires K % 8 == 0 (packing group size) -- true for all four L5 shapes
# (K in {240, 480}). Weight layout: output-major, W_CHUNKS = ceil(K/128)
# rows of up to 128 raw (unreplicated) scalars/row -- identical convention
# to asm_unpacked_linear_240to8.asm, generalized to W_CHUNKS>2 (needed for
# FFN2's K=480 -> W_CHUNKS=4).
#
# Every weight-chunk width is a multiple of 8 by construction (128 and the
# final tail K - 128*(W_CHUNKS-1), both used here only at K in {240,480}
# where the tail is 112 -- still %8==0), so each weight-chunk unrolls into
# an exact whole number of 8-partition packed-chunk passes, same alignment
# argument as the 240->8 version, with NO runtment branching anywhere.
#
# IPC fix applied throughout (see asm_packed_linear_240to8_masked.asm's IPC
# FIX comment for the full derivation): rc_idx_reg/k_idx pre-increment
# biased so their per-partition advance ADDs co-issue in the SAME bundle as
# the MULT.RC.VE that consumes the advanced value.
#
# This is a Jinja TEMPLATE rendered from Python (test harness passes K,
# N_OUT, and the per-weight-chunk width list as render context) -- not
# hand-specialized per shape.

{% set data_ptr    = "lr0"  %}
{% set rc_idx_reg  = "lr1"  %}
{% set k_idx       = "lr2"  %}
{% set chunk_w_ptr = "lr4"  %}
{% set out_ptr     = "lr6"  %}
{% set o_idx       = "lr7"  %}
{% set weight_row_off = "lr8" %}
{% set load_idx    = "lr9"  %}

{% set ZERO         = "cr0"  %}
{% set ONE          = "cr1"  %}
{% set DATA_BASE    = "cr2"  %}
{% set WEIGHTS_BASE = "cr3"  %}
{% set OUTPUT_BASE  = "cr4"  %}
{% set N_OUT_CR     = "cr9"  %}
{% set W_CHUNKS_CR  = "cr8"  %}
{% set SIXTEEN      = "cr10" %}
{% set NEG_SIXTEEN  = "cr11" %}
{% set NEG_ONE      = "cr12" %}
{% set DSTRUCT      = "cr15" %}

    SET {{ o_idx }} {{ ZERO }};;
    SET {{ weight_row_off }} {{ ZERO }};;
    SET {{ out_ptr }} {{ OUTPUT_BASE }};;

o_loop:
    SET {{ data_ptr }} {{ DATA_BASE }};;
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ ZERO }};;

    {%- for chunk_width in chunk_widths %}
    {%- set outer_index = loop.index0 %}
    {%- set outer_last = loop.last %}
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;
    SET {{ k_idx }} {{ NEG_ONE }};;
    {%- for pc in range(chunk_width // 8) %}
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ load_idx }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    SET {{ rc_idx_reg }} {{ NEG_SIXTEEN }};;
    {%- for p in range(8) %}
    {%- set is_first = (outer_index == 0 and pc == 0 and loop.index0 == 0) %}
    MULT.RC.VE {{ rc_idx_reg }} {{ k_idx }} 0 {{ load_idx }} {{ DSTRUCT }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};
    ADD {{ rc_idx_reg }} {{ rc_idx_reg }} {{ SIXTEEN }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;
    {%- endfor %}
    {%- endfor %}
    {%- if not outer_last %}
    ADD {{ chunk_w_ptr }} {{ chunk_w_ptr }} {{ ONE }};;
    {%- endif %}
    {%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;
    ADD {{ out_ptr }} {{ out_ptr }} {{ ONE }};;

    ADD {{ weight_row_off }} {{ weight_row_off }} {{ W_CHUNKS_CR }};
    INC {{ o_idx }} 1;;
    BLT {{ o_idx }} {{ N_OUT_CR }} o_loop;;

end:
    BKPT;;
