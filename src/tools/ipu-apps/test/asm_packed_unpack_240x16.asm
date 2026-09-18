# On-chip UNPACK: packed chunks (8 channels/row, 30 rows, 240 channels x 16
# tokens) -> unpacked (one channel per row, 240 rows). Attention-seam
# converter: QKᵀ/softmax/attn·V stay unpacked (structural -- scores have no
# channel axis), so a packed producer's output must convert on-chip before
# feeding an unpacked consumer (qk_scores_16x60 etc, read-only, not modified).
#
# Layer:   L5
# Scope:   single-stream
# Layout:  seam (packed -> unpacked)
# Shape:   240ch x 16tok, packing factor 8
# Status:  validated
# Related: asm_packed_unpack_192x64.asm (L4 counterpart); asm_packed_pack_240x16.asm
#          (reverse-direction seam partner)
# Tests:   test_packed_pack_unpack_240x16.py, exercised via test_full_layer_l5_packed.py
#
# Construction: for each packed chunk c (0..29), gather each of its 8
# partitions p_in=0..7 into lanes 0-15 via rc_idx=16*p_in, mask window 0
# (one-hot lanes 0-15), ACC.ADD.FIRST (single term, no accumulation across
# p_in -- each output row is independent, unlike packed layernorm's step 1
# which SUMS across p_in into one shared result). Store valid_elements=16
# to output row 8*c+p_in. Runtime loop over chunks (same fused-bundle
# loop-counter discipline as asm_packed_layernorm_240x16.asm -- see that
# file's header/friction-log entries for the loop-count derivation this
# mirrors exactly).
#
# STANDALONE: does not modify or import qk_scores_16x60, attn_v_16x60, or
# any softmax kernel.

{% set rc_idx_reg   = "lr0"  %}
{% set mask_off_lr  = "lr1"  %}
{% set read_ptr     = "lr2"  %}
{% set write_ptr    = "lr3"  %}
{% set row_idx      = "lr4"  %}
{% set row_limit    = "lr5"  %}
{% set row_stride   = "lr6"  %}
{% set mask_shift0  = "lr7"  %}
{% set slot0_lr     = "lr8"  %}

{% set ZERO         = "cr0"  %}
{% set ONE          = "cr1"  %}
{% set PACKED_BASE  = "cr2"  %}
{% set ROW_COUNT    = "cr3"  %}
{% set ROW_STRIDE   = "cr4"  %}
{% set UNPACKED_BASE = "cr5" %}
{% set MASK_BASE    = "cr6"  %}
{% set DSTRUCT16    = "cr7"  %}
{% set DSTRUCT128   = "cr8"  %}
{#- cr9-cr15 unused. #}

{% set N_PACKED_ROWS = 30 %}

{% macro inc(dest, n) %}
{%- set n = n % 512 -%}
{%- if n != 0 -%}
{%- for chunk in ((n // 255) * [255] + ([n % 255] if n % 255 else [])) %}
    INC {{ dest }} {{ chunk }};;
{%- endfor -%}
{%- endif -%}
{% endmacro %}

    SET {{ mask_off_lr }} {{ ZERO }};;
    SET {{ mask_shift0 }} {{ ZERO }};;
    SET {{ row_stride }} {{ ROW_STRIDE }};;
    SET {{ slot0_lr }} {{ ZERO }};;
    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ MASK_BASE }};;

    SET {{ read_ptr }} {{ PACKED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;   # write_ptr = -1: STR_POST_AAQ_REG's offset is a LIVE read and dispatches after this bundle's own ADD -- see asm_packed_layernorm_240x16.asm's step 3 note
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime chunk 0 into slot 0

    # ---- chunk 0 (peeled) ----
{%- for p_in in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, 16 * p_in) }}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    {%- if p_in < 7 %}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT16 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ UNPACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;
    {%- else %}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT16 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ UNPACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- endif %}
{%- endfor %}

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1 -- see asm_packed_layernorm_240x16.asm's loop-count derivation note

unpack_loop:
{%- for p_in in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, 16 * p_in) }}
    {%- if p_in < 7 %}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT16 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ UNPACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;
    {%- else %}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    {#- final p_in of this chunk: fuse write_ptr's advance, the trailing
        prefetch load+advance, AND row_idx's own advance+BLT all into one
        bundle -- exactly 3 "lr" ops (write_ptr, read_ptr, row_idx), the
        per-cycle maximum -- mirroring asm_packed_layernorm_240x16.asm's
        loop structure precisely (see that file for why row_idx's ADD and
        BLT must share the closing load's bundle). #}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT16 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ UNPACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} unpack_loop;;
    {%- endif %}
{%- endfor %}

end:
    BKPT;;
