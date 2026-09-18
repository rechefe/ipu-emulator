# On-chip PACK: unpacked (one channel per row, 192 rows) -> packed chunks
# (2 channels/row, 96 rows).
#
# Layer:   L4
# Scope:   single-stream
# Layout:  seam (unpacked -> packed)
# Shape:   192ch x 64tok, packing factor 2
# Status:  validated
# Related: asm_packed_pack_240x16.asm (L5 counterpart); asm_packed_unpack_192x64.asm
#          (reverse-direction seam partner)
# Tests:   test_packed_pack_unpack_192x64.py, exercised via test_full_layer_l4_packed.py
#
# L4 port of asm_packed_pack_240x16.asm: same
# construction, only the partition count/width change (8x16 -> 2x64, per
# partition_size(64)=64 -> parts_per_chunk=128/64=2). For the same
# attention-seam boundary (an unpacked producer -- attn·V, read-only, not
# modified -- feeding a packed consumer downstream).
#
# Construction: for each packed chunk c (0..95), for p_out=0..1: load
# UNPACKED_BASE row (2*c+p_out) into R_CYCLIC slot 0 (that row's 64 valid
# tokens sit at lanes 0-63), then a masked SCATTER write --
# rc_idx=(-64*p_out) mod 512, mask window p_out -- lands those same 64
# values into window p_out of the shared r_acc (same construction as
# asm_packed_layernorm_192x64.asm's step 2/5 broadcast, generalized: there
# the SAME 64 source values feed every window; here each window's SOURCE
# is a DIFFERENT row, loaded fresh before each window's own write).
# ACC.ADD.FIRST at p_out==0 (fired at the START OF EVERY CHUNK, not just
# once for the whole kernel run) then ACC.ADD at p_out=1 accumulates into
# disjoint 64-lane windows of one shared r_acc -- no reset between the 2
# source rows of one packed chunk, but a genuine reset at the start of each
# new chunk's row (same cross-chunk reset fix as
# asm_packed_pack_240x16.asm -- see docs/isa_friction_log.md's "Pack/unpack
# seam kernels: cross-chunk ACC.ADD reset bug" entry; an unconditional
# ACC.ADD at p_out==0 would let chunk c's first window silently add onto
# chunk (c-1)'s stale r_acc content). One store per chunk
# (valid_elements=128, the whole packed row).
#
# PIPELINING: every p_out's own MULT consumes the row PREFETCHED by the
# PREVIOUS iteration (or the kernel's own priming load, for the very
# first). Every iteration therefore ALSO issues its own trailing
# prefetch load for the FOLLOWING row -- including p_out=1, which
# prefetches the NEXT packed chunk's p_out=0 row -- except the absolute
# last iteration of the whole kernel (chunk 95, p_out=1), which has
# nothing left to prefetch (192 rows total, all already consumed).
#
# STANDALONE: does not modify or import qk_scores_64x48, attn_v_64x48,
# attn_scores_km_64x48, attn_v_bcast_48, layernorm_64x192, or any softmax
# kernel.

{% set rc_idx_reg   = "lr0"  %}
{% set mask_off_lr  = "lr1"  %}
{% set read_ptr     = "lr2"  %}
{% set write_ptr    = "lr3"  %}
{% set row_idx      = "lr4"  %}
{% set row_limit    = "lr5"  %}
{% set row_stride   = "lr6"  %}
{% set mask_shift0  = "lr7"  %}
{% set slot0_lr     = "lr8"  %}

{% set ZERO          = "cr0"  %}
{% set ONE           = "cr1"  %}
{% set UNPACKED_BASE = "cr2"  %}
{% set ROW_COUNT     = "cr3"  %}
{% set ROW_STRIDE    = "cr4"  %}
{% set PACKED_BASE   = "cr5"  %}
{% set MASK_BASE     = "cr6"  %}
{% set DSTRUCT128    = "cr7"  %}
{#- cr8-cr15 unused. #}

{% set N_PACKED_ROWS = 96 %}
{% set PS = 64 %}

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

    SET {{ read_ptr }} {{ UNPACKED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;   # write_ptr = -1: STR_POST_AAQ_REG's offset is a LIVE read and dispatches after this bundle's own ADD
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime unpacked row 0 into slot 0

    # ---- chunk 0 (peeled) ----
{%- for p_out in range(2) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - PS * p_out) % 512) }}
    {%- set is_first = (p_out == 0) %}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p_out }} {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
{%- endfor %}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ PACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1

pack_loop:
{%- for p_out in range(2) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - PS * p_out) % 512) }}
    {%- if p_out < 1 %}
    {#- p_out==0 must be ACC.ADD.FIRST -- it starts a NEW packed row's
        accumulation, not a continuation of the PREVIOUS chunk's r_acc. #}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p_out }} {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- else %}
    {#- p_out=1's load+row_idx/BLT all fuse into ONE bundle. This load
        prefetches the NEXT packed chunk's p_out=0 row;
        on the LAST loop;
        pass (processing chunk 95) it harmlessly reads one row past the
        last valid input (row 192) -- the result is never consumed. #}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p_out }} {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- endif %}
{%- endfor %}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ PACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} pack_loop;;

end:
    BKPT;;
