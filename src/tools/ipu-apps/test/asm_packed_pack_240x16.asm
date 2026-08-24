# On-chip PACK: unpacked (one channel per row, 240 rows) -> packed chunks
# (8 channels/row, 30 rows).
#
# Layer:   L5
# Scope:   single-stream
# Layout:  seam (unpacked -> packed)
# Shape:   240ch x 16tok, packing factor 8
# Status:  validated
# Related: asm_packed_pack_192x64.asm (L4 counterpart); asm_packed_unpack_240x16.asm
#          (reverse-direction seam partner)
# Tests:   test_packed_pack_unpack_240x16.py, exercised via test_full_layer_l5_packed.py
#
# Reverse of asm_packed_unpack_240x16.asm, for
# the same attention-seam boundary (an unpacked producer -- attn·V,
# read-only, not modified -- feeding a packed consumer downstream).
#
# Construction: for each packed chunk c (0..29), for p_out=0..7: load
# UNPACKED_BASE row (8*c+p_out) into R_CYCLIC slot 0 (that row's 16 valid
# tokens sit at lanes 0-15), then a masked SCATTER write --
# rc_idx=(-16*p_out) mod 512, mask window p_out -- lands those same 16
# values into window p_out of the shared r_acc (same construction as
# asm_packed_layernorm_240x16.asm's step 2/5 broadcast, generalized: there
# the SAME 16 source values feed every window; here each window's SOURCE
# is a DIFFERENT row, loaded fresh before each window's own write).
# ACC.ADD.FIRST at p_out==0 (fired at the START OF EVERY CHUNK, not just
# once for the whole kernel run) then ACC.ADD for p_out=1..7 accumulates
# into disjoint 16-lane windows of one shared r_acc -- no reset between the
# 8 source rows of one packed chunk, but a genuine reset at the start of
# each new chunk's row. An earlier draft used unconditional ACC.ADD at
# p_out==0 (copying the "loop body always uses ACC.ADD, only the outer peel
# uses .FIRST" shape from asm_packed_layernorm_240x16.asm's step 1, where
# the loop genuinely sums ACROSS chunks) -- wrong here since this kernel's
# chunks are independent, not cross-chunk accumulations; that bug let chunk
# c's first window silently add onto chunk (c-1)'s stale r_acc content,
# compounding across chunks. Fixed by using ACC.ADD.FIRST at p_out==0 in
# both the peeled first chunk and the runtime loop body -- see
# docs/isa_friction_log.md's "Pack/unpack seam kernels: cross-chunk
# ACC.ADD reset bug" entry. One store per chunk (valid_elements=128, the
# whole packed row).
#
# PIPELINING: every p_out's own MULT consumes the row PREFETCHED by the
# PREVIOUS iteration (or the kernel's own priming load, for the very
# first). Every iteration therefore ALSO issues its own trailing
# prefetch load for the FOLLOWING row -- including p_out=7, which
# prefetches the NEXT packed chunk's p_out=0 row -- except the absolute
# last iteration of the whole kernel (chunk 29, p_out=7), which has
# nothing left to prefetch (240 rows total, all already consumed). An
# earlier draft omitted p_out=7's load unconditionally (treating it like
# "the last unroll slot needs no prefetch," true only for the standalone
# 8-wide gather primitive this was adapted from, which reads ONE shared
# row 8 times rather than 8 DIFFERENT rows once each) -- that starves
# every chunk after the first of 1 of its 8 needed loads (7 issued
# instead of 8 per chunk, 210 total vs 240 needed), caught by desk-check
# before assembling: counting required loads per chunk against the
# unrolled bundle structure, not by a failing test.
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

{% set ZERO          = "cr0"  %}
{% set ONE           = "cr1"  %}
{% set UNPACKED_BASE = "cr2"  %}
{% set ROW_COUNT     = "cr3"  %}
{% set ROW_STRIDE    = "cr4"  %}
{% set PACKED_BASE   = "cr5"  %}
{% set MASK_BASE     = "cr6"  %}
{% set DSTRUCT128    = "cr7"  %}
{#- cr8-cr15 unused. DSTRUCT16 is not needed: every load here reads a
    one-channel-per-row source (native 128-lane width;
    only 16 lanes are;
    "live" data but the MULT reads all 128 regardless of quantize
    windows), and the only STORE is the packed 128-lane output. #}

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

    SET {{ read_ptr }} {{ UNPACKED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;   # write_ptr = -1: STR_POST_AAQ_REG's offset is a LIVE read and dispatches after this bundle's own ADD -- see asm_packed_layernorm_240x16.asm's step 3 note
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime unpacked row 0 into slot 0

    # ---- chunk 0 (peeled) ----
{%- for p_out in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - 16 * p_out) % 512) }}
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
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1 -- see asm_packed_layernorm_240x16.asm's loop-count derivation note

pack_loop:
{%- for p_out in range(8) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - 16 * p_out) % 512) }}
    {%- if p_out < 7 %}
    {#- p_out==0 must be ACC.ADD.FIRST -- it starts a NEW packed row's
        accumulation, not a continuation of the PREVIOUS chunk's r_acc.
        An earlier draft used unconditional ACC.ADD here (copying the
        loop-body-vs-peel split from kernels where the loop's first
        window genuinely continues an existing sum, e.g.
        asm_packed_layernorm_240x16.asm's step1, which sums ACROSS
        chunks) -- but this kernel's chunks are INDEPENDENT rows, each
        needing its own fresh r_acc, so omitting .FIRST let each new
        chunk's first window silently add onto the PREVIOUS chunk's
        stale r_acc content, compounding across chunks (row0 correct,
        row1 off by ~3, row9 off by ~13 -- a growing error, the
        signature of unbounded cross-iteration accumulation rather than
        a one-off indexing mistake). #}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p_out }} {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if p_out == 0 else "" }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- else %}
    {#- p_out=7's load+row_idx/BLT all fuse into ONE bundle: write_ptr,
        read_ptr, row_idx = exactly 3 "lr" ops (the per-cycle maximum),
        alongside the store (different slot) and this load (different
        slot again) -- mirrors asm_packed_layernorm_240x16.asm's identical
        fused closing-bundle structure. This load prefetches the NEXT
        packed chunk's p_out=0 row;
        on the LAST loop pass (processing;
        chunk 29) it harmlessly reads one row past the last valid input
        (row 240) -- the same harmless-extra-prefetch pattern
        layernorm_16x240.asm's own step1 loop has (see that kernel's
        241-loads-for-240-channels count, verified empirically earlier
        this session): the load's result is never consumed by another
        MULT, since the loop exits right after. #}
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
