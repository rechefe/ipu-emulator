{#- ==========================================================================
    packed_unpack.asm -- the MobileViT packed layout -> one channel per row,
    FP32. The template of packed_unpack_{192x64,240x16} (../registry.toml,
    E07-23), the old test/mobilevit/asm_packed_unpack_<shape>.asm programs
    (E02 digests).

    An attention-seam converter: the scores, softmax and attn.V kernels stay
    unpacked (the scores have no channel axis), so a packed producer's output
    is unpacked on chip before it feeds them. PACK = 128 // tokens channels
    share a packed row, channel c at row c // PACK, lanes (c % PACK) * tokens
    on. The notes below give the Layer 5 shape's numbers (240 channels x 16
    tokens: PACK 8, 30 packed rows); Layer 4 is 192 x 64 (PACK 2, 96 packed
    rows). The seam partner is packed_pack.asm.

    ABI (kernel.toml's [abi]; cr.<name> below):
      x            the packed rows
      packed_rows  the packed row count (the loop's bound)
      row_stride   1
      y            the unpacked rows, one channel a row; each store writes
                   tokens lanes (dstruct_tok), the rest of the row is untouched
      mask         one R_MASK row: slot p enables channel p's window
      dstruct_tok  dstructure, tokens valid elements (the stores)
      dstruct_row  dstructure, 128 valid elements
    The program sets every LR it reads.

    Construction: for each packed row c (0..29), gather each of its 8
    partitions p_in=0..7 into lanes 0-15 via rc_idx=16*p_in, mask window 0
    (one-hot lanes 0-15), ACC.ADD.FIRST (single term, no accumulation across
    p_in -- each output row is independent, unlike packed layernorm's step 1
    which SUMS across p_in into one shared result). Store valid_elements=16
    to output row 8*c+p_in. Runtime loop over rows (same fused-word
    loop-counter discipline as packed_layernorm.asm -- see that
    file's header for the loop-count derivation this mirrors exactly).
========================================================================== -#}
{%- set n_ch = channels %}
{%- set n_tok = tokens %}
{%- set PACK = 128 // n_tok %}
{%- set PS = n_tok %}

{% set rc_idx_reg   = "LR0"  %}
{% set mask_off_lr  = "LR1"  %}
{% set read_ptr     = "LR2"  %}
{% set write_ptr    = "LR3"  %}
{% set row_idx      = "LR4"  %}
{% set row_limit    = "LR5"  %}
{% set row_stride   = "LR6"  %}
{% set mask_shift0  = "LR7"  %}
{% set slot0_lr     = "LR8"  %}

{% set ZERO         = "CR0"  %}
{% set ONE          = "CR1"  %}
{% set PACKED_BASE  = cr.x  %}
{% set ROW_COUNT    = cr.packed_rows  %}
{% set ROW_STRIDE   = cr.row_stride  %}
{% set UNPACKED_BASE = cr.y %}
{% set MASK_BASE    = cr.mask  %}
{% set DSTRUCT_TOK    = cr.dstruct_tok  %}
{% set DSTRUCT128   = cr.dstruct_row  %}
{#- CR9-CR15 unused. #}

{% set N_PACKED_ROWS = n_ch // PACK %}

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
    LD.MASK {{ mask_off_lr }} {{ MASK_BASE }};;

    SET {{ read_ptr }} {{ PACKED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;   # write_ptr = -1: ST.AAQ's offset is a FORWARDED read and dispatches after this word's own ADD -- see packed_layernorm.asm's step 3 note
    LD.MULT.CYCLIC {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime row 0 into slot 0

    # ---- row 0 (peeled) ----
{%- for p_in in range(PACK) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, PS * p_in) }}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    {%- if p_in < PACK - 1 %}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT_TOK }};
    ST.AAQ {{ write_ptr }} {{ UNPACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;
    {%- else %}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT_TOK }};
    ST.AAQ {{ write_ptr }} {{ UNPACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LD.MULT.CYCLIC {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- endif %}
{%- endfor %}

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1 -- see packed_layernorm.asm's loop-count derivation note

unpack_loop:
{%- for p_in in range(PACK) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, PS * p_in) }}
    {%- if p_in < PACK - 1 %}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT_TOK }};
    ST.AAQ {{ write_ptr }} {{ UNPACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;
    {%- else %}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    {#- final p_in of this row: fuse write_ptr's advance, the trailing
        prefetch load+advance, AND row_idx's own advance+BLT.S all into one
        word -- exactly 3 "lr" ops (write_ptr, read_ptr, row_idx), the
        per-cycle maximum -- mirroring packed_layernorm.asm's
        loop structure precisely (see that file for why row_idx's ADD and
        BLT.S must share the closing load's word). #}
    ACTIVATE.QUANTIZE identity {{ DSTRUCT_TOK }};
    ST.AAQ {{ write_ptr }} {{ UNPACKED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LD.MULT.CYCLIC {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT.S {{ row_idx }} {{ row_limit }} unpack_loop;;
    {%- endif %}
{%- endfor %}

end:
    HALT;;
