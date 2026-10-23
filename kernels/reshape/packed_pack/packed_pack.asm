{#- ==========================================================================
    packed_pack.asm -- one channel per row -> the MobileViT packed layout, FP32.
    The template of packed_pack_{192x64,240x16} (../registry.toml, E07-23).

    PACK = 128 // tokens channels share a packed row: channel c's tokens land
    at row c // PACK, lanes (c % PACK) * tokens on. The notes below give the
    Layer 5 shape's numbers (240 channels x 16 tokens: PACK 8, 30 packed rows);
    Layer 4 is 192 x 64 (PACK 2, 96 packed rows). The seam partner is
    packed_unpack.asm.

    ABI (kernel.toml's [abi]; cr.<name> below):
      x            the unpacked rows, one channel a row, tokens lanes valid
      packed_rows  the packed row count (the outer loop's bound)
      row_stride   1
      y            the packed rows
      mask         one R_MASK row: slot p enables channel p's window
      dstruct_row  dstructure, 128 valid elements (the stores)
    The program sets every LR it reads. Its last prefetch reads the row after
    x; the value is never used.

    Reverse of packed_unpack.asm, for
    the same attention-seam boundary (an unpacked producer -- attn·V,
    read-only, not modified -- feeding a packed consumer downstream).

    Construction: for each packed row c (0..29), for p_out=0..7: load
    UNPACKED_BASE row (8*c+p_out) into R_CYCLIC slot 0 (that row's 16 valid
    tokens sit at lanes 0-15), then a masked SCATTER write --
    rc_idx=(-16*p_out) mod 512, mask window p_out -- lands those same 16
    values into window p_out of the shared r_acc (same construction as
    packed_layernorm.asm's step 2/5 broadcast, generalized: there
    the SAME 16 source values feed every window; here each window's SOURCE
    is a DIFFERENT row, loaded fresh before each window's own write).
    ACC.ADD.FIRST at p_out==0 (fired at the START OF EVERY ROW, not just
    once for the whole kernel run) then ACC.ADD for p_out=1..7 accumulates
    into disjoint 16-lane windows of one shared r_acc -- no reset between the
    8 source rows of one packed row, but a genuine reset at the start of
    each new packed row. An earlier draft used unconditional ACC.ADD at
    p_out==0 (copying the "loop body always uses ACC.ADD, only the outer peel
    uses .FIRST" shape from packed_layernorm.asm's step 1, where
    the loop genuinely sums ACROSS rows) -- wrong here since this kernel's
    rows are independent, not cross-row accumulations; that bug let row
    c's first window silently add onto row (c-1)'s stale r_acc content,
    compounding across rows. Fixed by using ACC.ADD.FIRST at p_out==0 in
    both the peeled first row and the runtime loop body. One store per row
    (valid_elements=128, the whole packed row).

    PIPELINING: every p_out's own MULT consumes the row PREFETCHED by the
    PREVIOUS iteration (or the kernel's own priming load, for the very
    first). Every iteration therefore ALSO issues its own trailing
    prefetch load for the FOLLOWING row -- including p_out=7, which
    prefetches the NEXT packed row's p_out=0 row -- the absolute last
    iteration of the whole kernel (row 29, p_out=7) included: its prefetch
    reads the row after x, whose value is never used (the loop body is the
    same on every pass; see the note at that load below). An
    earlier draft omitted p_out=7's load unconditionally (treating it like
    "the last unroll slot needs no prefetch," true only for the standalone
    8-wide gather primitive this was adapted from, which reads ONE shared
    row 8 times rather than 8 DIFFERENT rows once each) -- that starves
    every row after the first of 1 of its 8 needed loads (7 issued
    instead of 8 per row, 210 total vs 240 needed), caught by desk-check
    before assembling: counting required loads per row against the
    unrolled word structure, not by a failing test.
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

{% set ZERO          = "CR0"  %}
{% set ONE           = "CR1"  %}
{% set UNPACKED_BASE = cr.x  %}
{% set ROW_COUNT     = cr.packed_rows  %}
{% set ROW_STRIDE    = cr.row_stride  %}
{% set PACKED_BASE   = cr.y  %}
{% set MASK_BASE     = cr.mask  %}
{% set DSTRUCT128    = cr.dstruct_row  %}
{#- CR8-CR15 unused. DSTRUCT16 is not needed: every load here reads a
    one-channel-per-row source (native 128-lane width; only 16 lanes are
    "valid" data but the MULT reads all 128 regardless of quantize
    windows), and the only STORE is the packed 128-lane output. #}

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

    SET {{ read_ptr }} {{ UNPACKED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;   # write_ptr = -1: ACTIVATE.EXACT's offset is a FORWARDED read and dispatches after this word's own ADD -- see packed_layernorm.asm's step 3 note
    LD.MULT.CYCLIC {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime unpacked row 0 into slot 0

    # ---- row 0 (peeled) ----
{%- for p_out in range(PACK) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - PS * p_out) % 512) }}
    {%- set is_first = (p_out == 0) %}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p_out }} {{ mask_shift0 }} {{ DSTRUCT128 }}.L {{ cr.qf }}.L;
    ACC.ADD{{ ".FIRST" if is_first else "" }};
    LD.MULT.CYCLIC {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
{%- endfor %}
    ACTIVATE.EXACT identity {{ DSTRUCT128 }}.L {{ PACKED_BASE }} {{ write_ptr }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1 -- see packed_layernorm.asm's loop-count derivation note

pack_loop:
{%- for p_out in range(PACK) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - PS * p_out) % 512) }}
    {%- if p_out < PACK - 1 %}
    {#- p_out==0 must be ACC.ADD.FIRST -- it starts a NEW packed row's
        accumulation, not a continuation of the PREVIOUS row's r_acc.
        An earlier draft used unconditional ACC.ADD here (copying the
        loop-body-vs-peel split from kernels where the loop's first
        window genuinely continues an existing sum, e.g.
        packed_layernorm.asm's step1, which sums ACROSS
        rows) -- but this kernel's rows are INDEPENDENT rows, each
        needing its own fresh r_acc, so omitting .FIRST let each new
        row's first window silently add onto the PREVIOUS row's
        stale r_acc content, compounding across rows (row0 correct,
        row1 off by ~3, row9 off by ~13 -- a growing error, the
        signature of unbounded cross-iteration accumulation rather than
        a one-off indexing mistake). #}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p_out }} {{ mask_shift0 }} {{ DSTRUCT128 }}.L {{ cr.qf }}.L;
    ACC.ADD{{ ".FIRST" if p_out == 0 else "" }};
    LD.MULT.CYCLIC {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- else %}
    {#- p_out=7's load+row_idx/BLT.S all fuse into ONE word: write_ptr,
        read_ptr, row_idx = exactly 3 "lr" ops (the per-cycle maximum),
        alongside the store (different slot) and this load (different
        slot again) -- mirrors packed_layernorm.asm's identical
        fused closing-word structure. This load prefetches the NEXT
        packed row's p_out=0 row; on the LAST loop pass (processing
        row 29) it harmlessly reads one row past the last valid input
        (row 240) -- the same harmless-extra-prefetch pattern
        layernorm.asm (layernorm_16x240)'s own step1 loop has (see that kernel's
        241-loads-for-240-channels count, verified empirically): the
        load's result is never consumed by another MULT, since the loop
        exits right after. #}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p_out }} {{ mask_shift0 }} {{ DSTRUCT128 }}.L {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- endif %}
{%- endfor %}
    ACTIVATE.EXACT identity {{ DSTRUCT128 }}.L {{ PACKED_BASE }} {{ write_ptr }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT.S {{ row_idx }} {{ row_limit }} pack_loop;;

end:
    HALT;;
