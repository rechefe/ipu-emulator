# Packed LayerNorm, L4 shape: 192 channels x 64 tokens, PACKED 2 channels
# per 128-lane row (channel p's 64 tokens at lanes [64p, 64p+64)), 96 packed
# rows (192/2, exact). Row r holds channels [2r, 2r+2).
#
# Layer:   L4
# Scope:   single-stream
# Layout:  packed
# Shape:   192ch x 64tok, packing factor 2
# Status:  validated
# Related: asm_packed_layernorm_240x16.asm (L5 counterpart, same algorithm shape)
# Tests:   test_packed_layernorm_192x64.py
#
# L4 port of asm_packed_layernorm_240x16.asm: same six-step algorithm shape,
# only the partition count/width change (8x16 -> 2x64, per
# partition_size(64)=64 -> parts_per_chunk=128/64=2 -- see
# docs/isa_friction_log.md's L4 entry). The broadcast formula generalizes to
# rc_idx=(-64*p) mod 512 (NOT mod 128 -- same L5-derived correction, still
# applies here since the underlying MULT.RC.* cyclic ring is 512 elements
# regardless of partition width).
#
#   Step 1 (SUM):     for chunk c=0..95, p_in=0..1: gather packed chunk c's
#                     partition p_in (64 tokens) into lanes 0-63 via
#                     rc_idx=64*p_in, mask_offset=0 (one-hot lanes 0-63),
#                     scaled by -1/192 via MULT.RC.VV against an all-lanes
#                     -1/192 row loaded into R0 (fractional constant must
#                     come from XMEM, not a CR immediate -- MULT.RC.VE's
#                     CR-scalar path truncates to a signed byte). ACC.ADD
#                     (.FIRST once, at c=0,p_in=0) accumulates all 192
#                     channels' contributions into r_acc lanes 0-63. Store
#                     (valid_elements=64) -> NEG_MEAN row.
#
#   Step 2 (BCAST1):  broadcast NEG_MEAN's 64 values into a 128-lane tile
#                     (every 64-lane window p=0..1 holds the same 64 values)
#                     via rc_idx=(-64*p) mod 512, mask_offset=p, ACC.ADD
#                     (.FIRST at p=0). Store (valid_elements=128) ->
#                     NEG_MEAN_TILE.
#
#   Step 3 (CENTER):  centered[c] = x[c] + NEG_MEAN_TILE, partition-local
#                     elementwise. NEG_MEAN_TILE is loaded ONCE into
#                     R_CYCLIC slot 128 before the loop (constant across all
#                     96 chunks); x[c] cycles through slot 0. Two
#                     MULT.RC.VE(x1) + ACC.ADD/.FIRST per chunk, one store
#                     per chunk.
#
#   Step 4 (SUMSQ):   same cross-partition-gather shape as step 1, but the
#                     source is `centered` and the per-lane op is
#                     MULT.RC.VS (square in place). Raw (unscaled) sum of
#                     squares accumulates into r_acc lanes 0-63, stored to
#                     TEMP. A separate single MULT.RC.VV by an all-lanes
#                     1/192 row + rsqrt activation turns the raw sum into
#                     1/sigma. Store -> INVSTD row.
#
#   Step 5 (BCAST2):  broadcast INVSTD the same way as step 2.
#                     -> INVSTD_TILE (128 lanes).
#
#   Step 6 (AFFINE):  output[c] = centered[c]*INVSTD_TILE*GAMMA_TILE[c] +
#                     BETA_TILE[c], partition-local elementwise.
#                     GAMMA_TILE/BETA_TILE are PRE-REPLICATED by the harness
#                     at load time (192 scalars each, tiled 64x per channel
#                     into the matching packed-row/window layout). The two
#                     multiplications use MULT.RC.VV(ra=R0/R1) (genuine
#                     tensor product), NOT two MULT.RC.VE(x1)+ACC.ADD passes
#                     (that computes A+B, not A*B -- see
#                     asm_packed_layernorm_240x16.asm's header for the bug
#                     this avoids). The final addition (+beta) stays a
#                     MULT.RC.VE(x1)+ACC.ADD pass.
#
# RUNTIME CHUNK LOOPS (not fully unrolled): same 1024-instruction ceiling
# concern as the L5 kernel (which hit ~1.9x over at 240 channels fully
# unrolled). Steps 1/3/4a/6 are genuine runtime BLT loops over the chunk
# counter; only the 2-way partition unroll WITHIN one chunk (steps 1/4a)
# stays statically unrolled.
#
# PEELED FIRST CHUNK (steps 1, 3, 4a, 6): ACC.ADD.FIRST must fire exactly
# once for the whole kernel run -- chunk 0 unrolled once outside any loop
# with .FIRST at its one true first position; a runtime loop then covers
# chunks 1..95.
#
# CR budget: 16 total (cr0-cr15), all committed -- identical layout to the
# L5 kernel's register-name block (this part of the CR budget derivation
# has no width-dependent content, only the row-count constants change).

{% set rc_idx_reg   = "lr0"  %}  {# r_cyclic base ELEMENT offset for masked gather/scatter/multiply #}
{% set mask_off_lr  = "lr1"  %}  {# mask_offset carrier / LDR_MULT_MASK_REG offset / LDR_CYCLIC_MULT_REG index carrier #}
{% set read_ptr     = "lr2"  %}  {# per-step row read pointer (offset from a ZERO base -- absolute row number) #}
{% set write_ptr    = "lr3"  %}  {# per-step row write pointer #}
{% set gb_ptr       = "lr4"  %}  {# gamma/beta row pointer in step 6, recomputed each chunk from base + running chunk offset #}
{% set row_idx      = "lr5"  %}  {# packed-row / chunk counter (runtime loop induction variable) #}
{% set row_limit    = "lr6"  %}  {# runtime loop bound, set per step #}
{% set row_stride   = "lr7"  %}  {# 1 = row stride #}
{% set mask_shift0  = "lr8"  %}  {# const 0: mask_shift (no partition-boundary shifting used) #}
{% set slot0_lr     = "lr9"  %}  {# const 0: LDR_CYCLIC_MULT_REG index for slot 0 #}
{% set slot128_lr   = "lr10" %}  {# const 128: LDR_CYCLIC_MULT_REG index for slot 128 #}

{% set ZERO            = "cr0"  %}  {# const 0 #}
{% set ONE             = "cr1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE       = "cr2"  %}  {# base row of X, 96 packed rows #}
{% set ROW_COUNT       = "cr3"  %}  {# 96 #}
{% set ROW_STRIDE      = "cr4"  %}  {# 1 #}
{% set SCRATCH64       = "cr5"  %}  {# shared 64-value scratch row (see L5 header's lifetime-audit note -- identical here) #}
{% set NEG_MEAN_TILE   = "cr6"  %}  {# NEG_MEAN_TILE row (128 lanes); also NEG_INV_N scratch pre-step-2 #}
{% set CENTERED_BASE   = "cr7"  %}  {# 96 packed rows, centered #}
{% set INVSTD_TILE     = "cr8"  %}  {# 1/sigma tile (128 lanes); also INV_N scratch pre-step-5 #}
{% set GAMMA_TILE_BASE = "cr9"  %}  {# 96 packed rows, gamma pre-replicated #}
{% set BETA_TILE_BASE  = "cr10" %}  {# 96 packed rows, beta pre-replicated #}
{% set OUTPUT_BASE     = "cr11" %}  {# 96 packed rows, output #}
{% set MASK_BASE       = "cr12" %}  {# R_MASK source row: 2 one-hot 64-lane slots #}
{% set DSTRUCT64       = "cr13" %}  {# dstructure: valid_elements=64 #}
{% set DSTRUCT128      = "cr14" %}  {# dstructure: valid_elements=128 #}
{% set ALLONES_MASK_BASE = "cr15" %}  {# R_MASK source row: all 1024 bits set #}

{% set NEG_MEAN_BASE  = SCRATCH64 %}
{% set TEMP_BASE      = SCRATCH64 %}
{% set INVSTD_BASE    = SCRATCH64 %}
{% set NEG_INV_N_BASE = NEG_MEAN_TILE %}
{% set INV_N_BASE     = INVSTD_TILE %}
{% set N_PACKED_ROWS  = 96 %}
{% set N_CH           = 192 %}
{% set PS             = 64 %}
{% set SLOT_STRIDE    = 128 %}

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
    SET {{ slot128_lr }} {{ ZERO }};;
    {{- inc(slot128_lr, SLOT_STRIDE) }}
    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ MASK_BASE }};;

# -----------------------------------------------------------------------
# Step 1: -mean[t] = sum over 192 channels of x[ch,t] * (-1/192)
# -----------------------------------------------------------------------

    LDR_MULT_REG r0 {{ mask_off_lr }} {{ NEG_INV_N_BASE }};;   # r0 <- -1/192 (every lane)

    SET {{ read_ptr }} {{ DATA_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime chunk 0 into slot 0

{%- for p_in in range(2) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, PS * p_in) }}
    {%- set is_first = (p_in == 0) %}
    {%- set is_last_p = (p_in == 1) %}
    {%- if is_last_p %}
    MULT.RC.VV {{ rc_idx_reg }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- else %}
    MULT.RC.VV {{ rc_idx_reg }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};;
    {%- endif %}
{%- endfor %}

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;   # row_limit = ROW_COUNT-1

step1_loop:
{%- for p_in in range(2) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, PS * p_in) }}
    {%- if loop.last %}
    MULT.RC.VV {{ rc_idx_reg }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} step1_loop;;
    {%- else %}
    MULT.RC.VV {{ rc_idx_reg }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    {%- endif %}
{%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT64 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ NEG_MEAN_BASE }};;

# -----------------------------------------------------------------------
# Step 2: broadcast NEG_MEAN (64 values) into a 128-lane tile, one 64-lane
# window per partition p=0..1. rc_idx = (-64*p) mod 512. Fixed 2-way
# unroll, no runtime loop needed.
# -----------------------------------------------------------------------

    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ NEG_MEAN_BASE }} {{ slot0_lr }};;

{%- for p in range(2) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - PS * p) % 512) }}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p }} {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if p == 0 else "" }};;
{%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ NEG_MEAN_TILE }};;

# -----------------------------------------------------------------------
# Step 3: centered[c] = x[c] + NEG_MEAN_TILE, partition-local elementwise.
# -----------------------------------------------------------------------

    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ ALLONES_MASK_BASE }};;
    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ NEG_MEAN_TILE }} {{ slot128_lr }};;

    SET {{ read_ptr }} {{ DATA_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;

    MULT.RC.VE {{ slot0_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    MULT.RC.VE {{ slot128_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ CENTERED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;

step3_loop:
    MULT.RC.VE {{ slot0_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    MULT.RC.VE {{ slot128_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ CENTERED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} step3_loop;;

# -----------------------------------------------------------------------
# Step 4a: raw sum of squares (unscaled) -- same gather shape as step 1,
# source = centered, op = MULT.RC.VS.
# -----------------------------------------------------------------------

    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ MASK_BASE }};;
    SET {{ read_ptr }} {{ CENTERED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;

{%- for p_in in range(2) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, PS * p_in) }}
    {%- set is_first = (p_in == 0) %}
    {%- set is_last_p = (p_in == 1) %}
    {%- if is_last_p %}
    MULT.RC.VS {{ rc_idx_reg }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    {%- else %}
    MULT.RC.VS {{ rc_idx_reg }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if is_first else "" }};;
    {%- endif %}
{%- endfor %}

    SET {{ row_idx }} {{ ZERO }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;
    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;

step4a_loop:
{%- for p_in in range(2) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, PS * p_in) }}
    {%- if loop.last %}
    MULT.RC.VS {{ rc_idx_reg }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} step4a_loop;;
    {%- else %}
    MULT.RC.VS {{ rc_idx_reg }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    {%- endif %}
{%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT64 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};;

# -----------------------------------------------------------------------
# Step 4b: 1/sigma = rsqrt((1/192) * sum_of_squares)
# -----------------------------------------------------------------------

    LDR_MULT_REG r0 {{ mask_off_lr }} {{ TEMP_BASE }};;
    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ INV_N_BASE }} {{ slot0_lr }};;
    MULT.RC.VV {{ slot0_lr }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;

    ACTIVATE.QUANTIZE rsqrt {{ DSTRUCT64 }};;
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ INVSTD_BASE }};;

# -----------------------------------------------------------------------
# Step 5: broadcast INVSTD (64 values) into a 128-lane tile, same
# construction as step 2.
# -----------------------------------------------------------------------

    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ INVSTD_BASE }} {{ slot0_lr }};;

{%- for p in range(2) %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {{- inc(rc_idx_reg, (512 - PS * p) % 512) }}
    MULT.RC.VE {{ rc_idx_reg }} {{ ONE }} {{ p }} {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD{{ ".FIRST" if p == 0 else "" }};;
{%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ INVSTD_TILE }};;

# -----------------------------------------------------------------------
# Step 6: output[c] = centered[c] * INVSTD_TILE * GAMMA_TILE[c] + BETA_TILE[c]
# -----------------------------------------------------------------------

    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ ALLONES_MASK_BASE }};;
    LDR_MULT_REG r1 {{ mask_off_lr }} {{ INVSTD_TILE }};;

    SET {{ read_ptr }} {{ CENTERED_BASE }};;
    SUB {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    SET {{ write_ptr }} {{ ZERO }};;
    SUB {{ write_ptr }} {{ write_ptr }} {{ row_stride }};;
    SET {{ row_idx }} {{ ZERO }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;

    # ---- chunk 0 (peeled): row_idx == 0, pointers land at the tile bases ----
    MULT.RC.VV {{ slot0_lr }} r1 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};
    SET {{ gb_ptr }} {{ GAMMA_TILE_BASE }};;
    ADD {{ gb_ptr }} {{ gb_ptr }} {{ row_idx }};;
    LDR_CYCLIC_MULT_REG {{ gb_ptr }} {{ ZERO }} {{ slot0_lr }};;
    LDR_MULT_REG r0 {{ mask_off_lr }} {{ TEMP_BASE }};;
    MULT.RC.VV {{ slot0_lr }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};
    SET {{ gb_ptr }} {{ BETA_TILE_BASE }};;
    ADD {{ gb_ptr }} {{ gb_ptr }} {{ row_idx }};;
    LDR_CYCLIC_MULT_REG {{ gb_ptr }} {{ ZERO }} {{ slot0_lr }};;
    MULT.RC.VE {{ slot0_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ OUTPUT_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};;

    SET {{ row_limit }} {{ ROW_COUNT }};;
    SUB {{ row_limit }} {{ row_limit }} {{ ONE }};;

step6_loop:
    MULT.RC.VV {{ slot0_lr }} r1 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};
    SET {{ gb_ptr }} {{ GAMMA_TILE_BASE }};;
    ADD {{ gb_ptr }} {{ gb_ptr }} {{ row_idx }};;
    LDR_CYCLIC_MULT_REG {{ gb_ptr }} {{ ZERO }} {{ slot0_lr }};;
    LDR_MULT_REG r0 {{ mask_off_lr }} {{ TEMP_BASE }};;
    MULT.RC.VV {{ slot0_lr }} r0 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ mask_off_lr }} {{ TEMP_BASE }};
    SET {{ gb_ptr }} {{ BETA_TILE_BASE }};;
    ADD {{ gb_ptr }} {{ gb_ptr }} {{ row_idx }};;
    LDR_CYCLIC_MULT_REG {{ gb_ptr }} {{ ZERO }} {{ slot0_lr }};;
    MULT.RC.VE {{ slot0_lr }} {{ ONE }} 0 {{ mask_shift0 }} {{ DSTRUCT128 }};
    ACC.ADD;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT128 }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ OUTPUT_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ ZERO }} {{ slot0_lr }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ row_idx }} {{ row_idx }} {{ ONE }};
    BLT {{ row_idx }} {{ row_limit }} step6_loop;;

end:
    BKPT;;
