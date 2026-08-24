# SILU VARIANT (FFN1's nonlinearity): identical to
# asm_packed_output_linear_generic_p4.asm except the final store applies
#
# Layer:   L4
# Scope:   all-stream/P4
# Layout:  packed (input, output) -- scatter-on-write construction
# Shape:   Kch->2ch x 64tok, packing factor 2 (K parameterized)
# Status:  validated
# Related: asm_packed_output_linear_generic_p4.asm (identity-activation base this varies);
#          asm_packed_output_linear_silu.asm (L5 counterpart)
# Tests:   exercised via test_full_layer_l4_packed.py
#
# ACTIVATE.QUANTIZE silu instead of identity, matching every production
# FFN1 kernel (proj_ffn1_192_p4.asm) -- GELU does not exist anywhere in this
# codebase; per the L5 session's explicit user direction (see that
# session's report), this uses silu to match production.
#
# Packed-OUTPUT linear layer, L4 shape (K, N_OUT=2 fixed per call).
#
# L4 port of asm_packed_output_linear_generic.asm: partition_size(64)=64 ->
# parts_per_chunk=128/64=2 (see docs/isa_friction_log.md's L4 entry), so
# this kernel unrolls 2 partitions per chunk, not 8, and N_OUT is fixed at
# 2 per call (not 8): with N_OUT=2 the 2 R_MASK slots map 1:1 onto p_out,
# the alignment this construction depends on (generalizing L5's "the
# alignment holds only at P8" note to "holds at P{parts_per_chunk}").
# For N_OUT>2 this kernel is called N_OUT/2 times by the harness, each call
# producing ONE packed row of 2 output channels.
#
#   OUT[o, t] = sum_k W[o, k] * X[k, t]  for o = 0..1 (this call's 2 outputs), k = 0..K-1
#
#   rc_idx = 64*(p_in - p_out) mod 512
#
# REPLICATION SLOTS -- CONTRADICTS the L5 finding that "1 slot suffices".
# Direct enumeration of all 4 (p_in, p_out) pairs at ps=64 (script-verified,
# see docs/isa_friction_log.md):
#   p_in=0 p_out=0  rc_idx=0    max_read=63   slot=[0,0]
#   p_in=0 p_out=1  rc_idx=448  max_read=511  slot=[3,3]
#   p_in=1 p_out=0  rc_idx=64   max_read=127  slot=[0,0]
#   p_in=1 p_out=1  rc_idx=0    max_read=63   slot=[0,0]
# 3 of 4 pairs stay in slot 0, but (p_in=0,p_out=1) lands ENTIRELY in slot 3
# (elements 448-511), never touching slots 1 or 2. So this kernel replicates
# the packed chunk into R_CYCLIC slots 0 and 3 ONLY (2 LDR_CYCLIC_MULT_REG
# calls per chunk) -- not L5's 1-slot optimization, and not the naive
# 4-slot default either. This is a genuine, re-derived finding, not a port.
#
# Weight layout: 2 rows (one per p_out in this call), each split across
# ceil(K/128) weight-chunks -- same convention as the L5 kernel. K assumed
# a multiple of 2 (packing group size); L4's real shapes (192, 384) both
# qualify -- chunk_widths for K=192 are [128, 64], both even.
#
# RUNTIME CHUNK LOOP / PEELED FIRST CHUNK: same discipline as
# asm_packed_output_linear_generic.asm -- see that file's header for the
# full 1024-instruction-ceiling and .FIRST-firing-once rationale, unchanged
# here except the unroll width (2 not 8) and instruction counts.
#
# SEED LOOKUP TABLE (2 CRs, one per p_out): seed = (512 - 64*p_out - 64) % 512,
# computed once by the harness at load time.
#
# CROSS-KERNEL R_MASK HAZARD: this kernel loads a one-hot 2-slot R_MASK
# (one 64-lane window per output partition) via LDR_MULT_MASK_REG and never
# restores an all-ones R_MASK before halting. A downstream kernel in the
# same IpuState that relies on R_MASK's all-ones default (e.g.
# asm_packed_residual_add_192x64.asm) will silently read back zeros in
# partition 1 unless the caller explicitly reloads R_MASK first. See
# docs/isa_friction_log.md's "Cross-kernel R_MASK state bleed" entry (L5
# instance; the same hazard applies here at L4's 2-slot width).

{% set rc_idx_reg   = "lr0" %}
{% set w_ptr        = "lr1" %}
{% set out_ptr      = "lr2" %}
{% set k_idx        = "lr3" %}
{% set mask_off_lr  = "lr4" %}
{% set slot_lr      = "lr5" %}
{% set slot3_lr     = "lr6" %}
{% set chunk_base   = "lr7" %}
{% set chunk_ctr    = "lr8" %}   {# running packed-chunk counter across the WHOLE kernel, drives chunk_base #}
{% set pc_idx       = "lr9" %}   {# packed-chunk index within the current weight-chunk's runtime loop #}
{% set pc_bound     = "lr10" %}  {# this weight-chunk's pc_idx upper bound (inclusive) #}
{% set wc_idx       = "lr11" %}  {# weight-chunk index, runtime counter #}
{% set wc_bound     = "lr12" %}  {# W_CHUNKS #}
{% set k_base       = "lr13" %}  {# 2*pc_idx: this packed chunk's channel offset WITHIN the current weight-chunk's R0 row #}
{% set w_row_ptr    = "lr14" %}  {# walks the current weight-chunk's 2 per-p_out rows, reset to w_ptr at the start of every packed chunk #}

{% set ZERO          = "cr0"  %}
{% set DATA_BASE     = "cr2"  %}
{% set WEIGHTS_BASE  = "cr3"  %}
{% set OUTPUT_BASE   = "cr4"  %}
{% set MASK_BASE     = "cr5"  %}
{% set DSTRUCT_MULT  = "cr6"  %}
{% set DSTRUCT_STORE = "cr7"  %}
{% set SEED_CR = ["cr8", "cr9"] %}
{#- SEED_CR[p_out] holds (512 - 64*p_out - 64) % 512, the pre-increment-biased
    rc_idx seed for output partition p_out (harness-loaded constant). #}

{% set SLOT3 = 384 %}

{% macro inc(dest, n) %}
{%- set n = n % 512 -%}
{%- if n != 0 -%}
{%- for chunk in ((n // 255) * [255] + ([n % 255] if n % 255 else [])) %}
    INC {{ dest }} {{ chunk }};;
{%- endfor -%}
{%- endif -%}
{% endmacro %}

{% macro replicate_chunk() %}
    SET {{ chunk_base }} {{ DATA_BASE }};;
    ADD {{ chunk_base }} {{ chunk_base }} {{ chunk_ctr }};;
    SET {{ slot_lr }} {{ ZERO }};;
    LDR_CYCLIC_MULT_REG {{ chunk_base }} {{ ZERO }} {{ slot_lr }};;
    SET {{ slot3_lr }} {{ ZERO }};;
    {{- inc(slot3_lr, SLOT3) }}
    LDR_CYCLIC_MULT_REG {{ chunk_base }} {{ ZERO }} {{ slot3_lr }};;
{% endmacro %}

{% macro accumulate_chunk(first) %}
    {#- Each output partition p_out has ITS OWN weight row. w_row_ptr walks
        the CURRENT weight-chunk's 2 rows (p_out=0..1), reset to w_ptr
        (this weight-chunk's row 0) at the start of every packed chunk. #}
    ADD {{ w_row_ptr }} {{ w_ptr }} {{ ZERO }};;
    {%- for p_out in range(2) %}
    LDR_MULT_REG r0 {{ w_row_ptr }} {{ ZERO }};;
    SET {{ rc_idx_reg }} {{ SEED_CR[p_out] }};;
    ADD {{ k_idx }} {{ k_base }} {{ ZERO }};;
    DEC {{ k_idx }} 1;;
    {%- for p_in in range(2) %}
    INC {{ rc_idx_reg }} 64;
    INC {{ k_idx }} 1;
    MULT.RC.VE {{ rc_idx_reg }} {{ k_idx }} {{ p_out }} {{ mask_off_lr }} {{ DSTRUCT_MULT }};
    ACC.ADD{{ ".FIRST" if (first and p_out == 0 and p_in == 0) else "" }};;
    {%- endfor %}
    {%- if p_out < 1 %}
    INC {{ w_row_ptr }} 1;;
    {%- endif %}
    {%- endfor %}
{% endmacro %}

{% set chunk0_width = chunk_widths[0] %}
{% set chunk0_pc_count = chunk0_width // 2 %}

    SET {{ mask_off_lr }} {{ ZERO }};;
    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ MASK_BASE }};;
    SET {{ out_ptr }} {{ OUTPUT_BASE }};;
    SET {{ w_ptr }} {{ WEIGHTS_BASE }};;
    SET {{ chunk_ctr }} {{ ZERO }};;
    SET {{ k_base }} {{ ZERO }};;

    {#- ---- peeled: weight-chunk 0's packed chunk 0, the one true .FIRST position ---- #}
    {{ replicate_chunk() }}
    {{ accumulate_chunk(true) }}
    INC {{ chunk_ctr }} 1;;
    INC {{ k_base }} 2;;

    {%- if chunk0_pc_count > 1 %}
    {#- ---- weight-chunk 0's REMAINING packed chunks (pc=1..chunk0_pc_count-1) ---- #}
    SET {{ pc_idx }} {{ ZERO }};;
    INC {{ pc_idx }} 1;;
    SET {{ pc_bound }} {{ ZERO }};;
    INC {{ pc_bound }} {{ chunk0_pc_count - 1 }};;
wc0_pc_loop:
    {{ replicate_chunk() }}
    {{ accumulate_chunk(false) }}
    INC {{ chunk_ctr }} 1;;
    INC {{ k_base }} 2;;
    BLT {{ pc_idx }} {{ pc_bound }} wc0_pc_loop_inc;;
    B wc0_pc_loop_exit;;
wc0_pc_loop_inc:
    INC {{ pc_idx }} 1;;
    B wc0_pc_loop;;
wc0_pc_loop_exit:
    {%- endif %}

    {%- if chunk_widths | length > 1 %}
    {#- ---- weight-chunks 1..W_CHUNKS-1, each in full (pc=0..bound_i) ---- #}
    SET {{ wc_idx }} {{ ZERO }};;
    INC {{ wc_idx }} 1;;
    INC {{ w_ptr }} 2;;
    {#- +2, not +1: each weight-chunk now occupies 2 rows (one per p_out). #}
    SET {{ wc_bound }} {{ ZERO }};;
    INC {{ wc_bound }} {{ chunk_widths | length }};;

wc_loop:
    SET {{ k_base }} {{ ZERO }};;
    {%- for chunk_width in chunk_widths[1:] %}
    {%- set wc_num = loop.index %}
    {%- if not loop.last %}
    SET {{ pc_bound }} {{ ZERO }};;
    INC {{ pc_bound }} {{ wc_num }};;
    BGT {{ wc_idx }} {{ pc_bound }} pc_bound_skip_{{ wc_num }};;
    {%- endif %}
    SET {{ pc_bound }} {{ ZERO }};;
    INC {{ pc_bound }} {{ (chunk_width // 2) - 1 }};;
    {%- if not loop.last %}
    B pc_bound_done;;
pc_bound_skip_{{ wc_num }}:
    {%- endif %}
    {%- endfor %}
pc_bound_done:
    SET {{ pc_idx }} {{ ZERO }};;

wc_pc_loop:
    {{ replicate_chunk() }}
    {{ accumulate_chunk(false) }}
    INC {{ chunk_ctr }} 1;;
    INC {{ k_base }} 2;;
    BLT {{ pc_idx }} {{ pc_bound }} wc_pc_loop_inc;;
    B wc_pc_loop_exit;;
wc_pc_loop_inc:
    INC {{ pc_idx }} 1;;
    B wc_pc_loop;;
wc_pc_loop_exit:

    INC {{ wc_idx }} 1;;
    BLT {{ wc_idx }} {{ wc_bound }} wc_loop_continue;;
    B wc_loop_exit;;
wc_loop_continue:
    INC {{ w_ptr }} 2;;
    B wc_loop;;
wc_loop_exit:
    {%- endif %}

    ACTIVATE.QUANTIZE silu {{ DSTRUCT_STORE }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;

end:
    BKPT;;
