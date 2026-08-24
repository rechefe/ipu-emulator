# Packed-OUTPUT linear layer, TINY validation shape: K=8 (one packed chunk),
# N_OUT=8 output channels, 16 tokens. Validates the construction proposed to
# refute/confirm "path (b) never produces packed output":
#
# Layer:   L5
# Scope:   single-stream
# Layout:  packed (input, output) -- scatter-on-write construction
# Shape:   8ch->8ch x 16tok, packing factor 8
# Status:  validated
# Related: asm_packed_output_linear_generic.asm (K-generalized successor)
# Tests:   test_packed_output_linear_tiny.py
#
#   rc_idx = 16*(p_in - p_out) mod 512
#
# lands X[8c+p_in, t] at mult_res lane 16*p_out+t for ANY p_out, masked to
# that 16-lane window via mask_offset=p_out (R_MASK slot p_out, pre-built to
# have exactly bits [16*p_out, 16*p_out+16) set). ACC.ADD (not .FIRST, except
# the very first MULT of the whole kernel) accumulates into disjoint lanes of
# ONE shared r_acc across all 8 output channels -- no r_acc reset between
# output channels. One store at the end writes a packed row of 8 outputs.
#
# REQUIRES the packed chunk replicated into all 4 R_CYCLIC slots (index in
# {0,128,256,384}) -- rc_idx wraps mod 512, but LDR_CYCLIC_MULT_REG only ever
# writes ONE 128-element slot per call, so reads that cross a slot boundary
# (any p_in != p_out) would otherwise read stale/wrong data from an
# unreplicated ring. This replication is extra load cost not present in the
# original (unpacked-output) path (b).
#
# rc_idx is built by ADD/SUB arithmetic, not a lookup table (CR budget is
# only 16 total and a full 15-entry table doesn't fit alongside base
# pointers): for fixed p_out, rc_idx as p_in runs 0..7 is 16*(0-p_out),
# 16*(1-p_out), ..., i.e. a constant +16 step per p_in -- same increment
# path (b) already uses, just seeded at a different bias per p_out. Seed
# rc_idx_reg = 16*(-p_out) mod 512 once per p_out (via repeated SUB16 from
# 0, p_out times -- p_out is compile-time/unrolled so this is a fixed
# instruction count, not a runtime loop), then ADD 16 per p_in step.
#
# Weight layout: one row per output channel (8 rows), holding raw
# (unreplicated) W[p_out, 0:8] at R0[0:8] -- same LDR_MULT_REG + R0-index
# mechanism as every other kernel here.
#
# CROSS-KERNEL R_MASK HAZARD: this kernel loads a one-hot 8-slot R_MASK
# (one 16-lane window per output partition) via LDR_MULT_MASK_REG and never
# restores an all-ones R_MASK before halting. R_MASK is process-wide
# register state, not owned or reset by any kernel -- a downstream kernel
# in the same IpuState that relies on R_MASK's all-ones default (e.g.
# asm_packed_residual_add_240x16.asm) will silently read back zeros in
# partitions 1-7 unless the caller explicitly reloads R_MASK first. See
# docs/isa_friction_log.md's "Cross-kernel R_MASK state bleed" entry.

{% set rc_idx_reg   = "lr0" %}
{% set w_ptr        = "lr1" %}
{% set out_ptr      = "lr2" %}
{% set k_idx        = "lr3" %}  {# p_in, 0..7, built the same way path (b) builds it: pre-incremented ADD chain #}
{% set mask_off_lr  = "lr4" %}  {# always 0: LDR_MULT_MASK_REG's offset operand (LrIdx-only) #}
{% set slot_lr      = "lr5" %}  {# LDR_CYCLIC_MULT_REG's index operand (LrIdx-only) -- 0/128/256/384 in turn #}

{% set ZERO          = "cr0"  %}
{% set ONE           = "cr1"  %}
{% set DATA_BASE     = "cr2"  %}
{% set WEIGHTS_BASE  = "cr3"  %}
{% set OUTPUT_BASE   = "cr4"  %}
{% set SIXTEEN       = "cr5"  %}
{% set NEG_SIXTEEN   = "cr6"  %}
{% set NEG_ONE       = "cr7"  %}
{% set MASK_BASE     = "cr8"  %}
{% set SLOT_128      = "cr9"  %}
{% set SLOT_256      = "cr10" %}
{% set SLOT_384      = "cr11" %}
{% set DSTRUCT_MULT  = "cr14" %}
{% set DSTRUCT_STORE = "cr15" %}

    SET {{ mask_off_lr }} {{ ZERO }};;
    SET {{ slot_lr }} {{ ZERO }};;
    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ DATA_BASE }} {{ slot_lr }};;
    SET {{ slot_lr }} {{ SLOT_128 }};;
    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ DATA_BASE }} {{ slot_lr }};;
    SET {{ slot_lr }} {{ SLOT_256 }};;
    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ DATA_BASE }} {{ slot_lr }};;
    SET {{ slot_lr }} {{ SLOT_384 }};;
    LDR_CYCLIC_MULT_REG {{ mask_off_lr }} {{ DATA_BASE }} {{ slot_lr }};;
    LDR_MULT_MASK_REG {{ mask_off_lr }} {{ MASK_BASE }};;

    SET {{ w_ptr }} {{ WEIGHTS_BASE }};;
    SET {{ out_ptr }} {{ OUTPUT_BASE }};;

    {%- for p_out in range(8) %}
    LDR_MULT_REG r0 {{ w_ptr }} {{ ZERO }};;
    {#- IPC-fix-style pre-increment bias (see asm_packed_linear_240to8_masked.asm's
        IPC FIX comment): the MULT+advance-ADDs below are co-issued in ONE bundle,
        and LR-slot writes (_dispatch_lr_slots) dispatch BEFORE mult within a
        bundle, so a same-bundle ADD is visible to that SAME bundle's MULT --
        meaning the register must be seeded ONE STEP BEHIND the value the first
        MULT should see, exactly like data_ptr/k_idx elsewhere in this codebase.
        seed = (16*(0-p_out) - 16) mod 512 = (512 - 16*p_out - 16) mod 512;
        first bundle's co-issued ADD (+16) brings it to the correct p_in=0 value
        before that same bundle's MULT reads it live. #}
    {%- set seed = (512 - 16 * p_out - 16) % 512 %}
    SET {{ rc_idx_reg }} {{ ZERO }};;
    {%- if seed != 0 %}
    {%- for _ in range(seed // 16) %}
    ADD {{ rc_idx_reg }} {{ rc_idx_reg }} {{ SIXTEEN }};;
    {%- endfor %}
    {%- endif %}
    SET {{ k_idx }} {{ NEG_ONE }};;
    {%- for p_in in range(8) %}
    ADD {{ rc_idx_reg }} {{ rc_idx_reg }} {{ SIXTEEN }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    MULT.RC.VE {{ rc_idx_reg }} {{ k_idx }} {{ p_out }} {{ mask_off_lr }} {{ DSTRUCT_MULT }};
    ACC.ADD{{ ".FIRST" if p_out == 0 and p_in == 0 else "" }};;
    {%- endfor %}
    {%- if p_out < 7 %}
    ADD {{ w_ptr }} {{ w_ptr }} {{ ONE }};;
    {%- endif %}
    {%- endfor %}

    ACTIVATE.QUANTIZE identity {{ DSTRUCT_STORE }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;

end:
    BKPT;;
