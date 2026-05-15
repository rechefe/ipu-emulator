# Stride-2 Depthwise 3x3 Convolution — Small Spatial (cols <= 64)
#
# Jinja template parameterized by `cols` (32 or 64).
# Harness injects a "set cols = XX" Jinja directive before assembly.
#
# Input:  rows x cols x C  (INT8)
# Output: (rows/2) x (cols/2) x C  (INT8, quantized)
# Output is in the same interleaved chunk format as input.
#
# Each 128-byte input chunk has (128/cols) packed spatial rows.
# After conv + acc.stride(cols, on, on_inv, offset), each chunk yields
# 32 output elements.  4 input chunks fill r_acc (128 elements) -> aaq -> store.
#
# Flow per channel:
#   1. Conv all 4 positions -> aaq -> store to temp[0..3] (4 x 128 bytes)
#   2. Reload each temp, identity mult, stride into r_acc at offset 0/1/2/3
#   3. Final aaq -> store output chunk
#
# Constraint: num_input_chunks >= 8 (i.e. rows*cols >= 1024), so >= 2 groups.
#
# CR registers (set by harness):
#   cr0  = input base address
#   cr1  = kernel base address
#   cr2  = output base address
#   cr3  = mask base address
#   cr4  = cols
#   cr5  = num_input_chunks (per channel)
#   cr6  = group_stride (channels * 128)
#   cr7  = 1024 (channel group size = 8 * 128)
#   cr8  = temp region base address (512 bytes: 4 x 128)
#   cr9  = 1 (identity scalar for mult.ve.cr)
#   cr10 = zero region base address (128 bytes of zeros)
#   cr11 = num_groups - 1 (last group index)
#   cr12 = 128 (step constant for add)
#
# LR registers:
#   lr0  = 0     (zero, mask slot 0, mask_shift, S0 cyclic index, stride offset 0)
#   lr1  = 1     (mask slot 1 = left border, kc offset, stride offset 1)
#   lr2  = 2     (mask slot 2 = right border, stride offset 2)
#   lr3  = 128 - cols  (kr=-1 cyclic base)
#   lr4  = 128   (kr=0 cyclic base, S1 cyclic index, channel stride)
#   lr5  = 128 + cols  (kr=+1 cyclic base)
#   lr6  = kernel byte index (0..71)
#   lr7  = output pointer
#   lr8  = group base address (group_index * 4 * group_stride)
#   lr9  = group counter
#   lr10 = channel offset (0, 128, ..., group_stride-128)
#   lr11 = channel group limit (lr10 + 1024)
#   lr12 = kernel memory offset
#   lr13 = group_stride (copy of cr6)
#   lr14 = temp
#   lr15 = temp / stride offset 3 / group limit

# =========================================================================
# Jinja macros
# =========================================================================

{# --- 9 taps of 3x3 depthwise convolution --- #}
{% macro nine_taps() %}
    # --- kr=-1: cyclic base lr3, masks 1/0/2 ---
    sub                 lr14 lr3 lr1;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    add                 lr6 lr6 1;
    mult.ve.cyclic      lr3 0 lr0 lr6;
    acc;;

    add                 lr6 lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;

    # --- kr=0: cyclic base lr4, masks 1/0/2 ---
    add                 lr6 lr6 1;
    sub                 lr14 lr4 lr1;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    add                 lr6 lr6 1;
    mult.ve.cyclic      lr4 0 lr0 lr6;
    acc;;

    add                 lr6 lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;

    # --- kr=+1: cyclic base lr5, masks 1/0/2 ---
    add                 lr6 lr6 1;
    sub                 lr14 lr5 lr1;
    mult.ve.cyclic      lr14 1 lr0 lr6;
    acc;;

    add                 lr6 lr6 1;
    mult.ve.cyclic      lr5 0 lr0 lr6;
    acc;;

    add                 lr6 lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve.cyclic      lr14 2 lr0 lr6;
    acc;;
{% endmacro %}

{# --- Load S0/S1/S2 into cyclic register and reset_acc --- #}
{# lr14 must hold S1 memory address (offset from cr0) on entry. #}
{# border: "top" = S0 from zeros, "bottom" = S2 from zeros, "normal" = all from memory #}
{% macro load_s0_s1_s2(border="normal") %}
{% if border == "top" %}
    # Top border: S0 = zeros, load S1 and S2 from memory
    ldr_cyclic_mult_reg lr0 cr10 lr0;;

    ldr_cyclic_mult_reg lr14 cr0 lr4;;

    add                 lr15 lr14 lr13;
    add                 lr14 lr4 lr4;;

    ldr_cyclic_mult_reg lr15 cr0 lr14;
    reset_acc;;
{% elif border == "bottom" %}
    # Bottom border: S0 and S1 from memory, S2 = zeros
    sub                 lr15 lr14 lr13;
    ldr_cyclic_mult_reg lr15 cr0 lr0;;

    ldr_cyclic_mult_reg lr14 cr0 lr4;
    add                 lr15 lr4 lr4;;

    ldr_cyclic_mult_reg lr0 cr10 lr15;
    reset_acc;;
{% else %}
    # Normal: all from memory
    sub                 lr15 lr14 lr13;
    ldr_cyclic_mult_reg lr15 cr0 lr0;;

    ldr_cyclic_mult_reg lr14 cr0 lr4;;

    add                 lr15 lr14 lr13;
    add                 lr14 lr4 lr4;;

    ldr_cyclic_mult_reg lr15 cr0 lr14;
    reset_acc;;
{% endif %}
{% endmacro %}

{# --- Stride phase: reload 4 temp areas and decimate into r_acc --- #}
{# cols_param: spatial width (passed explicitly) #}
{% macro stride_phase(cols_param) %}
    # Reload temp[0] -> stride offset 0
    ldr_cyclic_mult_reg lr0 cr8 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;
    reset_acc;;

    acc.stride          {{ cols_param }} on on_inv lr0;;

    # Reload temp[1] -> stride offset 1
    ldr_cyclic_mult_reg lr4 cr8 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;
    acc.stride          {{ cols_param }} on on_inv lr1;;

    # Reload temp[2] -> stride offset 2
    add                 lr14 lr4 lr4;;

    ldr_cyclic_mult_reg lr14 cr8 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;
    acc.stride          {{ cols_param }} on on_inv lr2;;

    # Reload temp[3] -> stride offset 3
    add                 lr14 lr14 lr4;;
    add                 lr15 lr0 3;;

    ldr_cyclic_mult_reg lr14 cr8 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;
    acc.stride          {{ cols_param }} on on_inv lr15;;
{% endmacro %}

# =========================================================================
# Initialization
# =========================================================================

    SET                 lr0 cr13;
    ldr_mult_mask_reg   lr0 cr3;;

    add                 lr4 lr0 cr12;
    SET                 lr1 cr14;;

    add                 lr2 lr0 2;
    sub                 lr3 lr4 cr4;;

    add                 lr5 lr4 cr4;
    add                 lr13 lr0 cr6;;

    SET                 lr7 cr13;
    SET                 lr8 cr13;;

    SET                 lr9 cr13;;

# =========================================================================
# Group 0: position 0 = top border, positions 1-3 = normal
# =========================================================================

    SET                 lr10 cr13;
    SET                 lr12 cr13;;

g0_kg_loop:
    ldr_mult_reg        r0 lr12 cr1;
    SET                 lr6 cr13;;

    add                 lr11 lr10 cr7;;

g0_ch_loop:

    # --- Position 0: top border -> conv -> store to temp+0 ---
    add                 lr14 lr8 lr10;;
{{ load_s0_s1_s2(border="top") }}
{{ nine_taps() }}
    aaq;;
    xmem.store_aaq_result lr0 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 1: normal -> conv -> store to temp+128 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    aaq;;
    xmem.store_aaq_result lr4 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 2: normal -> conv -> store to temp+256 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    add                 lr14 lr4 lr4;;
    aaq;;
    xmem.store_aaq_result lr14 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 3: normal -> conv -> store to temp+384 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    add                 lr14 lr4 lr4;;
    add                 lr14 lr14 lr4;;
    aaq;;
    xmem.store_aaq_result lr14 cr8;;

    # --- Stride phase: reload all 4 temps and decimate into r_acc ---
{{ stride_phase(cols) }}

    # Output packed result
    aaq;;
    xmem.store_aaq_result lr7 cr2;;

    # Advance channel
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;;

    add                 lr7 lr7 cr12;;

    blt                 lr10 lr11 g0_ch_loop;;

    # Advance kernel group
    add                 lr12 lr12 cr12;;

    blt                 lr10 lr13 g0_kg_loop;;

    # Advance to next group
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;

    add                 lr9 lr9 1;;

    # Check if middle groups exist (skip if only 2 groups)
    add                 lr15 lr0 cr11;;

    blt                 lr9 lr15 mid_group_start;;

    b                   last_group;;

# =========================================================================
# Middle groups: all 4 positions normal
# =========================================================================

mid_group_start:
    SET                 lr10 cr13;
    SET                 lr12 cr13;;

mid_kg_loop:
    ldr_mult_reg        r0 lr12 cr1;
    SET                 lr6 cr13;;

    add                 lr11 lr10 cr7;;

mid_ch_loop:

    # --- Position 0: normal -> conv -> store to temp+0 ---
    add                 lr14 lr8 lr10;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    aaq;;
    xmem.store_aaq_result lr0 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 1: normal -> conv -> store to temp+128 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    aaq;;
    xmem.store_aaq_result lr4 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 2: normal -> conv -> store to temp+256 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    add                 lr14 lr4 lr4;;
    aaq;;
    xmem.store_aaq_result lr14 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 3: normal -> conv -> store to temp+384 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    add                 lr14 lr4 lr4;;
    add                 lr14 lr14 lr4;;
    aaq;;
    xmem.store_aaq_result lr14 cr8;;

    # --- Stride phase ---
{{ stride_phase(cols) }}

    # Output packed result
    aaq;;
    xmem.store_aaq_result lr7 cr2;;

    # Advance channel
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;;

    add                 lr7 lr7 cr12;;

    blt                 lr10 lr11 mid_ch_loop;;

    # Advance kernel group
    add                 lr12 lr12 cr12;;

    blt                 lr10 lr13 mid_kg_loop;;

    # Advance to next group
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;

    add                 lr9 lr9 1;;

    # Continue middle if not at last group yet
    add                 lr15 lr0 cr11;;

    blt                 lr9 lr15 mid_group_start;;

# =========================================================================
# Last group: positions 0-2 normal, position 3 = bottom border
# =========================================================================

last_group:
    SET                 lr10 cr13;
    SET                 lr12 cr13;;

last_kg_loop:
    ldr_mult_reg        r0 lr12 cr1;
    SET                 lr6 cr13;;

    add                 lr11 lr10 cr7;;

last_ch_loop:

    # --- Position 0: normal -> conv -> store to temp+0 ---
    add                 lr14 lr8 lr10;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    aaq;;
    xmem.store_aaq_result lr0 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 1: normal -> conv -> store to temp+128 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    aaq;;
    xmem.store_aaq_result lr4 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 2: normal -> conv -> store to temp+256 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="normal") }}
{{ nine_taps() }}
    add                 lr14 lr4 lr4;;
    aaq;;
    xmem.store_aaq_result lr14 cr8;;

    sub                 lr6 lr6 8;;

    # --- Position 3: bottom border -> conv -> store to temp+384 ---
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
{{ load_s0_s1_s2(border="bottom") }}
{{ nine_taps() }}
    add                 lr14 lr4 lr4;;
    add                 lr14 lr14 lr4;;
    aaq;;
    xmem.store_aaq_result lr14 cr8;;

    # --- Stride phase ---
{{ stride_phase(cols) }}

    # Output packed result
    aaq;;
    xmem.store_aaq_result lr7 cr2;;

    # Advance channel
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;;

    add                 lr7 lr7 cr12;;

    blt                 lr10 lr11 last_ch_loop;;

    # Advance kernel group
    add                 lr12 lr12 cr12;;

    blt                 lr10 lr13 last_kg_loop;;

end:
    bkpt;;
