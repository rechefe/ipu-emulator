# Standard 3x3 convolution: 1 input channel -> 8 output channels, 128x128.
#
# Each output channel has its own 3x3 kernel applied to the single input.
# (Cross-channel mixing: all outputs read from the same input.)
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 rows x 128 bytes = 16384 bytes)
#   cr1 = kernel base  (8 filters x 9 bytes = 72 bytes, padded to 128)
#   cr2 = output base  (128 rows x 8 ch x 512 bytes = 524288 bytes)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Input layout (single channel):
#   row r, col c: offset = r * 128 + c
#   Row stride = 128 bytes
#
# Output layout (interleaved by row, 32-bit accumulators):
#   row r, filter f: offset = (r * 8 + f) * 512
#
# Kernel layout in r0[0..71]:
#   For filter f (0..7):
#     r0[f*9 + 0] = k(-1,-1)   r0[f*9 + 1] = k(-1, 0)   r0[f*9 + 2] = k(-1,+1)
#     r0[f*9 + 3] = k( 0,-1)   r0[f*9 + 4] = k( 0, 0)   r0[f*9 + 5] = k( 0,+1)
#     r0[f*9 + 6] = k(+1,-1)   r0[f*9 + 7] = k(+1, 0)   r0[f*9 + 8] = k(+1,+1)
#
# Key difference from depthwise: all output channels read the SAME input.
#
# Mask slots in r_mask:
#   slot 0: all zeros       -> no masking        (dc = 0)
#   slot 1: bit 0 set       -> zero position 0   (dc = -1, left border)
#   slot 2: bit 127 set     -> zero position 127 (dc = +1, right border)
#
# Register allocation:
#   lr0  = current row input base offset (row * 128)
#   lr1  = 127  (row loop end: loop while row_counter < 127)
#   lr2  = output write offset
#   lr3  = kernel byte index (0-71, auto-incrementing across filters)
#   lr4  = temp
#   lr5  = 128  (input row stride)
#   lr6  = row counter (main loop)
#   lr7  = 511  (cyclic offset for dc=-1)
#   lr8  = 0    (cyclic offset dc=0, mask slot 0, mask_shift, cyclic index)
#   lr9  = 1    (cyclic offset dc=+1, mask slot 1)
#   lr10 = 2    (mask slot 2)
#   lr11 = output channel counter (0..7)
#   lr12 = temp (computed input offsets)
#   lr13 = 8    (channel loop end)

# ===========================================================================
# Initialization
# ===========================================================================

    ldr_mult_reg        r0 lr8 cr1;
    set                 lr5 128;
    set                 lr7 511;;

    ldr_mult_mask_reg   lr8 cr3;
    set                 lr9 1;;

    set                 lr10 2;
    set                 lr1 127;;

    set                 lr13 8;;

# ===========================================================================
# Row 0 (top border) — kernel rows 1 and 2 only (no row -1)
# ===========================================================================

    set                 lr11 0;
    set                 lr3 3;;
    set                 lr2 0;;

row0_ch_loop:
    reset_acc;;

    # Load row 0 into r_cyclic (same input for all output channels)
    ldr_cyclic_mult_reg lr8 cr0 lr8;;

    # Kernel row 1: taps 3, 4, 5 (center row)
    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Load row 1 into r_cyclic
    ldr_cyclic_mult_reg lr5 cr0 lr8;
    incr                lr3 1;;

    # Kernel row 2: taps 6, 7, 8 (bottom row)
    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store output, advance kernel index (+4 = skip 3 border taps + 1)
    str_acc_reg         lr2 cr2;
    incr                lr3 4;;

    incr                lr11 1;
    incr                lr2 512;;

    blt                 lr11 lr13 row0_ch_loop;;

# ===========================================================================
# Rows 1–126 (main loop) — all 3 kernel rows, 8 output channels each
# ===========================================================================

    set                 lr0 128;
    set                 lr6 1;;

row_loop:
    set                 lr11 0;
    set                 lr3 0;;

ch_loop:
    reset_acc;;

    # --- Kernel row 0: taps 0-2 applied to row(r-1) ---
    sub                 lr12 lr0 lr5;
    ldr_cyclic_mult_reg lr12 cr0 lr8;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 1: taps 3-5 applied to row(r) ---
    ldr_cyclic_mult_reg lr0 cr0 lr8;
    incr                lr3 1;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 2: taps 6-8 applied to row(r+1) ---
    add                 lr12 lr0 lr5;
    ldr_cyclic_mult_reg lr12 cr0 lr8;
    incr                lr3 1;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store and advance to next output channel
    str_acc_reg         lr2 cr2;
    incr                lr3 1;;

    incr                lr11 1;
    incr                lr2 512;;

    blt                 lr11 lr13 ch_loop;;

    # Advance to next row
    incr                lr0 128;
    incr                lr6 1;;

    blt                 lr6 lr1 row_loop;;

# ===========================================================================
# Row 127 (bottom border) — kernel rows 0 and 1 only (no row 128)
# ===========================================================================

    set                 lr11 0;
    set                 lr3 0;;

row127_ch_loop:
    reset_acc;;

    # --- Kernel row 0: taps 0-2 applied to row 126 ---
    sub                 lr12 lr0 lr5;
    ldr_cyclic_mult_reg lr12 cr0 lr8;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 1: taps 3-5 applied to row 127 ---
    ldr_cyclic_mult_reg lr0 cr0 lr8;
    incr                lr3 1;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store and advance (+4 = skip 3 missing taps + 1)
    str_acc_reg         lr2 cr2;
    incr                lr3 4;;

    incr                lr11 1;
    incr                lr2 512;;

    blt                 lr11 lr13 row127_ch_loop;;

end:
    bkpt;;
