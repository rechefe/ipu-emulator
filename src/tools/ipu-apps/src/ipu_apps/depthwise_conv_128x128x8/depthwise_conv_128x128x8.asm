# Multi-channel depthwise convolution: 8 channels, 128x128, 3x3 kernel.
#
# Each channel has its own 3x3 kernel (depthwise = no cross-channel mixing).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 rows x 8 ch x 128 bytes = 131072 bytes)
#   cr1 = kernel base  (8 ch x 9 bytes = 72 bytes, padded to 128)
#   cr2 = output base  (128 rows x 8 ch x 512 bytes = 524288 bytes)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Input layout (interleaved by row):
#   row r, channel ch, col c: offset = (r * 8 + ch) * 128 + c
#   Row stride = 8 * 128 = 1024 bytes
#   Channel stride = 128 bytes
#
# Output layout (interleaved by row, 32-bit accumulators):
#   row r, channel ch: offset = (r * 8 + ch) * 512
#
# Kernel layout in r0[0..71]:
#   For channel ch (0..7):
#     r0[ch*9 + 0] = k(-1,-1)   r0[ch*9 + 1] = k(-1, 0)   r0[ch*9 + 2] = k(-1,+1)
#     r0[ch*9 + 3] = k( 0,-1)   r0[ch*9 + 4] = k( 0, 0)   r0[ch*9 + 5] = k( 0,+1)
#     r0[ch*9 + 6] = k(+1,-1)   r0[ch*9 + 7] = k(+1, 0)   r0[ch*9 + 8] = k(+1,+1)
#
# Mask slots in r_mask:
#   slot 0: all zeros       -> no masking        (dc = 0)
#   slot 1: bit 0 set       -> zero position 0   (dc = -1, left border)
#   slot 2: bit 127 set     -> zero position 127 (dc = +1, right border)
#
# Register allocation:
#   lr0  = current row input base offset (row * 1024)
#   lr1  = 127  (row loop end: loop while row_counter < 127)
#   lr2  = output write offset
#   lr3  = kernel byte index (0-71, auto-incrementing across channels)
#   lr4  = temp (row + channel input offset for current row)
#   lr5  = 1024 (input row stride = 8 * 128; also channel loop end)
#   lr6  = row counter (main loop)
#   lr7  = 511  (cyclic offset for dc=-1)
#   lr8  = 0    (cyclic offset dc=0, mask slot 0, mask_shift, cyclic index)
#   lr9  = 1    (cyclic offset dc=+1, mask slot 1)
#   lr10 = 2    (mask slot 2)
#   lr11 = channel offset within row (0, 128, 256, ..., 896)
#   lr12 = temp (computed input offsets)
#   lr13-lr15 = temps

# ===========================================================================
# Initialization
# ===========================================================================

    ldr_mult_reg        r0 lr8 cr1;
    set                 lr5 1024;
    set                 lr7 511;;

    ldr_mult_mask_reg   lr8 cr3;
    set                 lr9 1;;

    set                 lr10 2;
    set                 lr1 127;;

# ===========================================================================
# Row 0 (top border) — kernel rows 1 and 2 only (no row -1)
# ===========================================================================

    set                 lr11 0;
    set                 lr3 3;;
    set                 lr2 0;;

row0_ch_loop:
    reset_acc;
    add                 lr4 lr8 lr11;;

    # Load row 0, this channel into r_cyclic
    ldr_cyclic_mult_reg lr4 cr0 lr8;;

    # Kernel row 1: taps 3, 4, 5 (center row)
    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Load row 1, this channel into r_cyclic
    add                 lr12 lr4 lr5;
    ldr_cyclic_mult_reg lr12 cr0 lr8;
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

    incr                lr11 128;
    incr                lr2 512;;

    blt                 lr11 lr5 row0_ch_loop;;

# ===========================================================================
# Rows 1–126 (main loop) — all 3 kernel rows, 8 channels each
# ===========================================================================

    set                 lr0 1024;
    set                 lr6 1;;

row_loop:
    set                 lr11 0;
    set                 lr3 0;;

ch_loop:
    reset_acc;
    add                 lr4 lr0 lr11;;

    # --- Kernel row 0: taps 0-2 applied to row(r-1) ---
    sub                 lr12 lr4 lr5;
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
    ldr_cyclic_mult_reg lr4 cr0 lr8;
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
    add                 lr12 lr4 lr5;
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

    # Store and advance to next channel
    str_acc_reg         lr2 cr2;
    incr                lr3 1;;

    incr                lr11 128;
    incr                lr2 512;;

    blt                 lr11 lr5 ch_loop;;

    # Advance to next row
    incr                lr0 1024;
    incr                lr6 1;;

    blt                 lr6 lr1 row_loop;;

# ===========================================================================
# Row 127 (bottom border) — kernel rows 0 and 1 only (no row 128)
# ===========================================================================

    set                 lr11 0;
    set                 lr3 0;;

row127_ch_loop:
    reset_acc;
    add                 lr4 lr0 lr11;;

    # --- Kernel row 0: taps 0-2 applied to row 126 ---
    sub                 lr12 lr4 lr5;
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
    ldr_cyclic_mult_reg lr4 cr0 lr8;
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

    incr                lr11 128;
    incr                lr2 512;;

    blt                 lr11 lr5 row127_ch_loop;;

end:
    bkpt;;
