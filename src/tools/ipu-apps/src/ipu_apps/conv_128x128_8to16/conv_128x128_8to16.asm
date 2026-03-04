# Standard 3x3 convolution: 8 input channels -> 16 output channels, 128x128.
#
# Each output channel has 8 input-channel 3x3 kernels (full cross-channel mixing).
# Kernel per filter: 8 * 9 = 72 bytes, fits in r0 (128 bytes).
# r0 is reloaded per output filter (72 bytes/filter, padded to 128, stored at
# 128-byte XMEM boundaries).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 rows x 8 ch x 128 bytes = 131072 bytes)
#   cr1 = kernel base  (16 filters x 128 bytes each = 2048 bytes)
#                       Each filter: 8 in_ch x 9 bytes = 72 bytes, padded to 128
#   cr2 = output base  (128 rows x 16 out_ch x 512 bytes = 1048576 bytes)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Input layout (interleaved by row, 8 channels):
#   row r, channel ch, col c: offset = (r * 8 + ch) * 128 + c
#   Row stride = 8 * 128 = 1024 bytes
#   Channel stride = 128 bytes
#
# Output layout (interleaved by row, 32-bit accumulators):
#   row r, filter f: offset = (r * 16 + f) * 512
#
# Kernel layout per filter f (at XMEM offset f * 128 from kernel base):
#   For input channel ch (0..7):
#     byte[ch*9 + 0] = k(ch,-1,-1)   byte[ch*9 + 1] = k(ch,-1, 0)   byte[ch*9 + 2] = k(ch,-1,+1)
#     byte[ch*9 + 3] = k(ch, 0,-1)   byte[ch*9 + 4] = k(ch, 0, 0)   byte[ch*9 + 5] = k(ch, 0,+1)
#     byte[ch*9 + 6] = k(ch,+1,-1)   byte[ch*9 + 7] = k(ch,+1, 0)   byte[ch*9 + 8] = k(ch,+1,+1)
#   Bytes 72-127: padding (unused)
#
# Algorithm:
#   For each output filter f:
#     Load filter f's kernel (128 bytes) into r0
#     For each input channel ch:
#       Load 3 input rows (r-1, r, r+1) for channel ch
#       Apply 9 kernel taps, accumulating into acc
#     Store acc as one row of output channel f
#
# Mask slots in r_mask:
#   slot 0: all zeros       -> no masking        (dc = 0)
#   slot 1: bit 0 set       -> zero position 0   (dc = -1, left border)
#   slot 2: bit 127 set     -> zero position 127 (dc = +1, right border)
#
# Register allocation:
#   lr0  = current row input base offset (row * 1024)
#   lr1  = 127  (row loop end)
#   lr2  = output write offset
#   lr3  = kernel byte index (0-71 within current filter's r0)
#   lr4  = current input channel base (row_base + ch * 128, advances per channel)
#   lr5  = 1024 (input row stride = 8 * 128; also input channel loop end)
#   lr6  = row counter (main loop)
#   lr7  = 511  (cyclic offset for dc=-1)
#   lr8  = 0    (cyclic offset dc=0, mask slot 0, mask_shift, cyclic index)
#   lr9  = 1    (cyclic offset dc=+1, mask slot 1)
#   lr10 = 2    (mask slot 2)
#   lr11 = input channel counter (0, 128, 256, ..., 896)
#   lr12 = temp (computed input offsets)
#   lr14 = filter kernel offset (0, 128, ..., 1920)
#   lr15 = 2048 (filter loop end = 16 * 128)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 1024;
    set                 lr7 511;;

    ldr_mult_mask_reg   lr8 cr3;
    set                 lr9 1;;

    set                 lr10 2;
    set                 lr1 127;;

    set                 lr15 2048;;

# ===========================================================================
# Row 0 (top border) -- kernel rows 1 and 2 only (no row -1)
# ===========================================================================

    set                 lr14 0;
    set                 lr2 0;;

row0_filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 3;
    add                 lr4 lr8 lr8;;

    set                 lr11 0;;

row0_ch_loop:
    # Load row 0, this input channel
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

    # Load row 1, this input channel
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

    # Advance to next input channel (+4 = skip 3 missing taps + 1)
    incr                lr3 4;
    incr                lr4 128;;

    incr                lr11 128;;

    blt                 lr11 lr5 row0_ch_loop;;

    # Store output, advance to next filter
    str_acc_reg         lr2 cr2;
    incr                lr14 128;;

    incr                lr2 512;;

    blt                 lr14 lr15 row0_filter_loop;;

# ===========================================================================
# Rows 1-126 (main loop) -- all 3 kernel rows, 8 in_ch, 16 out filters
# ===========================================================================

    set                 lr0 1024;
    set                 lr6 1;;

row_loop:
    set                 lr14 0;;

filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    add                 lr4 lr0 lr8;;

    set                 lr11 0;;

ch_loop:
    # --- Kernel row 0: taps 0-2, row(r-1), this input channel ---
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

    # --- Kernel row 1: taps 3-5, row(r), this input channel ---
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

    # --- Kernel row 2: taps 6-8, row(r+1), this input channel ---
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

    # Advance to next input channel
    incr                lr3 1;
    incr                lr4 128;;

    incr                lr11 128;;

    blt                 lr11 lr5 ch_loop;;

    # Store output, advance to next filter
    str_acc_reg         lr2 cr2;
    incr                lr14 128;;

    incr                lr2 512;;

    blt                 lr14 lr15 filter_loop;;

    # Advance to next row
    incr                lr0 1024;
    incr                lr6 1;;

    blt                 lr6 lr1 row_loop;;

# ===========================================================================
# Row 127 (bottom border) -- kernel rows 0 and 1 only (no row 128)
# ===========================================================================

    set                 lr14 0;;

row127_filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    add                 lr4 lr0 lr8;;

    set                 lr11 0;;

row127_ch_loop:
    # --- Kernel row 0: taps 0-2, row 126, this input channel ---
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

    # --- Kernel row 1: taps 3-5, row 127, this input channel ---
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

    # Advance to next input channel (+4 = skip 3 missing taps + 1)
    incr                lr3 4;
    incr                lr4 128;;

    incr                lr11 128;;

    blt                 lr11 lr5 row127_ch_loop;;

    # Store output, advance to next filter
    str_acc_reg         lr2 cr2;
    incr                lr14 128;;

    incr                lr2 512;;

    blt                 lr14 lr15 row127_filter_loop;;

end:
    bkpt;;
