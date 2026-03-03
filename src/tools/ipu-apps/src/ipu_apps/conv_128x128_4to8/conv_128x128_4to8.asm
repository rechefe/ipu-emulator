# Standard 3x3 convolution: 4 input channels -> 8 output channels, 128x128.
#
# Each output channel has 4 input-channel 3x3 kernels (full cross-channel mixing).
# Kernel doesn't fit in r0 all at once (288 bytes > 128), so r0 is reloaded
# per output filter (36 bytes/filter, padded to 128, stored at 128-byte XMEM boundaries).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 rows x 4 ch x 128 bytes = 65536 bytes)
#   cr1 = kernel base  (8 filters x 128 bytes each = 1024 bytes)
#                       Each filter: 4 in_ch x 9 bytes = 36 bytes, padded to 128
#   cr2 = output base  (128 rows x 8 out_ch x 512 bytes = 524288 bytes)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Input layout (interleaved by row, 4 channels):
#   row r, channel ch, col c: offset = (r * 4 + ch) * 128 + c
#   Row stride = 4 * 128 = 512 bytes
#   Channel stride = 128 bytes
#
# Output layout (interleaved by row, 32-bit accumulators):
#   row r, filter f: offset = (r * 8 + f) * 512
#
# Kernel layout per filter f (at XMEM offset f * 128 from kernel base):
#   For input channel ch (0..3):
#     byte[ch*9 + 0] = k(ch,-1,-1)   byte[ch*9 + 1] = k(ch,-1, 0)   byte[ch*9 + 2] = k(ch,-1,+1)
#     byte[ch*9 + 3] = k(ch, 0,-1)   byte[ch*9 + 4] = k(ch, 0, 0)   byte[ch*9 + 5] = k(ch, 0,+1)
#     byte[ch*9 + 6] = k(ch,+1,-1)   byte[ch*9 + 7] = k(ch,+1, 0)   byte[ch*9 + 8] = k(ch,+1,+1)
#   Bytes 36-127: padding (unused)
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
#   lr0  = current row input base offset (row * 512)
#   lr1  = 127  (row loop end)
#   lr2  = output write offset
#   lr3  = kernel byte index (0-35 within current filter's r0)
#   lr4  = current input channel base (row_base + ch * 128, advances per channel)
#   lr5  = 512  (input row stride = 4 * 128; also input channel loop end)
#   lr6  = row counter (main loop)
#   lr7  = 511  (cyclic offset for dc=-1)
#   lr8  = 0    (cyclic offset dc=0, mask slot 0, mask_shift, cyclic index)
#   lr9  = 1    (cyclic offset dc=+1, mask slot 1)
#   lr10 = 2    (mask slot 2)
#   lr11 = input channel counter (0, 128, 256, 384)
#   lr12 = temp (computed input offsets)
#   lr14 = filter kernel offset (0, 128, ..., 896)
#   lr15 = 1024 (filter loop end = 8 * 128)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 512;
    set                 lr7 511;;

    ldr_mult_mask_reg   lr8 cr3;
    set                 lr9 1;;

    set                 lr10 2;
    set                 lr1 127;;

    set                 lr15 1024;;

# ===========================================================================
# Row 0 (top border) — kernel rows 1 and 2 only (no row -1)
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
# Rows 1–126 (main loop) — all 3 kernel rows, 4 in_ch, 8 out filters
# ===========================================================================

    set                 lr0 512;
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
    incr                lr0 512;
    incr                lr6 1;;

    blt                 lr6 lr1 row_loop;;

# ===========================================================================
# Row 127 (bottom border) — kernel rows 0 and 1 only (no row 128)
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
