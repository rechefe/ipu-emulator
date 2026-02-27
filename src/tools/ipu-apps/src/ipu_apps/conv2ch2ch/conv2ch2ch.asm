# 2-channel to 2-channel convolution: 128x128x2 input, 3x3x2x2 kernel, 128x128x2 output.
#
# Regular convolution with 2 input channels and 2 output channels.
# Each output channel sums over both input channels.
#
# Border handling:
#   - Row 0 and row 127 are handled outside the main loop (missing top/bottom neighbor).
#   - Column 0 and column 127 edges are masked via r_mask.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 rows x 2 ch x 128 bytes = 32768 bytes, interleaved by row)
#   cr1 = kernel base  (36 bytes in positions 0-35, padded to 128 bytes)
#   cr2 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#   cr3 = output base  (128 rows x 2 ch x 512 bytes = 131072 bytes, interleaved by row)
#
# Input interleaving: row0_ch0, row0_ch1, row1_ch0, row1_ch1, ...
#   ch0 row r: offset = r * 256
#   ch1 row r: offset = r * 256 + 128
#
# Output interleaving: row0_outch0, row0_outch1, row1_outch0, row1_outch1, ...
#   outch0 row r: offset = r * 1024
#   outch1 row r: offset = r * 1024 + 512
#
# Kernel layout in r0[0..35]:
#   Output channel 0 (indices 0-17):
#     [0..2]   = row-1_inch0   [3..5]   = row-1_inch1
#     [6..8]   = row0_inch0    [9..11]  = row0_inch1
#     [12..14] = row+1_inch0   [15..17] = row+1_inch1
#   Output channel 1 (indices 18-35):
#     [18..20] = row-1_inch0   [21..23] = row-1_inch1
#     [24..26] = row0_inch0    [27..29] = row0_inch1
#     [30..32] = row+1_inch0   [33..35] = row+1_inch1
#
# Mask slots in r_mask:
#   slot 0: all zeros         -> no masking        (dc = 0)
#   slot 1: bit 0 set         -> zero position 0   (dc = -1, left border)
#   slot 2: bit 127 set       -> zero position 127 (dc = +1, right border)
#
# Register allocation:
#   lr0  = current row ch0 input offset (r * 256, main loop variable)
#   lr1  = end offset for main loop (exclusive)
#   lr2  = output write offset for outch0 (r * 1024)
#   lr3  = temp / kernel index
#   lr4  = temp (computed offsets)
#   lr5  = 256  (input row stride = 128 * 2 channels)
#   lr6  = 512  (output channel offset within a row pair)
#   lr7  = 511  (cyclic offset for dc=-1)
#   lr8  = 0    (multi-purpose zero)
#   lr9  = 1    (cyclic offset dc=+1, mask slot 1)
#   lr10 = 2    (mask slot 2)
#   lr11 = 128  (input channel offset within a row pair)

# ===========================================================================
# Initialization
# ===========================================================================

    # Load kernel weights into r0 and mask data into r_mask
    ldr_mult_reg        r0 lr8 cr1;
    set                 lr5 256;
    set                 lr7 511;;

    ldr_mult_mask_reg   lr8 cr2;
    set                 lr6 512;
    set                 lr9 1;;

    set                 lr10 2;
    set                 lr11 128;;

# ===========================================================================
# Row 0 (top border) — kernel rows 0 and +1 only (no row -1)
# ===========================================================================

    # ---- Output channel 0 (kernel indices 6-17) ----

    reset_acc;;

    # Kernel row 0, inch0: k6,k7,k8 on input ch0 row 0
    ldr_cyclic_mult_reg lr8 cr0 lr8;
    set                 lr3 6;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 7;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 8;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Kernel row 0, inch1: k9,k10,k11 on input ch1 row 0
    ldr_cyclic_mult_reg lr11 cr0 lr8;
    set                 lr3 9;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 10;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 11;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Kernel row +1, inch0: k12,k13,k14 on input ch0 row 1
    ldr_cyclic_mult_reg lr5 cr0 lr8;
    set                 lr3 12;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 13;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 14;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Kernel row +1, inch1: k15,k16,k17 on input ch1 row 1
    add                 lr4 lr5 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 15;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 16;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 17;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store outch0 row 0
    str_acc_reg         lr8 cr3;;

    # ---- Output channel 1 (kernel indices 24-35) ----

    reset_acc;;

    # Kernel row 0, inch0: k24,k25,k26 on input ch0 row 0
    ldr_cyclic_mult_reg lr8 cr0 lr8;
    set                 lr3 24;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 25;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 26;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Kernel row 0, inch1: k27,k28,k29 on input ch1 row 0
    ldr_cyclic_mult_reg lr11 cr0 lr8;
    set                 lr3 27;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 28;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 29;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Kernel row +1, inch0: k30,k31,k32 on input ch0 row 1
    ldr_cyclic_mult_reg lr5 cr0 lr8;
    set                 lr3 30;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 31;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 32;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Kernel row +1, inch1: k33,k34,k35 on input ch1 row 1
    add                 lr4 lr5 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 33;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 34;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 35;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store outch1 row 0
    str_acc_reg         lr6 cr3;;

# ===========================================================================
# Rows 1–126 (main loop) — all 3 kernel rows, both input channels, both output channels
# ===========================================================================

    set                 lr0 256;
    set                 lr1 32512;;
    set                 lr2 1024;;

row_loop:

    # ================================================================
    # Output channel 0 (kernel indices 0-17)
    # ================================================================

    reset_acc;;

    # --- Kernel row -1, inch0: k0,k1,k2 on input ch0 row (r-1) ---
    sub                 lr3 lr0 lr5;
    ldr_cyclic_mult_reg lr3 cr0 lr8;;

    mult.ve             r0 lr7 lr9 lr8 lr8;
    acc;;

    mult.ve             r0 lr8 lr8 lr8 lr9;
    acc;;

    mult.ve             r0 lr9 lr10 lr8 lr10;
    acc;;

    # --- Kernel row -1, inch1: k3,k4,k5 on input ch1 row (r-1) ---
    sub                 lr4 lr0 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 3;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 4;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 5;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 0, inch0: k6,k7,k8 on input ch0 row r ---
    ldr_cyclic_mult_reg lr0 cr0 lr8;
    set                 lr3 6;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 7;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 8;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 0, inch1: k9,k10,k11 on input ch1 row r ---
    add                 lr4 lr0 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 9;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 10;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 11;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row +1, inch0: k12,k13,k14 on input ch0 row (r+1) ---
    add                 lr4 lr0 lr5;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 12;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 13;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 14;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row +1, inch1: k15,k16,k17 on input ch1 row (r+1) ---
    add                 lr4 lr4 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 15;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 16;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 17;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store outch0
    str_acc_reg         lr2 cr3;;

    # ================================================================
    # Output channel 1 (kernel indices 18-35)
    # ================================================================

    reset_acc;;

    # --- Kernel row -1, inch0: k18,k19,k20 on input ch0 row (r-1) ---
    sub                 lr3 lr0 lr5;
    ldr_cyclic_mult_reg lr3 cr0 lr8;
    set                 lr4 18;;

    mult.ve             r0 lr7 lr9 lr8 lr4;
    acc;;

    set                 lr3 19;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 20;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row -1, inch1: k21,k22,k23 on input ch1 row (r-1) ---
    sub                 lr4 lr0 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 21;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 22;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 23;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 0, inch0: k24,k25,k26 on input ch0 row r ---
    ldr_cyclic_mult_reg lr0 cr0 lr8;
    set                 lr3 24;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 25;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 26;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 0, inch1: k27,k28,k29 on input ch1 row r ---
    add                 lr4 lr0 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 27;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 28;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 29;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row +1, inch0: k30,k31,k32 on input ch0 row (r+1) ---
    add                 lr4 lr0 lr5;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 30;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 31;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 32;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row +1, inch1: k33,k34,k35 on input ch1 row (r+1) ---
    add                 lr4 lr4 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 33;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 34;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 35;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store outch1
    add                 lr4 lr2 lr6;
    str_acc_reg         lr4 cr3;;

    # Advance to next row
    incr                lr0 256;
    incr                lr2 1024;;

    blt                 lr0 lr1 row_loop;;

# ===========================================================================
# Row 127 (bottom border) — kernel rows -1 and 0 only (no row 128)
# ===========================================================================

    # ---- Output channel 0 (kernel indices 0-11) ----

    reset_acc;;

    # --- Kernel row -1, inch0: k0,k1,k2 on input ch0 row 126 ---
    sub                 lr3 lr0 lr5;
    ldr_cyclic_mult_reg lr3 cr0 lr8;;

    mult.ve             r0 lr7 lr9 lr8 lr8;
    acc;;

    mult.ve             r0 lr8 lr8 lr8 lr9;
    acc;;

    mult.ve             r0 lr9 lr10 lr8 lr10;
    acc;;

    # --- Kernel row -1, inch1: k3,k4,k5 on input ch1 row 126 ---
    sub                 lr4 lr0 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 3;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 4;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 5;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 0, inch0: k6,k7,k8 on input ch0 row 127 ---
    ldr_cyclic_mult_reg lr0 cr0 lr8;
    set                 lr3 6;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 7;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 8;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 0, inch1: k9,k10,k11 on input ch1 row 127 ---
    add                 lr4 lr0 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 9;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 10;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 11;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store outch0 row 127
    str_acc_reg         lr2 cr3;;

    # ---- Output channel 1 (kernel indices 18-29) ----

    reset_acc;;

    # --- Kernel row -1, inch0: k18,k19,k20 on input ch0 row 126 ---
    sub                 lr3 lr0 lr5;
    ldr_cyclic_mult_reg lr3 cr0 lr8;
    set                 lr4 18;;

    mult.ve             r0 lr7 lr9 lr8 lr4;
    acc;;

    set                 lr3 19;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 20;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row -1, inch1: k21,k22,k23 on input ch1 row 126 ---
    sub                 lr4 lr0 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 21;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 22;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 23;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 0, inch0: k24,k25,k26 on input ch0 row 127 ---
    ldr_cyclic_mult_reg lr0 cr0 lr8;
    set                 lr3 24;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 25;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 26;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 0, inch1: k27,k28,k29 on input ch1 row 127 ---
    add                 lr4 lr0 lr11;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 27;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 28;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 29;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store outch1 row 127
    add                 lr4 lr2 lr6;
    str_acc_reg         lr4 cr3;;

end:
    bkpt;;
