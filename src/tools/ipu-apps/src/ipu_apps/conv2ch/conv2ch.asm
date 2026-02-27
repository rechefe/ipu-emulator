# 2-channel convolution: 128x128x2 input, 3x3x2 kernel, 128x128x1 output.
#
# Regular convolution: each output pixel sums over both input channels.
#
# Border handling:
#   - Row 0 and row 127 are handled outside the main loop (missing top/bottom neighbor).
#   - Column 0 and column 127 edges are masked via r_mask.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 rows x 2 channels x 128 bytes = 32768 bytes, interleaved by row)
#   cr1 = kernel base  (18 bytes in positions 0-17, padded to 128 bytes)
#   cr2 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#   cr3 = output base  (128 rows x 512 bytes = 65536 bytes, 32-bit accumulator words)
#
# Input interleaving: row0_ch0, row0_ch1, row1_ch0, row1_ch1, ...
#   ch0 row r: offset = r * 256
#   ch1 row r: offset = r * 256 + 128
#
# Kernel layout in r0[0..17] (interleaved same way):
#   [0]=k_ch0(-1,-1)  [1]=k_ch0(-1, 0)  [2]=k_ch0(-1,+1)
#   [3]=k_ch1(-1,-1)  [4]=k_ch1(-1, 0)  [5]=k_ch1(-1,+1)
#   [6]=k_ch0( 0,-1)  [7]=k_ch0( 0, 0)  [8]=k_ch0( 0,+1)
#   [9]=k_ch1( 0,-1) [10]=k_ch1( 0, 0) [11]=k_ch1( 0,+1)
#  [12]=k_ch0(+1,-1) [13]=k_ch0(+1, 0) [14]=k_ch0(+1,+1)
#  [15]=k_ch1(+1,-1) [16]=k_ch1(+1, 0) [17]=k_ch1(+1,+1)
#
# Mask slots in r_mask:
#   slot 0: all zeros         -> no masking        (dc = 0)
#   slot 1: bit 0 set         -> zero position 0   (dc = -1, left border)
#   slot 2: bit 127 set       -> zero position 127 (dc = +1, right border)
#
# Register allocation:
#   lr0  = current row ch0 input offset (r * 256, main loop variable)
#   lr1  = end offset for main loop (exclusive)
#   lr2  = output write offset
#   lr3  = temp / kernel index
#   lr4  = temp (computed offsets)
#   lr5  = 256  (row stride = 128 * 2 channels)
#   lr6  = 512  (output stride)
#   lr7  = 511  (cyclic offset for dc=-1)
#   lr8  = 0    (multi-purpose zero: cyclic offset dc=0, mask slot 0, mask_shift, cyclic index)
#   lr9  = 1    (cyclic offset dc=+1, mask slot 1)
#   lr10 = 2    (mask slot 2)
#   lr11 = 128  (channel offset within a row pair)

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

    reset_acc;;

    # --- Kernel row 0, ch0: k6,k7,k8 applied to input ch0 row 0 ---
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

    # --- Kernel row 0, ch1: k9,k10,k11 applied to input ch1 row 0 ---
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

    # --- Kernel row +1, ch0: k12,k13,k14 applied to input ch0 row 1 ---
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

    # --- Kernel row +1, ch1: k15,k16,k17 applied to input ch1 row 1 ---
    # ch1 row 1 offset = 256 + 128 = 384
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

    # Store row 0 output
    str_acc_reg         lr8 cr3;;

# ===========================================================================
# Rows 1–126 (main loop) — all 3 kernel rows, both channels
# ===========================================================================

    set                 lr0 256;
    set                 lr1 32512;;
    set                 lr2 512;;

row_loop:
    reset_acc;;

    # === Kernel row -1, ch0: k0,k1,k2 applied to input ch0 row (r-1) ===
    sub                 lr3 lr0 lr5;
    ldr_cyclic_mult_reg lr3 cr0 lr8;;

    mult.ve             r0 lr7 lr9 lr8 lr8;
    acc;;

    mult.ve             r0 lr8 lr8 lr8 lr9;
    acc;;

    mult.ve             r0 lr9 lr10 lr8 lr10;
    acc;;

    # === Kernel row -1, ch1: k3,k4,k5 applied to input ch1 row (r-1) ===
    # ch1 row (r-1) = lr3 + 128; but lr3 was written several cycles ago
    # snapshot lr3 still = lr0 - 256
    add                 lr4 lr3 lr11;
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

    # === Kernel row 0, ch0: k6,k7,k8 applied to input ch0 row r ===
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

    # === Kernel row 0, ch1: k9,k10,k11 applied to input ch1 row r ===
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

    # === Kernel row +1, ch0: k12,k13,k14 applied to input ch0 row (r+1) ===
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

    # === Kernel row +1, ch1: k15,k16,k17 applied to input ch1 row (r+1) ===
    # lr4 = lr0 + 256 from the add above (still in snapshot)
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

    # Store output and advance
    str_acc_reg         lr2 cr3;;

    incr                lr0 256;
    incr                lr2 512;;

    blt                 lr0 lr1 row_loop;;

# ===========================================================================
# Row 127 (bottom border) — kernel rows -1 and 0 only (no row 128)
# ===========================================================================

    reset_acc;;

    # === Kernel row -1, ch0: k0,k1,k2 applied to input ch0 row 126 ===
    sub                 lr3 lr0 lr5;
    ldr_cyclic_mult_reg lr3 cr0 lr8;;

    mult.ve             r0 lr7 lr9 lr8 lr8;
    acc;;

    mult.ve             r0 lr8 lr8 lr8 lr9;
    acc;;

    mult.ve             r0 lr9 lr10 lr8 lr10;
    acc;;

    # === Kernel row -1, ch1: k3,k4,k5 applied to input ch1 row 126 ===
    add                 lr4 lr3 lr11;
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

    # === Kernel row 0, ch0: k6,k7,k8 applied to input ch0 row 127 ===
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

    # === Kernel row 0, ch1: k9,k10,k11 applied to input ch1 row 127 ===
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

    # Store row 127 output
    str_acc_reg         lr2 cr3;;

end:
    bkpt;;
