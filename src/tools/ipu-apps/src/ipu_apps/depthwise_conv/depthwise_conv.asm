# Depthwise convolution: 1 channel, 128x128, 3x3 kernel, same-size output.
#
# Border handling:
#   - Row 0 and row 127 are handled outside the main loop (missing top/bottom neighbor).
#   - Column 0 and column 127 edges are masked via r_mask.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 rows x 128 bytes = 16384 bytes)
#   cr1 = kernel base  (9 bytes in positions 0-8, padded to 128 bytes)
#   cr2 = output base  (128 rows x 512 bytes = 65536 bytes, 32-bit accumulator words)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Kernel layout in r0[0..8]:
#   [0]=k(-1,-1)  [1]=k(-1, 0)  [2]=k(-1,+1)
#   [3]=k( 0,-1)  [4]=k( 0, 0)  [5]=k( 0,+1)
#   [6]=k(+1,-1)  [7]=k(+1, 0)  [8]=k(+1,+1)
#
# Mask slots in r_mask:
#   slot 0: all zeros         -> no masking        (dc = 0)
#   slot 1: bit 0 set         -> zero position 0   (dc = -1, left border)
#   slot 2: bit 127 set       -> zero position 127 (dc = +1, right border)
#
# Register allocation:
#   lr0  = current row input offset  (main loop variable)
#   lr1  = end offset for main loop  (exclusive)
#   lr2  = output write offset
#   lr3  = temp / kernel index
#   lr4  = temp (row+1 offset)
#   lr5  = 128  (row stride constant)
#   lr6  = 512  (output stride constant)
#   lr7  = 511  (cyclic offset for dc=-1)
#   lr8  = 0    (cyclic offset dc=0, mask slot 0, mask_shift, cyclic index, kernel idx 0)
#   lr9  = 1    (cyclic offset dc=+1, mask slot 1, kernel idx 1)
#   lr10 = 2    (mask slot 2, kernel idx 2)
#   lr15 = 0    (unused alias, kept at 0)

# ===========================================================================
# Initialization
# ===========================================================================

    # Load kernel weights into r0 and mask data into r_mask
    ldr_mult_reg        r0 lr8 cr1;
    set                 lr5 128;
    set                 lr7 511;;

    ldr_mult_mask_reg   lr8 cr3;
    set                 lr6 512;
    set                 lr9 1;;

    set                 lr10 2;;

# ===========================================================================
# Row 0 (top border) — kernel rows 1 and 2 only (no row -1)
# ===========================================================================

    reset_acc;;

    # Load input row 0 into r_cyclic
    ldr_cyclic_mult_reg lr8 cr0 lr8;;

    # Kernel row 1: k3, k4, k5 applied to row 0
    set                 lr3 3;
    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 4;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 5;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Load input row 1 into r_cyclic
    ldr_cyclic_mult_reg lr5 cr0 lr8;
    set                 lr3 6;;

    # Kernel row 2: k6, k7, k8 applied to row 1
    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 7;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 8;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store row 0 output
    str_acc_reg         lr8 cr2;;

# ===========================================================================
# Rows 1–126 (main loop) — all 3 kernel rows
# ===========================================================================

    set                 lr0 128;
    set                 lr1 16256;;
    set                 lr2 512;;

row_loop:
    reset_acc;;

    # --- Kernel row 0: k0, k1, k2 applied to row (r-1) ---
    sub                 lr3 lr0 lr5;
    ldr_cyclic_mult_reg lr3 cr0 lr8;;

    mult.ve             r0 lr7 lr9 lr8 lr8;
    acc;;

    mult.ve             r0 lr8 lr8 lr8 lr9;
    acc;;

    mult.ve             r0 lr9 lr10 lr8 lr10;
    acc;;

    # --- Kernel row 1: k3, k4, k5 applied to row r ---
    ldr_cyclic_mult_reg lr0 cr0 lr8;
    set                 lr3 3;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 4;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 5;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # --- Kernel row 2: k6, k7, k8 applied to row (r+1) ---
    add                 lr4 lr0 lr5;
    ldr_cyclic_mult_reg lr4 cr0 lr8;
    set                 lr3 6;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 7;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 8;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store output and advance
    str_acc_reg         lr2 cr2;;

    incr                lr0 128;
    incr                lr2 512;;

    blt                 lr0 lr1 row_loop;;

# ===========================================================================
# Row 127 (bottom border) — kernel rows 0 and 1 only (no row 128)
# ===========================================================================

    reset_acc;;

    # --- Kernel row 0: k0, k1, k2 applied to row 126 ---
    sub                 lr3 lr0 lr5;
    ldr_cyclic_mult_reg lr3 cr0 lr8;;

    mult.ve             r0 lr7 lr9 lr8 lr8;
    acc;;

    mult.ve             r0 lr8 lr8 lr8 lr9;
    acc;;

    mult.ve             r0 lr9 lr10 lr8 lr10;
    acc;;

    # --- Kernel row 1: k3, k4, k5 applied to row 127 ---
    ldr_cyclic_mult_reg lr0 cr0 lr8;
    set                 lr3 3;;

    mult.ve             r0 lr7 lr9 lr8 lr3;
    acc;;

    set                 lr3 4;
    mult.ve             r0 lr8 lr8 lr8 lr3;
    acc;;

    set                 lr3 5;
    mult.ve             r0 lr9 lr10 lr8 lr3;
    acc;;

    # Store row 127 output
    str_acc_reg         lr2 cr2;;

end:
    bkpt;;
