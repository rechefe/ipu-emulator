# Stride-2 Depthwise 3x3 Convolution
#
# Computes depthwise 3x3 convolution with stride 2 in both dimensions.
# Input: 128x128xC (INT8), Output: 64x64xC (INT8, quantized).
#
# Output row j centers at input row (2j + 1), needing input rows 2j, 2j+1, 2j+2.
# Output chunk k packs output rows 2k and 2k+1 (2 x 64 = 128 bytes).
#   Row 2k   centers at input row 4k+1 (needs 4k, 4k+1, 4k+2)
#   Row 2k+1 centers at input row 4k+3 (needs 4k+2, 4k+3, 4k+4)
#
# Stride flow per channel per output chunk:
#   1. Compute full 128-elem conv for row A -> r_acc -> aaq -> temp_A
#   2. Compute full 128-elem conv for row B -> r_acc -> aaq -> temp_B
#   3. Load temp_A, identity mult, acc.stride offset=0 -> r_acc[0:63]
#   4. Load temp_B, identity mult, acc.stride offset=2 -> r_acc[64:127]
#   5. aaq -> xmem.store_aaq_result -> output
#
# Last chunk (k=31): row 62 computed normally, row 63 skipped (needs
# input row 128 which doesn't exist). Output zeros for row 63.
#
# CR registers:
#   cr0 = input base address
#   cr1 = kernel base address
#   cr2 = output base address
#   cr3 = mask base address
#   cr4 = cols (128)
#   cr5 = num_output_chunks (32)
#   cr6 = group_stride (channels * 128)
#   cr7 = 1024 (channel group size = 8 * 128)
#   cr8 = temp region base address
#   cr9 = 1 (identity scalar for mult.ve.cr)
#   cr12 = 128 (step constant for add)
#
# LR registers:
#   lr0  = 0     (zero, mask slot 0, mask_shift)
#   lr1  = 1     (mask slot 1 = left border, kc offset)
#   lr2  = 2     (mask slot 2 = right border, stride offset for row B)
#   lr3  = 128 - cols  (kr=-1 base)
#   lr4  = 128   (kr=0 base, S1 cyclic index, channel stride)
#   lr5  = 128 + cols  (kr=+1 base)
#   lr6  = kernel byte index
#   lr7  = output pointer
#   lr8  = input base for chunk pair (4k * group_stride)
#   lr9  = output chunk counter
#   lr10 = channel offset (0, 128, ..., group_stride-128)
#   lr11 = channel group limit
#   lr12 = kernel memory offset
#   lr13 = group_stride
#   lr14 = temp
#   lr15 = temp
#
# Mask slots (same as non-strided depthwise):
#   slot 0: all zeros      -> no masking (kc=0)
#   slot 1: left border    -> zero col 0 of each packed row (kc=-1)
#   slot 2: right border   -> zero last col of each packed row (kc=+1)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr0 0;
    ldr_mult_mask_reg   lr0 cr3;;

    set                 lr4 128;
    set                 lr1 1;;

    set                 lr2 2;
    sub                 lr3 lr4 cr4;;

    add                 lr5 lr4 cr4;
    add                 lr13 lr0 cr6;;

# ===========================================================================
# Main loop: output chunks 0..30  (2 full output rows per chunk)
# ===========================================================================

    set                 lr8 0;
    set                 lr7 0;;

    set                 lr9 0;;

chunk_loop:
    set                 lr10 0;
    set                 lr12 0;;

chunk_kg_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr11 lr10 cr7;;

chunk_ch_loop:

    # =======================================================================
    # Row A: conv at input row 4k+1 (S0=4k, S1=4k+1, S2=4k+2)
    # =======================================================================

    # Compute S1 addr = lr8 + lr13 + lr10
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;

    # Load S0 (row 4k) at cyclic[0]
    sub                 lr15 lr14 lr13;
    ldr_cyclic_mult_reg lr15 cr0 lr0;;

    # Load S1 (row 4k+1) at cyclic[128]
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

    # Load S2 (row 4k+2) at cyclic[256]
    add                 lr15 lr14 lr13;
    add                 lr14 lr4 lr4;;

    ldr_cyclic_mult_reg lr15 cr0 lr14;
    reset_acc;;

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

    # Quantize row A result and store to temp_A
    aaq;;
    xmem.store_aaq_result lr0 cr8;;

    # =======================================================================
    # Row B: conv at input row 4k+3 (S0=4k+2, S1=4k+3, S2=4k+4)
    # =======================================================================

    # Reset lr6 to channel's kernel base (undo 9 increments, currently at base+8)
    sub                 lr6 lr6 8;;

    # Compute S1 addr = lr8 + 3*lr13 + lr10
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;
    add                 lr14 lr14 lr13;;

    # Load S0 (row 4k+2) at cyclic[0]
    sub                 lr15 lr14 lr13;
    ldr_cyclic_mult_reg lr15 cr0 lr0;;

    # Load S1 (row 4k+3) at cyclic[128]
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

    # Load S2 (row 4k+4) at cyclic[256]
    add                 lr15 lr14 lr13;
    add                 lr14 lr4 lr4;;

    ldr_cyclic_mult_reg lr15 cr0 lr14;
    reset_acc;;

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

    # Quantize row B result and store to temp_B
    aaq;;
    xmem.store_aaq_result lr4 cr8;;

    # =======================================================================
    # Stride step: decimate both rows and pack into one output chunk
    # =======================================================================

    # Load temp_A into cyclic[0], identity multiply, reset acc
    ldr_cyclic_mult_reg lr0 cr8 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;
    reset_acc;;

    # Stride row A -> r_acc[0:63]
    acc.stride           64 on off lr0;;

    # Load temp_B into cyclic[0], identity multiply
    ldr_cyclic_mult_reg lr4 cr8 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;;

    # Stride row B -> r_acc[64:127]
    acc.stride           64 on off lr2;;

    # Quantize packed result and store to output
    aaq;;
    xmem.store_aaq_result lr7 cr2;;

    # Advance to next channel
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;;

    add                 lr7 lr7 cr12;;

    blt                 lr10 lr11 chunk_ch_loop;;

    # Advance to next kernel group
    add                 lr12 lr12 cr12;;

    blt                 lr10 lr13 chunk_kg_loop;;

    # Advance to next output chunk pair
    # Input base advances by 4 rows = 4 * group_stride
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;
    add                 lr8 lr8 lr13;;

    add                 lr9 lr9 1;;

    set                 lr15 31;;

    blt                 lr9 lr15 chunk_loop;;

# ===========================================================================
# Last chunk (k=31): row 62 only, row 63 skipped (bottom border)
# ===========================================================================

    set                 lr10 0;
    set                 lr12 0;;

last_kg_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr11 lr10 cr7;;

last_ch_loop:

    # =======================================================================
    # Row A only (output row 62, center at input row 125)
    # S0=row124, S1=row125, S2=row126
    # =======================================================================

    # Compute S1 addr = lr8 + lr13 + lr10
    add                 lr14 lr8 lr10;;
    add                 lr14 lr14 lr13;;

    # Load S0 at cyclic[0]
    sub                 lr15 lr14 lr13;
    ldr_cyclic_mult_reg lr15 cr0 lr0;;

    # Load S1 at cyclic[128]
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

    # Load S2 at cyclic[256]
    add                 lr15 lr14 lr13;
    add                 lr14 lr4 lr4;;

    ldr_cyclic_mult_reg lr15 cr0 lr14;
    reset_acc;;

    # --- 9 taps (same pattern) ---
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

    # Quantize row A and store to temp_A
    aaq;;
    xmem.store_aaq_result lr0 cr8;;

    # =======================================================================
    # Stride step: row A only, row B = zeros (from reset_acc)
    # =======================================================================

    # Load temp_A, identity multiply, reset acc
    ldr_cyclic_mult_reg lr0 cr8 lr0;
    mult.ve.cr          lr0 0 lr0 cr9;
    reset_acc;;

    # Stride row A -> r_acc[0:63], r_acc[64:127] stays zero
    acc.stride           64 on off lr0;;

    # Quantize and store
    aaq;;
    xmem.store_aaq_result lr7 cr2;;

    # Advance
    add                 lr6 lr6 1;
    add                 lr10 lr10 cr12;;

    add                 lr7 lr7 cr12;;

    blt                 lr10 lr11 last_ch_loop;;

    add                 lr12 lr12 cr12;;

    blt                 lr10 lr13 last_kg_loop;;

end:
    bkpt;;
