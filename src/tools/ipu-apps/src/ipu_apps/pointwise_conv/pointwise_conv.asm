# Pointwise convolution: 128x128x8 input, 1x1x8x8 kernel, 128x128x8 output.
#
# 1x1 convolution with 8 input channels and 8 output channels.
# No spatial kernel — pure channel mixing (pointwise).
#
# Pipelining strategy:
#   r_cyclic has 4 slots (S0-S3, 128 bytes each, 512 total).
#   8 input channels require loading in two groups:
#     Group A (ich0-3): loaded into S0-S3 before the inner loop
#     Group B (ich4-7): loaded into S0-S3 during computation (pipelined)
#   After mult.ve reads a slot, the next cycle loads new data into that slot.
#   This hides all load latency, achieving 100% SIMD utilization.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 rows x 8 ch x 128 bytes = 131072 bytes)
#   cr1 = kernel base  (64 bytes in positions 0-63, padded to 128 bytes)
#   cr2 = mask   base  (128 bytes: all zeros, no masking needed)
#   cr3 = output base  (128 rows x 8 ch x 512 bytes = 524288 bytes)
#
# Input interleaving: row0_ch0, row0_ch1, ..., row0_ch7, row1_ch0, ...
#   Channel ic, row r: offset = r * 1024 + ic * 128
#
# Output interleaving: row0_outch0, ..., row0_outch7, row1_outch0, ...
#   Output channel oc, row r: offset = (r * 8 + oc) * 512
#
# Kernel layout in r0[0..63]:
#   kernel[oc * 8 + ic] = weight for output channel oc, input channel ic
#
# Register allocation:
#   lr0  = input row base address (r * 1024)
#   lr1  = output pointer (pre-offset by -512, auto-advances by 512 each store)
#   lr2  = row counter (0..127)
#   lr3  = output channel counter (0..7)
#   lr4  = kernel index (0..63, advances linearly by 1 per mult.ve)
#   lr5  = 512  (output stride / input group-B offset)
#   lr6  = 128  (cyclic S1 offset / row limit)
#   lr7  = 0    (cyclic S0 offset / no-mask constant)
#   lr8  = 256  (cyclic S2 offset)
#   lr9  = 384  (cyclic S3 offset)
#   lr10 = 8    (output channel limit)
#   lr14 = temp (address computation)

# ===========================================================================
# Initialization
# ===========================================================================

    # Load kernel weights into r0 and mask data into r_mask
    # (lr7 defaults to 0 — used as zero offset)
    ldr_mult_reg        r0 lr7 cr1;
    set                 lr5 512;
    set                 lr6 128;;

    ldr_mult_mask_reg   lr7 cr2;
    set                 lr8 256;
    set                 lr9 384;;

    set                 lr10 8;
    set                 lr1 -512;;

# ===========================================================================
# Row loop (128 rows)
# ===========================================================================

row_loop:

    # Load initial 4 input channels (group A) into r_cyclic S0-S3
    ldr_cyclic_mult_reg lr0 cr0 lr7;;

    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;;

    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;;

    add                 lr14 lr0 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    set                 lr3 0;
    set                 lr4 0;;

# ---------------------------------------------------------------------------
# Inner loop: 8 output channels x 12 cycles each
# ---------------------------------------------------------------------------

LOOP_OCH:

    # W1: Reset accumulator
    reset_acc;;

    # W2: mult ich0 (S0), kernel[lr4=oc*8+0], accumulate
    mult.ve             r0 lr7 lr7 lr7 lr4;
    acc;;

    # W3: incr lr4; load ich4->S0; mult ich1 (S1); acc
    incr                lr4 1;
    add                 lr14 lr0 lr5;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve             r0 lr6 lr7 lr7 lr4;
    acc;;

    # W4: incr lr4; load ich5->S1; mult ich2 (S2); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r0 lr8 lr7 lr7 lr4;
    acc;;

    # W5: incr lr4; load ich6->S2; mult ich3 (S3); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r0 lr9 lr7 lr7 lr4;
    acc;;

    # W6: incr lr4; load ich7->S3; mult ich4 (S0); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve             r0 lr7 lr7 lr7 lr4;
    acc;;

    # W7: incr lr4; load ich0->S0; mult ich5 (S1); acc
    incr                lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    mult.ve             r0 lr6 lr7 lr7 lr4;
    acc;;

    # W8: incr lr4; load ich1->S1; mult ich6 (S2); acc
    incr                lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r0 lr8 lr7 lr7 lr4;
    acc;;

    # W9: incr lr4; load ich2->S2; mult ich7 (S3); acc
    incr                lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r0 lr9 lr7 lr7 lr4;
    acc;;

    # W10: incr lr4; load ich3->S3
    incr                lr4 1;
    add                 lr14 lr0 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # W11: advance output pointer (+512), store accumulator, incr oc counter
    add                 lr1 lr1 lr5;
    incr                lr3 1;
    str_acc_reg         lr1 cr3;;

    # W12: branch if more output channels
    blt                 lr3 lr10 LOOP_OCH;;

# ---------------------------------------------------------------------------
# Advance to next row
# ---------------------------------------------------------------------------

    incr                lr0 1024;
    incr                lr2 1;;

    blt                 lr2 lr6 row_loop;;

end:
    bkpt;;
