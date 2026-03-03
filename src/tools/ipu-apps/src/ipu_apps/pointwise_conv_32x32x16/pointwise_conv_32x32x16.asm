# Pointwise convolution: 32x32x16 input, 1x1x16x16 kernel, 32x32x16 output.
#
# 1x1 convolution with 16 input channels and 16 output channels.
# No spatial kernel — pure channel mixing (pointwise).
#
# Since spatial width (32) < SIMD width (128), 4 spatial rows are packed
# per 128-byte chunk (128 / 32 = 4 rows per chunk).
# The outer loop iterates over 8 row groups (32 rows / 4 per group).
#
# Kernel has 16x16 = 256 weights, exceeding a single mult register (128 bytes).
# Solution: load kernel[0..127] into r0 (oc 0-7) and kernel[128..255] into r1
# (oc 8-15). Two inner loops: LOOP_OCH_A uses r0, LOOP_OCH_B uses r1.
#
# Pipelining strategy:
#   r_cyclic has 4 slots (S0-S3, 128 bytes each, 512 total).
#   16 input channels require loading in four groups:
#     Group A (ich0-3): loaded into S0-S3 before the inner loop
#     Groups B-D (ich4-15): loaded during computation (pipelined)
#   After mult.ve reads a slot, the next cycle loads new data into that slot.
#   This hides all load latency, achieving 100% SIMD utilization.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (8 row-groups x 16 ch x 128 bytes = 16384 bytes)
#   cr1 = kernel base  (256 bytes: oc0-7 in [0..127], oc8-15 in [128..255])
#   cr2 = mask   base  (128 bytes: all zeros, no masking needed)
#   cr3 = output base  (8 row-groups x 16 ch x 512 bytes = 65536 bytes)
#
# Input interleaving (per row group of 4 rows):
#   rg0_ch0(128B), rg0_ch1(128B), ..., rg0_ch15(128B), rg1_ch0, ...
#   Channel ic, row-group rg: offset = rg * 2048 + ic * 128
#
# Output interleaving (per row group):
#   rg0_outch0(512B), rg0_outch1(512B), ..., rg0_outch15(512B), rg1_outch0, ...
#   Output channel oc, row-group rg: offset = (rg * 16 + oc) * 512
#
# Kernel layout:
#   r0[oc * 16 + ic] for oc=0..7,  ic=0..15  (first 128 bytes)
#   r1[(oc-8)*16 + ic] for oc=8..15, ic=0..15  (second 128 bytes)
#
# Register allocation:
#   lr0  = input row-group base address (rg * 2048)
#   lr1  = output pointer (pre-offset by -512, auto-advances by 512 each store)
#   lr2  = row-group counter (0..7)
#   lr3  = output channel counter (0..15)
#   lr4  = kernel index (0..127 per kernel half, advances by 1 per mult.ve)
#   lr5  = 512  (output stride)
#   lr6  = 128  (cyclic S1 offset / channel stride)
#   lr7  = 0    (cyclic S0 offset / no-mask constant)
#   lr8  = 256  (cyclic S2 offset)
#   lr9  = 384  (cyclic S3 offset)
#   lr10 = 8    (first-half output channel limit / row-group limit)
#   lr11 = 16   (total output channel limit)
#   lr14 = temp (address computation)

# ===========================================================================
# Initialization
# ===========================================================================

    # Load kernel weights: first half into r0, second half into r1
    # (lr7 defaults to 0 — used as zero offset)
    ldr_mult_reg        r0 lr7 cr1;
    set                 lr5 512;
    set                 lr6 128;;

    ldr_mult_reg        r1 lr6 cr1;
    set                 lr8 256;
    set                 lr9 384;;

    # Load mask data (all zeros — no masking)
    ldr_mult_mask_reg   lr7 cr2;
    set                 lr10 8;
    set                 lr11 16;;

    set                 lr1 -512;;

# ===========================================================================
# Row-group loop (8 row groups, 4 spatial rows each)
# ===========================================================================

row_loop:

    # Load initial 4 input channels (Group A) into r_cyclic S0-S3
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
# Inner loop A: output channels 0-7, kernel from r0 (20 cycles each)
# ---------------------------------------------------------------------------

LOOP_OCH_A:

    # W1: Reset accumulator
    reset_acc;;

    # W2: mult ich0 (S0), kernel[lr4], accumulate
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

    # W7: incr lr4; load ich8->S0; mult ich5 (S1); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve             r0 lr6 lr7 lr7 lr4;
    acc;;

    # W8: incr lr4; load ich9->S1; mult ich6 (S2); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r0 lr8 lr7 lr7 lr4;
    acc;;

    # W9: incr lr4; load ich10->S2; mult ich7 (S3); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r0 lr9 lr7 lr7 lr4;
    acc;;

    # W10: incr lr4; load ich11->S3; mult ich8 (S0); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve             r0 lr7 lr7 lr7 lr4;
    acc;;

    # W11: incr lr4; load ich12->S0; mult ich9 (S1); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve             r0 lr6 lr7 lr7 lr4;
    acc;;

    # W12: incr lr4; load ich13->S1; mult ich10 (S2); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r0 lr8 lr7 lr7 lr4;
    acc;;

    # W13: incr lr4; load ich14->S2; mult ich11 (S3); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r0 lr9 lr7 lr7 lr4;
    acc;;

    # W14: incr lr4; load ich15->S3; mult ich12 (S0); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve             r0 lr7 lr7 lr7 lr4;
    acc;;

    # W15: incr lr4; load ich0->S0 (reload); mult ich13 (S1); acc
    incr                lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    mult.ve             r0 lr6 lr7 lr7 lr4;
    acc;;

    # W16: incr lr4; load ich1->S1 (reload); mult ich14 (S2); acc
    incr                lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r0 lr8 lr7 lr7 lr4;
    acc;;

    # W17: incr lr4; load ich2->S2 (reload); mult ich15 (S3); acc
    incr                lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r0 lr9 lr7 lr7 lr4;
    acc;;

    # W18: incr lr4; load ich3->S3 (reload)
    incr                lr4 1;
    add                 lr14 lr0 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # W19: advance output pointer (+512), store accumulator, incr oc counter
    add                 lr1 lr1 lr5;
    incr                lr3 1;
    str_acc_reg         lr1 cr3;;

    # W20: branch if more output channels in first half
    blt                 lr3 lr10 LOOP_OCH_A;;

# ---------------------------------------------------------------------------
# Transition: reset kernel index for second half
# ---------------------------------------------------------------------------

    set                 lr4 0;;

# ---------------------------------------------------------------------------
# Inner loop B: output channels 8-15, kernel from r1 (20 cycles each)
# ---------------------------------------------------------------------------

LOOP_OCH_B:

    # W1: Reset accumulator
    reset_acc;;

    # W2: mult ich0 (S0), kernel[lr4], accumulate
    mult.ve             r1 lr7 lr7 lr7 lr4;
    acc;;

    # W3: incr lr4; load ich4->S0; mult ich1 (S1); acc
    incr                lr4 1;
    add                 lr14 lr0 lr5;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve             r1 lr6 lr7 lr7 lr4;
    acc;;

    # W4: incr lr4; load ich5->S1; mult ich2 (S2); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r1 lr8 lr7 lr7 lr4;
    acc;;

    # W5: incr lr4; load ich6->S2; mult ich3 (S3); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r1 lr9 lr7 lr7 lr4;
    acc;;

    # W6: incr lr4; load ich7->S3; mult ich4 (S0); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve             r1 lr7 lr7 lr7 lr4;
    acc;;

    # W7: incr lr4; load ich8->S0; mult ich5 (S1); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve             r1 lr6 lr7 lr7 lr4;
    acc;;

    # W8: incr lr4; load ich9->S1; mult ich6 (S2); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r1 lr8 lr7 lr7 lr4;
    acc;;

    # W9: incr lr4; load ich10->S2; mult ich7 (S3); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r1 lr9 lr7 lr7 lr4;
    acc;;

    # W10: incr lr4; load ich11->S3; mult ich8 (S0); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve             r1 lr7 lr7 lr7 lr4;
    acc;;

    # W11: incr lr4; load ich12->S0; mult ich9 (S1); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve             r1 lr6 lr7 lr7 lr4;
    acc;;

    # W12: incr lr4; load ich13->S1; mult ich10 (S2); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r1 lr8 lr7 lr7 lr4;
    acc;;

    # W13: incr lr4; load ich14->S2; mult ich11 (S3); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r1 lr9 lr7 lr7 lr4;
    acc;;

    # W14: incr lr4; load ich15->S3; mult ich12 (S0); acc
    incr                lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve             r1 lr7 lr7 lr7 lr4;
    acc;;

    # W15: incr lr4; load ich0->S0 (reload); mult ich13 (S1); acc
    incr                lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    mult.ve             r1 lr6 lr7 lr7 lr4;
    acc;;

    # W16: incr lr4; load ich1->S1 (reload); mult ich14 (S2); acc
    incr                lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r1 lr8 lr7 lr7 lr4;
    acc;;

    # W17: incr lr4; load ich2->S2 (reload); mult ich15 (S3); acc
    incr                lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r1 lr9 lr7 lr7 lr4;
    acc;;

    # W18: incr lr4; load ich3->S3 (reload)
    incr                lr4 1;
    add                 lr14 lr0 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # W19: advance output pointer (+512), store accumulator, incr oc counter
    add                 lr1 lr1 lr5;
    incr                lr3 1;
    str_acc_reg         lr1 cr3;;

    # W20: branch if more output channels in second half
    blt                 lr3 lr11 LOOP_OCH_B;;

# ---------------------------------------------------------------------------
# Advance to next row group
# ---------------------------------------------------------------------------

    incr                lr0 2048;
    incr                lr2 1;;

    blt                 lr2 lr10 row_loop;;

end:
    bkpt;;
