# Pointwise convolution: 128x128x16 input, 1x1x16x64 kernel, 128x128x64 output.
#
# 1x1 convolution with 16 input channels and 64 output channels.
# No spatial kernel — pure channel mixing (pointwise).
#
# Since spatial width (128) = SIMD width (128), 1 spatial row is packed
# per 128-byte chunk (128 / 128 = 1 row per chunk).
# The outer loop iterates over 128 row groups (128 rows / 1 per group).
#
# Kernel has 16x64 = 1024 weights, requiring multiple mult register reloads.
# Each mult register (r0, r1) holds 128 bytes = 8 output channels x 16 inputs.
# Solution: kernel_group_loop iterates 4 times, each time loading a new pair
# of r0/r1 (256 bytes) to cover 16 output channels per group.
#   Group 0: r0=kernel[0..127] (oc 0-7), r1=kernel[128..255] (oc 8-15)
#   Group 1: r0=kernel[256..383] (oc 16-23), r1=kernel[384..511] (oc 24-31)
#   Group 2: r0=kernel[512..639] (oc 32-39), r1=kernel[640..767] (oc 40-47)
#   Group 3: r0=kernel[768..895] (oc 48-55), r1=kernel[896..1023] (oc 56-63)
#
# Pipelining strategy:
#   r_cyclic has 4 slots (S0-S3, 128 bytes each, 512 total).
#   16 input channels require loading in four groups:
#     Group A (ich0-3): loaded into S0-S3 before the inner loop
#     Groups B-D (ich4-15): loaded during computation (pipelined)
#   After mult.ve reads a slot, the next cycle loads new data into that slot.
#   This hides all load latency, achieving 100% SIMD utilization.
#   Each output channel requires 20 VLIW words (16 mult+acc + pipeline overhead).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (128 row-groups x 16 ch x 128 bytes = 262144 bytes)
#   cr1 = kernel base  (1024 bytes, split into 4 groups of 256 bytes)
#   cr2 = mask   base  (128 bytes: all zeros, no masking needed)
#   cr3 = output base  (128 row-groups x 64 ch x 512 bytes = 4194304 bytes)
#
# Input interleaving (per row group of 1 row):
#   rg0_ch0(128B), rg0_ch1(128B), ..., rg0_ch15(128B), rg1_ch0, ...
#   Channel ic, row-group rg: offset = rg * 2048 + ic * 128
#
# Output interleaving (per row group):
#   rg0_outch0(512B), rg0_outch1(512B), ..., rg0_outch63(512B), rg1_outch0, ...
#   Output channel oc, row-group rg: offset = (rg * 64 + oc) * 512
#
# Kernel layout:
#   kernel[oc * 16 + ic] for oc=0..63, ic=0..15
#   Within each r register (128 bytes): 8 output channels x 16 input channels
#
# Register allocation:
#   lr0  = input row-group base address (rg * 2048)
#   lr1  = output pointer (pre-offset by -512, auto-advances by 512 each store)
#   lr2  = row-group counter (0..127)
#   lr3  = output channel counter (0..63)
#   lr4  = kernel index (0..127 per register, advances by 1 per mult.ve)
#   lr5  = 512  (output stride)
#   lr6  = 128  (cyclic S1 offset / channel stride / row-group limit)
#   lr7  = 0    (cyclic S0 offset / no-mask constant)
#   lr8  = 256  (cyclic S2 offset / kernel group stride)
#   lr9  = 384  (cyclic S3 offset)
#   lr10 = 8    (output channels per kernel half)
#   lr12 = kernel memory offset (0, 256, 512, 768 — advances per kernel group)
#   lr13 = inner loop output channel limit (lr3 + lr10, computed dynamically)
#   lr14 = temp (address computation)
#   lr15 = 64   (total output channel limit)

# ===========================================================================
# Initialization
# ===========================================================================

    # Load mask data (all zeros — no masking)
    # (lr7 defaults to 0 — used as zero offset)
    ldr_mult_mask_reg   lr7 cr2;
    set                 lr5 512;
    set                 lr6 128;;

    set                 lr8 256;
    set                 lr9 384;;

    set                 lr10 8;
    set                 lr15 64;;

    set                 lr1 -512;;

# ===========================================================================
# Row-group loop (128 row groups, 1 spatial row each)
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
    set                 lr12 0;;

# ===========================================================================
# Kernel-group loop (4 groups of 16 output channels each)
# ===========================================================================

kernel_group_loop:

    # Load kernel pair: r0 = kernel[lr12..lr12+127], r1 = kernel[lr12+128..lr12+255]
    ldr_mult_reg        r0 lr12 cr1;;

    add                 lr14 lr12 lr6;
    ldr_mult_reg        r1 lr14 cr1;;

    set                 lr4 0;
    add                 lr13 lr3 lr10;;

# ---------------------------------------------------------------------------
# Inner loop A: 8 output channels from r0 (20 cycles each)
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

    # W15: incr lr4; reload ich0->S0; mult ich13 (S1); acc
    incr                lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    mult.ve             r0 lr6 lr7 lr7 lr4;
    acc;;

    # W16: incr lr4; reload ich1->S1; mult ich14 (S2); acc
    incr                lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r0 lr8 lr7 lr7 lr4;
    acc;;

    # W17: incr lr4; reload ich2->S2; mult ich15 (S3); acc
    incr                lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r0 lr9 lr7 lr7 lr4;
    acc;;

    # W18: incr lr4; reload ich3->S3
    incr                lr4 1;
    add                 lr14 lr0 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # W19: advance output pointer (+512), store accumulator, incr oc counter
    add                 lr1 lr1 lr5;
    incr                lr3 1;
    str_acc_reg         lr1 cr3;;

    # W20: branch if more output channels in first half
    blt                 lr3 lr13 LOOP_OCH_A;;

# ---------------------------------------------------------------------------
# Transition: reset kernel index, compute new limit for second half
# ---------------------------------------------------------------------------

    set                 lr4 0;
    add                 lr13 lr3 lr10;;

# ---------------------------------------------------------------------------
# Inner loop B: 8 output channels from r1 (20 cycles each)
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

    # W15: incr lr4; reload ich0->S0; mult ich13 (S1); acc
    incr                lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    mult.ve             r1 lr6 lr7 lr7 lr4;
    acc;;

    # W16: incr lr4; reload ich1->S1; mult ich14 (S2); acc
    incr                lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve             r1 lr8 lr7 lr7 lr4;
    acc;;

    # W17: incr lr4; reload ich2->S2; mult ich15 (S3); acc
    incr                lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve             r1 lr9 lr7 lr7 lr4;
    acc;;

    # W18: incr lr4; reload ich3->S3
    incr                lr4 1;
    add                 lr14 lr0 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # W19: advance output pointer (+512), store accumulator, incr oc counter
    add                 lr1 lr1 lr5;
    incr                lr3 1;
    str_acc_reg         lr1 cr3;;

    # W20: branch if more output channels in second half
    blt                 lr3 lr13 LOOP_OCH_B;;

# ---------------------------------------------------------------------------
# Advance to next kernel group (next 16 output channels)
# ---------------------------------------------------------------------------

    add                 lr12 lr12 lr8;
    blt                 lr3 lr15 kernel_group_loop;;

# ---------------------------------------------------------------------------
# Advance to next row group
# ---------------------------------------------------------------------------

    incr                lr0 2048;
    incr                lr2 1;;

    blt                 lr2 lr6 row_loop;;

end:
    bkpt;;
