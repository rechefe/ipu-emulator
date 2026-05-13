# Universal Depthwise 3x3 Convolution: 8x8 spatial, flexible channels.
#
# Each channel has its own independent 3x3 kernel (no cross-channel mixing).
# Paired-channel processing: channels 2k and 2k+1 share one accumulator.
# Channel A (even) accumulates in lanes 0-63, channel B (odd) in lanes 64-127.
# One str_acc_reg per pair stores all 512 bytes (both channels valid).
#
# 8x8 spatial fits in 64 bytes (half a 128-byte chunk).
# Two input channels packed per chunk: ch A at bytes 0-63, ch B at bytes 64-127.
# Cyclic register: chunk loaded at index 128.
#   Channel A data at cyclic[128..191], channel B at cyclic[192..255].
#
# Mask groups (4 groups x 128 bytes = 512 bytes):
#   Group A (offset 0):   for channel A (zero lanes 64-127, bleed for kr=+1)
#   Group D (offset 384): for channel B (zero lanes 0-63, bleed for kr=-1)
#
# Kernel layout: ceil(num_channels/8) groups of 128 bytes each.
#   Group g: 8 channel kernels (9 bytes each) = 72 bytes + 56 padding.
#   Paired processing uses 4 pairs per group (lr3 advances by 18 per pair).
#
# CR registers (set by harness):
#   cr0  = input base address
#   cr1  = kernel base address
#   cr2  = output base address
#   cr3  = mask base address
#   cr4  = total_input_bytes (= num_channels / 2 * 128)
#   cr12 = 128  (step constant: chunk / kernel group advance)
#   cr13 = 256  (step constant: output pointer +512 via two adds)
#
# LR registers:
#   lr0  = 0     (mask_shift; also zero constant for add src_a)
#   lr1  = 384   (mask group D offset)
#   lr2  = output write offset (0, 512, ...)
#   lr3  = kernel byte index within r0 (0..71, +18 per pair)
#   lr4  = temp  (cyclic offset for mult.ve.cyclic)
#   lr5  = 128   (cyclic load index)
#   lr7  = kernel group offset (0, 128, 256, ...)
#   lr10 = input chunk offset (0, 128, ...)
#   lr11 = 72    (pair sub-loop end: 4 pairs x 18 bytes = 72)
#   lr15 = total_input_bytes (copy of cr4, outer loop limit)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 128;
    set                 lr11 72;;

    set                 lr1 384;
    set                 lr0 0;;

# Copy CR parameter to LR for use in blt
    add                 lr15 lr0 cr4;;

    set                 lr7 0;
    set                 lr2 0;;

    set                 lr10 0;;

# ===========================================================================
# Kernel group loop (ceil(num_channels/8) groups, 4 pairs each)
# ===========================================================================

kernel_group_process:
    ldr_mult_reg        r0 lr7 cr1;
    set                 lr3 0;;

# ---------------------------------------------------------------------------
# Pair loop (4 pairs per kernel group)
# ---------------------------------------------------------------------------

pair_loop:
    reset_acc;
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    # ===== Channel A (lanes 0-63, mask group A at offset 0) =====
    ldr_mult_mask_reg   lr0 cr3;;

    # kr=-1, kc=-1 (offset 119, slot 1)
    set                 lr4 119;
    mult.ve.cyclic      lr4 1 lr0 lr3;
    acc;;

    # kr=-1, kc=0 (offset 120, slot 0)
    add                 lr3 lr3 1;
    set                 lr4 120;
    mult.ve.cyclic      lr4 0 lr0 lr3;
    acc;;

    # kr=-1, kc=+1 (offset 121, slot 2)
    add                 lr3 lr3 1;
    set                 lr4 121;
    mult.ve.cyclic      lr4 2 lr0 lr3;
    acc;;

    # kr=0, kc=-1 (offset 127, slot 1)
    add                 lr3 lr3 1;
    set                 lr4 127;
    mult.ve.cyclic      lr4 1 lr0 lr3;
    acc;;

    # kr=0, kc=0 (offset 128, slot 0)
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr5 0 lr0 lr3;
    acc;;

    # kr=0, kc=+1 (offset 129, slot 2)
    add                 lr3 lr3 1;
    set                 lr4 129;
    mult.ve.cyclic      lr4 2 lr0 lr3;
    acc;;

    # kr=+1, kc=-1 (offset 135, slot 4 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 135;
    mult.ve.cyclic      lr4 4 lr0 lr3;
    acc;;

    # kr=+1, kc=0 (offset 136, slot 3 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 136;
    mult.ve.cyclic      lr4 3 lr0 lr3;
    acc;;

    # kr=+1, kc=+1 (offset 137, slot 5 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 137;
    mult.ve.cyclic      lr4 5 lr0 lr3;
    acc;;

    # ===== Channel B (lanes 64-127, mask group D at offset 384) =====
    add                 lr3 lr3 1;
    ldr_mult_mask_reg   lr1 cr3;;

    # kr=-1, kc=-1 (offset 119, slot 4 -- bleed)
    set                 lr4 119;
    mult.ve.cyclic      lr4 4 lr0 lr3;
    acc;;

    # kr=-1, kc=0 (offset 120, slot 3 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 120;
    mult.ve.cyclic      lr4 3 lr0 lr3;
    acc;;

    # kr=-1, kc=+1 (offset 121, slot 5 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 121;
    mult.ve.cyclic      lr4 5 lr0 lr3;
    acc;;

    # kr=0, kc=-1 (offset 127, slot 1)
    add                 lr3 lr3 1;
    set                 lr4 127;
    mult.ve.cyclic      lr4 1 lr0 lr3;
    acc;;

    # kr=0, kc=0 (offset 128, slot 0)
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr5 0 lr0 lr3;
    acc;;

    # kr=0, kc=+1 (offset 129, slot 2)
    add                 lr3 lr3 1;
    set                 lr4 129;
    mult.ve.cyclic      lr4 2 lr0 lr3;
    acc;;

    # kr=+1, kc=-1 (offset 135, slot 1)
    add                 lr3 lr3 1;
    set                 lr4 135;
    mult.ve.cyclic      lr4 1 lr0 lr3;
    acc;;

    # kr=+1, kc=0 (offset 136, slot 0)
    add                 lr3 lr3 1;
    set                 lr4 136;
    mult.ve.cyclic      lr4 0 lr0 lr3;
    acc;;

    # kr=+1, kc=+1 (offset 137, slot 2)
    add                 lr3 lr3 1;
    set                 lr4 137;
    mult.ve.cyclic      lr4 2 lr0 lr3;
    acc;;

    # Store result, advance pointers
    str_acc_reg         lr2 cr2;
    add                 lr10 lr10 cr12;;

    add                 lr2 lr2 cr13;
    add                 lr3 lr3 1;;

    add                 lr2 lr2 cr13;;

    blt                 lr3 lr11 pair_loop;;

    # Next kernel group
    add                 lr7 lr7 cr12;
    blt                 lr10 lr15 kernel_group_process;;

end:
    bkpt;;
