# Depthwise 3x3 convolution: 160 channels, 8x8 spatial.
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
# Mask groups (reused from standard conv pattern):
#   Group A (offset 0):   for channel A (zero lanes 64-127, bleed slots for kr=+1)
#   Group D (offset 384): for channel B (zero lanes 0-63, bleed slots for kr=-1)
#
# Kernel layout: 20 groups of 128 bytes each.
#   Group g: 8 channel kernels (9 bytes each) = 72 bytes + 56 padding.
#   Within a group, channel c's kernel at offset c*9.
#   Paired processing uses 4 pairs per group (lr3 advances by 18 per pair).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (80 chunks x 128 bytes = 10240 bytes)
#   cr1 = kernel base  (20 groups x 128 bytes = 2560 bytes)
#   cr2 = output base  (80 pairs x 512 bytes = 40960 bytes)
#   cr3 = mask   base  (512 bytes: 4 groups of 128 bytes each)
#
# Register allocation:
#   lr0  = 0     (mask_shift, mask_slot_0)
#   lr1  = 384   (mask group D offset)
#   lr2  = output write offset (0, 512, ..., 40448)
#   lr3  = kernel byte index within r0 (0..71, +18 per pair)
#   lr4  = temp  (cyclic offset for mult.ve)
#   lr5  = 128   (cyclic load index)
#   lr6  = 10240 (total input bytes, outer loop end)
#   lr7  = kernel group offset (0, 128, 256, ..., 2432)
#   lr8  = 1     (mask_slot_1 = kc=-1)
#   lr9  = 2     (mask_slot_2 = kc=+1)
#   lr10 = input chunk offset (0, 128, ..., 10112)
#   lr11 = 72    (pair sub-loop end)
#   lr13 = temp  (mask slot indices 3/4/5 for bleed taps)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 128;
    set                 lr8 1;;

    set                 lr9 2;
    set                 lr11 72;;

    set                 lr6 10240;
    set                 lr1 384;;

    set                 lr7 0;
    set                 lr2 0;;

    set                 lr10 0;;

# ===========================================================================
# Kernel group loop (20 groups of 8 channels = 4 pairs each)
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
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # kr=-1, kc=0 (offset 120, slot 0)
    incr                lr3 1;
    set                 lr4 120;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    # kr=-1, kc=+1 (offset 121, slot 2)
    incr                lr3 1;
    set                 lr4 121;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # kr=0, kc=-1 (offset 127, slot 1)
    incr                lr3 1;
    set                 lr4 127;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # kr=0, kc=0 (offset 128, slot 0)
    incr                lr3 1;
    mult.ve             r0 lr5 lr0 lr0 lr3;
    acc;;

    # kr=0, kc=+1 (offset 129, slot 2)
    incr                lr3 1;
    set                 lr4 129;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # kr=+1, kc=-1 (offset 135, slot 4 -- bleed)
    incr                lr3 1;
    set                 lr4 135;;

    set                 lr13 4;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=+1, kc=0 (offset 136, slot 3 -- bleed)
    incr                lr3 1;
    set                 lr4 136;;

    set                 lr13 3;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=+1, kc=+1 (offset 137, slot 5 -- bleed)
    incr                lr3 1;
    set                 lr4 137;;

    set                 lr13 5;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # ===== Channel B (lanes 64-127, mask group D at offset 384) =====
    incr                lr3 1;
    ldr_mult_mask_reg   lr1 cr3;;

    # kr=-1, kc=-1 (offset 119, slot 4 -- bleed)
    set                 lr4 119;
    set                 lr13 4;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=-1, kc=0 (offset 120, slot 3 -- bleed)
    incr                lr3 1;
    set                 lr4 120;;

    set                 lr13 3;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=-1, kc=+1 (offset 121, slot 5 -- bleed)
    incr                lr3 1;
    set                 lr4 121;;

    set                 lr13 5;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=0, kc=-1 (offset 127, slot 1)
    incr                lr3 1;
    set                 lr4 127;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # kr=0, kc=0 (offset 128, slot 0)
    incr                lr3 1;
    mult.ve             r0 lr5 lr0 lr0 lr3;
    acc;;

    # kr=0, kc=+1 (offset 129, slot 2)
    incr                lr3 1;
    set                 lr4 129;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # kr=+1, kc=-1 (offset 135, slot 1)
    incr                lr3 1;
    set                 lr4 135;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # kr=+1, kc=0 (offset 136, slot 0)
    incr                lr3 1;
    set                 lr4 136;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    # kr=+1, kc=+1 (offset 137, slot 2)
    incr                lr3 1;
    set                 lr4 137;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # Store result, advance pointers
    str_acc_reg         lr2 cr2;
    incr                lr10 128;;

    incr                lr2 512;
    incr                lr3 1;;

    blt                 lr3 lr11 pair_loop;;

    # Next kernel group
    incr                lr7 128;
    blt                 lr10 lr6 kernel_group_process;;

end:
    bkpt;;
