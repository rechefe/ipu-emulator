# Standard 3x3 convolution: 160 input channels -> 160 output channels, 8x8.
#
# Paired-filter processing: two output filters share one accumulator.
# Filter f0 (even) accumulates in lanes 0-63, filter f1 (odd) in lanes 64-127.
# One str_acc_reg per pair stores all 512 bytes (both filters valid).
#
# Uses mask-swapping technique: 2 input channels are packed per 128-byte
# chunk (each channel is 8x8 = 64 bytes).  Both channels accumulate into
# the active lane region using different cyclic offsets.
#
# Four mask groups (128 bytes each) are stored in XMEM:
#   Groups A,B: for f0 (zero lanes 64-127, active lanes 0-63)
#   Groups C,D: for f1 (zero lanes 0-63, active lanes 64-127)
#
# Since the entire spatial plane (8x8=64 bytes) fits in half a chunk,
# there is only 1 row-group -- no group loop needed.
#
# 160 input channels = 80 packed chunks.
# Kernel per filter: 160 * 9 = 1440 bytes -> 20 r0 blocks of 128 bytes each.
#   Block 0:  channels 0-7    (72 bytes + 56 padding)
#   Block 1:  channels 8-15   (72 bytes + 56 padding)
#   ...
#   Block 19: channels 152-159 (72 bytes + 56 padding)
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (80 chunks x 128 bytes = 10240 bytes)
#   cr1 = kernel base  (160 filters x 20 blocks x 128 bytes = 409600 bytes)
#   cr2 = output base  (80 pairs x 512 bytes = 40960 bytes; all lanes valid)
#   cr3 = mask   base  (512 bytes: 4 groups of 128 bytes each)
#
# Cyclic register usage:
#   Load one packed chunk (128B) at cyclic index 128.
#   Channel A data at cyclic[128..191], channel B at cyclic[192..255].
#
#   Filter f0 (lanes 0-63):
#     Channel A (cyclic base 128):
#       kr=-1: 119/120/121   kr=0: 127/128/129   kr=+1: 135/136/137 (bleed)
#     Channel B (cyclic base 192):
#       kr=-1: 183/184/185 (bleed)   kr=0: 191/192/193   kr=+1: 199/200/201
#
#   Filter f1 (lanes 64-127):
#     Channel A (cyclic base 64):
#       kr=-1: 55/56/57   kr=0: 63/64/65   kr=+1: 71/72/73 (bleed)
#     Channel B (cyclic base 128):
#       kr=-1: 119/120/121 (bleed)   kr=0: 127/128/129   kr=+1: 135/136/137
#
# Block loop structure:
#   Each filter has 20 kernel blocks (8 channels each).
#   The chunk loop processes 4 chunks per block (lr3: 0->72).
#   After a block, reload r0 with the next kernel block and continue.
#   lr7 tracks the kernel block address, advancing by 128 per block.
#
# Register allocation:
#   lr0  = 0     (mask_shift, mask_slot_0)
#   lr1  = temp  (address computation); 384 during f1 (group D offset)
#   lr2  = output write offset (0, 512, ..., 40448)
#   lr3  = kernel byte index within r0 (0-71)
#   lr4  = temp  (cyclic offset for mult.ve)
#   lr5  = 128   (cyclic load index, mask group B offset)
#   lr6  = 10240 (total input bytes, block loop end)
#   lr7  = kernel block address (advances by 128 per block)
#   lr8  = 1     (mask_slot_1 = kc=-1)
#   lr9  = 2     (mask_slot_2 = kc=+1)
#   lr10 = chunk address offset (0..10112)
#   lr11 = 72    (kernel taps per block, chunk sub-loop end)
#   lr12 = 256   (mask group C offset)
#   lr13 = temp  (mask slot indices 3/4/5 for bleed taps)
#   lr14 = kernel pair offset (0, 5120, 10240, ..., 404480)
#   lr15 = 40960 (output loop end = 80 pairs x 512)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 128;
    set                 lr8 1;;

    set                 lr9 2;
    set                 lr11 72;;

    set                 lr12 256;
    set                 lr15 20480;;

    set                 lr6 10240;
    add                 lr15 lr15 lr15;;

    set                 lr14 0;
    set                 lr2 0;;

# ===========================================================================
# Filter pair loop (80 pairs of 2 output filters)
# ===========================================================================

filter_pair_loop:
    add                 lr7 lr14 lr0;
    reset_acc;
    set                 lr10 0;;

# ###########################################################################
# Filter f0 (even filter, accumulates in lanes 0-63)
# Cyclic offsets: ch A base 128, ch B base 192
# Mask groups: A (offset 0), B (offset 128)
# 20 kernel blocks, 4 chunks each, with reload loop
# ###########################################################################

f0_process:
    ldr_mult_reg        r0 lr7 cr1;
    set                 lr3 0;;

# ---------------------------------------------------------------------------
# f0 chunk sub-loop (4 chunks per block, 18 taps per chunk)
# ---------------------------------------------------------------------------

f0_chunk_loop:
    # Load packed chunk (2 channels) into cyclic at index 128
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    # ===== Channel A (cyclic base 128) -- load mask group A =====
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

    # ===== Channel B (cyclic base 192) -- load mask group B =====
    incr                lr3 1;
    ldr_mult_mask_reg   lr5 cr3;;

    # kr=-1, kc=-1 (offset 183, slot 4 -- bleed)
    set                 lr4 183;;

    set                 lr13 4;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=-1, kc=0 (offset 184, slot 3 -- bleed)
    incr                lr3 1;
    set                 lr4 184;;

    set                 lr13 3;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=-1, kc=+1 (offset 185, slot 5 -- bleed)
    incr                lr3 1;
    set                 lr4 185;;

    set                 lr13 5;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=0, kc=-1 (offset 191, slot 1)
    incr                lr3 1;
    set                 lr4 191;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # kr=0, kc=0 (offset 192, slot 0)
    incr                lr3 1;
    set                 lr4 192;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    # kr=0, kc=+1 (offset 193, slot 2)
    incr                lr3 1;
    set                 lr4 193;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # kr=+1, kc=-1 (offset 199, slot 1)
    incr                lr3 1;
    set                 lr4 199;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # kr=+1, kc=0 (offset 200, slot 0)
    incr                lr3 1;
    set                 lr4 200;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    # kr=+1, kc=+1 (offset 201, slot 2)
    incr                lr3 1;
    set                 lr4 201;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # Advance to next chunk
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr3 lr11 f0_chunk_loop;;

    # Advance to next kernel block; reload if more blocks remain
    incr                lr7 128;
    blt                 lr10 lr6 f0_process;;

# ###########################################################################
# Filter f1 (odd filter, accumulates in lanes 64-127)
# Cyclic offsets: ch A base 64, ch B base 128
# Mask groups: C (offset 256 = lr12), D (offset 384 = lr12 + lr5)
# lr7 = lr14 + 2560 (naturally from f0's 20 block increments)
# ###########################################################################

    set                 lr10 0;
    add                 lr1 lr12 lr5;;

f1_process:
    ldr_mult_reg        r0 lr7 cr1;
    set                 lr3 0;;

# ---------------------------------------------------------------------------
# f1 chunk sub-loop (4 chunks per block, 18 taps per chunk)
# lr1 = 384 (group D offset) -- preserved through f1 processing
# ---------------------------------------------------------------------------

f1_chunk_loop:
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    # ===== Channel A (cyclic base 64) -- load mask group C =====
    ldr_mult_mask_reg   lr12 cr3;;

    # kr=-1, kc=-1 (offset 55, slot 1)
    set                 lr4 55;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # kr=-1, kc=0 (offset 56, slot 0)
    incr                lr3 1;
    set                 lr4 56;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    # kr=-1, kc=+1 (offset 57, slot 2)
    incr                lr3 1;
    set                 lr4 57;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # kr=0, kc=-1 (offset 63, slot 1)
    incr                lr3 1;
    set                 lr4 63;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # kr=0, kc=0 (offset 64, slot 0)
    incr                lr3 1;
    set                 lr4 64;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    # kr=0, kc=+1 (offset 65, slot 2)
    incr                lr3 1;
    set                 lr4 65;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # kr=+1, kc=-1 (offset 71, slot 4 -- bleed)
    incr                lr3 1;
    set                 lr4 71;;

    set                 lr13 4;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=+1, kc=0 (offset 72, slot 3 -- bleed)
    incr                lr3 1;
    set                 lr4 72;;

    set                 lr13 3;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # kr=+1, kc=+1 (offset 73, slot 5 -- bleed)
    incr                lr3 1;
    set                 lr4 73;;

    set                 lr13 5;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # ===== Channel B (cyclic base 128) -- load mask group D =====
    incr                lr3 1;
    ldr_mult_mask_reg   lr1 cr3;;

    # kr=-1, kc=-1 (offset 119, slot 4 -- bleed)
    set                 lr4 119;;

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

    # Advance to next chunk
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr3 lr11 f1_chunk_loop;;

    # Advance to next kernel block; reload if more blocks remain
    incr                lr7 128;
    blt                 lr10 lr6 f1_process;;

# ---------------------------------------------------------------------------
# Store result (both filters), advance to next filter pair
# ---------------------------------------------------------------------------

    str_acc_reg         lr2 cr2;
    incr                lr14 5120;;

    incr                lr2 512;;

    blt                 lr2 lr15 filter_pair_loop;;

end:
    bkpt;;
