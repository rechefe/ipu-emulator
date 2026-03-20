# Standard 3x3 convolution: 16 input channels -> 16 output channels, 8x8.
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
# 16 input channels = 8 packed chunks.
# Kernel per filter: 16 * 9 = 144 bytes -> 2 r0 blocks of 128 bytes each.
#   Block 0: channels 0-7  (72 bytes + 56 padding)
#   Block 1: channels 8-15 (72 bytes + 56 padding)
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (8 chunks x 128 bytes = 1024 bytes)
#   cr1 = kernel base  (16 filters x 2 blocks x 128 bytes = 4096 bytes)
#   cr2 = output base  (8 pairs x 512 bytes = 4096 bytes; all lanes valid)
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
# Mask groups in XMEM (each 128 bytes = 8 slots of 16 bytes = 128 bits):
#
#   Group A (offset 0, f0 channel A, active lanes 0-63):
#     slot 0: {64-127}                              (kc=0)
#     slot 1: {64-127} + {0,8,16,24,32,40,48,56}   (kc=-1)
#     slot 2: {64-127} + {7,15,23,31,39,47,55,63}  (kc=+1)
#     slot 3: {56-127}                              (kr=+1 bleed)
#     slot 4: {56-127} + {0,8,16,24,32,40,48}      (kr=+1 bleed, kc=-1)
#     slot 5: {56-127} + {7,15,23,31,39,47,55}     (kr=+1 bleed, kc=+1)
#
#   Group B (offset 128, f0 channel B, active lanes 0-63):
#     slot 0-2: same as group A
#     slot 3: {0-7} + {64-127}                      (kr=-1 bleed)
#     slot 4: {0-7} + {8,16,24,32,40,48,56} + {64-127}
#     slot 5: {0-7} + {15,23,31,39,47,55,63} + {64-127}
#
#   Group C (offset 256, f1 channel A, active lanes 64-127):
#     slot 0: {0-63}                                  (kc=0)
#     slot 1: {0-63} + {64,72,80,88,96,104,112,120}  (kc=-1)
#     slot 2: {0-63} + {71,79,87,95,103,111,119,127} (kc=+1)
#     slot 3: {0-63} + {120-127}                      (kr=+1 bleed)
#     slot 4: {0-63} + {120-127} + {64,72,80,88,96,104,112}
#     slot 5: {0-63} + {120-127} + {71,79,87,95,103,111,119}
#
#   Group D (offset 384, f1 channel B, active lanes 64-127):
#     slot 0-2: same as group C
#     slot 3: {0-71}                                  (kr=-1 bleed)
#     slot 4: {0-71} + {72,80,88,96,104,112,120}
#     slot 5: {0-71} + {79,87,95,103,111,119,127}
#
# Register allocation:
#   lr0  = 0     (mask_shift, mask_slot_0)
#   lr1  = temp  (address computation, group D offset)
#   lr2  = output write offset (0, 512, ..., 3584)
#   lr3  = kernel byte index within r0 (0-71)
#   lr4  = temp  (cyclic offset for mult.ve)
#   lr5  = 128   (cyclic load index, mask group B offset)
#   lr6  = temp  (kernel block address)
#   lr7  = chunk loop end (1024 for block 1)
#   lr8  = 1     (mask_slot_1 = kc=-1)
#   lr9  = 2     (mask_slot_2 = kc=+1)
#   lr10 = chunk address offset (0..896)
#   lr11 = 72    (kernel taps per block, chunk sub-loop end)
#   lr12 = 256   (mask group C offset)
#   lr13 = temp  (mask slot indices 3/4/5 for bleed taps)
#   lr14 = kernel pair offset (0, 512, 1024, ..., 3584)
#   lr15 = 4096  (filter pair loop end = 8 * 512)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 128;
    set                 lr8 1;;

    set                 lr9 2;
    set                 lr11 72;;

    set                 lr12 256;
    set                 lr15 4096;;

    set                 lr14 0;
    set                 lr2 0;;

# ===========================================================================
# Filter pair loop (8 pairs of 2 output filters)
# ===========================================================================

filter_pair_loop:

# ###########################################################################
# Filter f0 (even filter, accumulates in lanes 0-63)
# Cyclic offsets: ch A base 128, ch B base 192
# Mask groups: A (offset 0), B (offset 128)
# ###########################################################################

    # Load f0 r0 block 0 (channels 0-7 kernel)
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

# ---------------------------------------------------------------------------
# f0 chunk sub-loop: block 0 (chunks 0-3, input channels 0-7)
# ---------------------------------------------------------------------------

f0_chunk_loop_b0:
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

    blt                 lr3 lr11 f0_chunk_loop_b0;;

# ---------------------------------------------------------------------------
# f0: Load r0 block 1 (channels 8-15 kernel) and process chunks 4-7
# ---------------------------------------------------------------------------

    add                 lr6 lr14 lr5;
    ldr_mult_reg        r0 lr6 cr1;
    set                 lr3 0;;

    set                 lr7 1024;;

f0_chunk_loop_b1:
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

    blt                 lr10 lr7 f0_chunk_loop_b1;;

# ###########################################################################
# Filter f1 (odd filter, accumulates in lanes 64-127)
# Cyclic offsets: ch A base 64, ch B base 128
# Mask groups: C (offset 256 = lr12), D (offset 384 = lr12 + lr5)
# ###########################################################################

    # Load f1 r0 block 0 (kernel at lr14 + 256)
    add                 lr6 lr14 lr12;
    set                 lr3 0;;

    ldr_mult_reg        r0 lr6 cr1;
    set                 lr10 0;
    add                 lr1 lr12 lr5;;

# ---------------------------------------------------------------------------
# f1 chunk sub-loop: block 0 (chunks 0-3, input channels 0-7)
# lr1 = 384 (group D offset) -- preserved through f1 processing
# ---------------------------------------------------------------------------

f1_chunk_loop_b0:
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

    blt                 lr3 lr11 f1_chunk_loop_b0;;

# ---------------------------------------------------------------------------
# f1: Load r0 block 1 (channels 8-15 kernel) and process chunks 4-7
# ---------------------------------------------------------------------------

    # f1 block 1 kernel at lr14 + 384 = lr14 + lr1
    add                 lr6 lr14 lr1;
    ldr_mult_reg        r0 lr6 cr1;
    set                 lr3 0;;

f1_chunk_loop_b1:
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

    blt                 lr10 lr7 f1_chunk_loop_b1;;

# ---------------------------------------------------------------------------
# Store result (both filters), advance to next filter pair
# ---------------------------------------------------------------------------

    str_acc_reg         lr2 cr2;
    incr                lr14 512;;

    incr                lr2 512;;

    blt                 lr14 lr15 filter_pair_loop;;

end:
    bkpt;;
