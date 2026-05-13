# Universal Standard 3x3 Convolution: 8x8 spatial, flexible channels.
#
# Paired-filter processing: two output filters share one accumulator.
# Filter f0 (even) accumulates in lanes 0-63, filter f1 (odd) in lanes 64-127.
# One str_acc_reg per pair stores all 512 bytes (both filters valid).
#
# 8x8 spatial fits in 64 bytes (half a 128-byte chunk).
# Two input channels packed per chunk: ch A at bytes 0-63, ch B at bytes 64-127.
# Cyclic register: chunk loaded at index 128.
#   Channel A data at cyclic[128..191], channel B at cyclic[192..255].
#
# f0 (lanes 0-63): ch A cyclic base 128, ch B cyclic base 192
#   Mask groups: A (offset 0), B (offset 128)
# f1 (lanes 64-127): ch A cyclic base 64, ch B cyclic base 128
#   Mask groups: C (offset 256), D (offset 384)
#
# Kernel layout: ceil(in_channels/8) blocks per filter, 128 bytes each.
#   Each block: 8 channels x 9 taps = 72 bytes + 56 padding.
#   Per filter pair: f0 blocks first, then f1 blocks.
#
# Each input chunk provides 2 channels (A and B), consuming 18 kernel bytes.
# After 4 chunks (72 bytes), r0 is exhausted -> reload next block.
#
# CR registers (set by harness):
#   cr0  = input base address
#   cr1  = kernel base address
#   cr2  = output base address
#   cr3  = mask base address
#   cr4  = kernel_bytes_per_filter (= ceil(in_channels/8) * 128)
#   cr5  = total_input_bytes (= in_channels / 2 * 128)
#   cr6  = total_output_bytes (= out_channels / 2 * 512)
#   cr12 = 128  (step constant: chunk / kernel block advance)
#   cr13 = 256  (step constant: output pointer +512 via two adds)
#
# LR registers:
#   lr0  = 0     (mask_shift; also zero constant for add src_a)
#   lr1  = temp  (mask group D offset 384)
#   lr2  = output write offset
#   lr3  = kernel byte index within r0 (0..71)
#   lr4  = temp  (cyclic offset for mult.ve.cyclic)
#   lr5  = 128   (cyclic load index, mask group B offset)
#   lr6  = temp  (kernel block address)
#   lr7  = 72    (r0 reload threshold)
#   lr10 = input chunk offset (0, 128, ...)
#   lr11 = total_input_bytes (copy of cr5, ic loop limit)
#   lr12 = 256   (mask group C offset)
#   lr14 = kernel pair base offset (0, 2*kbpf, 4*kbpf, ...)
#   lr15 = total_output_bytes (copy of cr6, oc pair loop limit)
#
# Mask slots (mult.ve.cyclic immediate mask_offset):
#   0: kc=0   1: kc=-1   2: kc=+1
#   3: kr bleed kc=0   4: kr bleed kc=-1   5: kr bleed kc=+1

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 128;
    set                 lr7 72;;

    set                 lr12 256;
    set                 lr0 0;;

# Copy CR parameters to LR for use in blt
    add                 lr11 lr0 cr5;
    add                 lr15 lr0 cr6;;

    set                 lr14 0;
    set                 lr2 0;;

# ===========================================================================
# Filter pair loop (processes 2 output filters per iteration)
# ===========================================================================

filter_pair_loop:

# ###########################################################################
# Filter f0 (even filter, accumulates in lanes 0-63)
# Cyclic offsets: ch A base 128, ch B base 192
# Mask groups: A (offset 0), B (offset 128)
# ###########################################################################

    # Load f0 first kernel block
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

    # lr6 = current kernel block offset (advances on reload)
    add                 lr6 lr14 lr0;;

# ---------------------------------------------------------------------------
# f0 IC loop: iterate over all input chunks
# ---------------------------------------------------------------------------

f0_ic_loop:
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    # ===== Channel A (cyclic base 128) -- load mask group A =====
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

    # ===== Channel B (cyclic base 192) -- load mask group B =====
    add                 lr3 lr3 1;
    ldr_mult_mask_reg   lr5 cr3;;

    # kr=-1, kc=-1 (offset 183, slot 4 -- bleed)
    set                 lr4 183;
    mult.ve.cyclic      lr4 4 lr0 lr3;
    acc;;

    # kr=-1, kc=0 (offset 184, slot 3 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 184;
    mult.ve.cyclic      lr4 3 lr0 lr3;
    acc;;

    # kr=-1, kc=+1 (offset 185, slot 5 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 185;
    mult.ve.cyclic      lr4 5 lr0 lr3;
    acc;;

    # kr=0, kc=-1 (offset 191, slot 1)
    add                 lr3 lr3 1;
    set                 lr4 191;
    mult.ve.cyclic      lr4 1 lr0 lr3;
    acc;;

    # kr=0, kc=0 (offset 192, slot 0)
    add                 lr3 lr3 1;
    set                 lr4 192;
    mult.ve.cyclic      lr4 0 lr0 lr3;
    acc;;

    # kr=0, kc=+1 (offset 193, slot 2)
    add                 lr3 lr3 1;
    set                 lr4 193;
    mult.ve.cyclic      lr4 2 lr0 lr3;
    acc;;

    # kr=+1, kc=-1 (offset 199, slot 1)
    add                 lr3 lr3 1;
    set                 lr4 199;
    mult.ve.cyclic      lr4 1 lr0 lr3;
    acc;;

    # kr=+1, kc=0 (offset 200, slot 0)
    add                 lr3 lr3 1;
    set                 lr4 200;
    mult.ve.cyclic      lr4 0 lr0 lr3;
    acc;;

    # kr=+1, kc=+1 (offset 201, slot 2)
    add                 lr3 lr3 1;
    set                 lr4 201;
    mult.ve.cyclic      lr4 2 lr0 lr3;
    acc;;

    # Advance to next chunk
    add                 lr3 lr3 1;
    add                 lr10 lr10 cr12;;

    # Check if we exhausted r0 block (lr3 == 72)
    blt                 lr3 lr7 f0_skip_reload;;

    # Reload next kernel block
    add                 lr6 lr6 cr12;;

    ldr_mult_reg        r0 lr6 cr1;
    set                 lr3 0;;

f0_skip_reload:
    blt                 lr10 lr11 f0_ic_loop;;

# ###########################################################################
# Filter f1 (odd filter, accumulates in lanes 64-127)
# Cyclic offsets: ch A base 64, ch B base 128
# Mask groups: C (offset 256 = lr12), D (offset 384)
# ###########################################################################

    # f1 kernel starts at lr14 + kernel_bytes_per_filter
    add                 lr6 lr14 cr4;;

    ldr_mult_reg        r0 lr6 cr1;
    set                 lr3 0;
    set                 lr10 0;;

    # lr1 = 384 (mask group D offset)
    add                 lr1 lr12 lr5;;

# ---------------------------------------------------------------------------
# f1 IC loop: iterate over all input chunks
# ---------------------------------------------------------------------------

f1_ic_loop:
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    # ===== Channel A (cyclic base 64) -- load mask group C =====
    ldr_mult_mask_reg   lr12 cr3;;

    # kr=-1, kc=-1 (offset 55, slot 1)
    set                 lr4 55;
    mult.ve.cyclic      lr4 1 lr0 lr3;
    acc;;

    # kr=-1, kc=0 (offset 56, slot 0)
    add                 lr3 lr3 1;
    set                 lr4 56;
    mult.ve.cyclic      lr4 0 lr0 lr3;
    acc;;

    # kr=-1, kc=+1 (offset 57, slot 2)
    add                 lr3 lr3 1;
    set                 lr4 57;
    mult.ve.cyclic      lr4 2 lr0 lr3;
    acc;;

    # kr=0, kc=-1 (offset 63, slot 1)
    add                 lr3 lr3 1;
    set                 lr4 63;
    mult.ve.cyclic      lr4 1 lr0 lr3;
    acc;;

    # kr=0, kc=0 (offset 64, slot 0)
    add                 lr3 lr3 1;
    set                 lr4 64;
    mult.ve.cyclic      lr4 0 lr0 lr3;
    acc;;

    # kr=0, kc=+1 (offset 65, slot 2)
    add                 lr3 lr3 1;
    set                 lr4 65;
    mult.ve.cyclic      lr4 2 lr0 lr3;
    acc;;

    # kr=+1, kc=-1 (offset 71, slot 4 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 71;
    mult.ve.cyclic      lr4 4 lr0 lr3;
    acc;;

    # kr=+1, kc=0 (offset 72, slot 3 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 72;
    mult.ve.cyclic      lr4 3 lr0 lr3;
    acc;;

    # kr=+1, kc=+1 (offset 73, slot 5 -- bleed)
    add                 lr3 lr3 1;
    set                 lr4 73;
    mult.ve.cyclic      lr4 5 lr0 lr3;
    acc;;

    # ===== Channel B (cyclic base 128) -- load mask group D =====
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

    # Advance to next chunk
    add                 lr3 lr3 1;
    add                 lr10 lr10 cr12;;

    # Check if we exhausted r0 block (lr3 == 72)
    blt                 lr3 lr7 f1_skip_reload;;

    # Reload next kernel block
    add                 lr6 lr6 cr12;;

    ldr_mult_reg        r0 lr6 cr1;
    set                 lr3 0;;

f1_skip_reload:
    blt                 lr10 lr11 f1_ic_loop;;

# ---------------------------------------------------------------------------
# Store result (both filters), advance to next filter pair
# ---------------------------------------------------------------------------

    str_acc_reg         lr2 cr2;;

    # Advance kernel: lr14 += 2 * kernel_bytes_per_filter.
    # Output pointer lr2 += 512 via two sequential add cr13.
    add                 lr14 lr14 cr4;
    add                 lr2 lr2 cr13;;

    add                 lr14 lr14 cr4;
    add                 lr2 lr2 cr13;;

    blt                 lr2 lr15 filter_pair_loop;;

end:
    bkpt;;
