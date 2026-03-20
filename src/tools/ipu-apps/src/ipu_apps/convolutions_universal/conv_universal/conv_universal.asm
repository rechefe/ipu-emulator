# Universal Standard 3x3 Convolution
#
# A single binary that handles ANY valid standard convolution configuration.
# Parameters are passed via CR registers at runtime.
#
# Supported configurations:
#   - Spatial: any rows/cols where cols is power-of-2 in [16..128]
#     and rows*cols >= 256 (at least 2 chunks)
#   - in_channels: must be a multiple of 8, >= 8
#   - out_channels: >= 1
#
# The cyclic register holds 3 neighboring chunks:
#   S0 (index 0):   previous chunk
#   S1 (index 128): current chunk
#   S2 (index 256): next chunk
#
# Vertical neighbor access via universal formula:
#   kr=-1 base = 128 - cols
#   kr= 0 base = 128
#   kr=+1 base = 128 + cols
# Horizontal shift: kc offset of -1 or +1 added to base.
#
# CR register parameters (set by harness):
#   cr0 = input base address
#   cr1 = kernel base address
#   cr2 = output base address
#   cr3 = mask base address
#   cr4 = cols (spatial width)
#   cr5 = num_chunks (= rows * cols / 128)
#   cr6 = in_group_stride (= in_channels * 128)
#   cr7 = 1024 (channel group size = 8 * 128, constant)
#   cr8 = total_kernel_bytes (= out_channels * (in_channels/8) * 128)
#
# Mask slots (precomputed by harness, depend on cols):
#   slot 0: all zeros          -> no masking (kc=0)
#   slot 1: left border        -> zero col 0 of each packed row (kc=-1)
#   slot 2: right border       -> zero last col of each packed row (kc=+1)
#   slot 3: bottom row         -> zero last spatial row in chunk
#   slot 4: left + bottom      -> union of slot 1 and slot 3
#   slot 5: right + bottom     -> union of slot 2 and slot 3
#
# LR register allocation:
#   lr0  = 0     (zero constant, mask slot 0, mask_shift, S0 cyclic index)
#   lr1  = 1     (mask slot 1 = left border, kc offset)
#   lr2  = 2     (mask slot 2 = right border)
#   lr3  = 128 - cols  (kr=-1 cyclic base)
#   lr4  = 128   (kr=0 cyclic base, S1 cyclic index, channel stride)
#   lr5  = 128 + cols  (kr=+1 cyclic base)
#   lr6  = kernel byte index within r0 (0..71 per r0 load)
#   lr7  = output pointer (global, continuous)
#   lr8  = chunk base address (chunk_index * in_group_stride)
#   lr9  = chunk counter
#   lr10 = channel offset within chunk (0, 128, ..., in_group_stride-128)
#   lr11 = channel group limit (lr10 + 1024)
#   lr12 = kernel memory offset (0, 128, ...; reset to 0 per chunk section)
#   lr13 = total_kernel_bytes (filter loop limit, copy of cr8)
#   lr14 = temp
#   lr15 = temp / chunk loop limit / bottom border mask slot
#
# Note: in_group_stride (cr6) is accessed directly via CR in add/sub ops.
# The filter loop is: blt lr12 lr13 (where lr13 = cr8, set once per section).

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr0 0;
    ldr_mult_mask_reg   lr0 cr3;;

    set                 lr4 128;
    set                 lr1 1;;

    set                 lr2 2;
    sub                 lr3 lr4 cr4;;

    add                 lr5 lr4 cr4;;

# ===========================================================================
# Section 1: Chunk 0 (top border)
# S0 = zeros (cyclic register initialized to 0), load S1 and S2 only.
# kr=-1 taps read from S0 = zeros -> automatic zero-padding.
# ===========================================================================

    set                 lr8 0;
    set                 lr7 0;;

    set                 lr12 0;
    add                 lr13 cr8 lr0;;

g0_filter_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    set                 lr10 0;
    reset_acc;;

    add                 lr11 lr10 cr7;;

g0_ch_loop:
    # Compute S1 address = chunk_base + channel_offset
    add                 lr14 lr8 lr10;;

    # Load S1 (current chunk, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

    # Load S2 (next chunk, this channel) at cyclic index 256
    add                 lr15 lr14 cr6;
    add                 lr14 lr4 lr4;;

    ldr_cyclic_mult_reg lr15 cr0 lr14;;

    # --- kr=-1: cyclic base lr3 (128-cols), masks 1/0/2 ---
    sub                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    incr                lr6 1;
    mult.ve             r0 lr3 lr0 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- kr=0: cyclic base lr4 (128), masks 1/0/2 ---
    incr                lr6 1;
    sub                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    incr                lr6 1;
    mult.ve             r0 lr4 lr0 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- kr=+1: cyclic base lr5 (128+cols), masks 1/0/2 ---
    incr                lr6 1;
    sub                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    incr                lr6 1;
    mult.ve             r0 lr5 lr0 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # Advance to next input channel
    incr                lr6 1;
    incr                lr10 128;;

    blt                 lr10 lr11 g0_ch_loop;;

    # Advance kernel offset; check for more channel groups
    incr                lr12 128;
    add                 lr15 cr6 lr0;;

    blt                 lr10 lr15 g0_reload;;

    # All input channels done -- store output and advance filter
    str_acc_reg         lr7 cr2;;

    incr                lr7 512;;

    blt                 lr12 lr13 g0_filter_loop;;

    b                   main_setup;;

g0_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr11 lr10 cr7;;

    b                   g0_ch_loop;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop)
# Load all 3 chunks (S0, S1, S2). All 9 taps with normal masks.
# ===========================================================================

main_setup:
    add                 lr8 cr6 lr0;
    set                 lr9 1;;

    sub                 lr15 cr5 lr1;;

row_loop:
    set                 lr12 0;;

filter_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    set                 lr10 0;
    reset_acc;;

    add                 lr11 lr10 cr7;;

ch_loop:
    # Compute S1 address = chunk_base + channel_offset
    add                 lr14 lr8 lr10;;

    # Load S0 (prev chunk) at cyclic index 0
    sub                 lr14 lr14 cr6;
    ldr_cyclic_mult_reg lr14 cr0 lr0;;

    # Recompute S1 address and load at cyclic index 128
    add                 lr14 lr8 lr10;
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

    # Load S2 (next chunk) at cyclic index 256
    add                 lr14 lr14 cr6;
    add                 lr15 lr4 lr4;;

    # Note: lr15 trashed here (=256), restored after all filters done
    ldr_cyclic_mult_reg lr14 cr0 lr15;;

    # --- kr=-1: cyclic base lr3 (128-cols), masks 1/0/2 ---
    sub                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    incr                lr6 1;
    mult.ve             r0 lr3 lr0 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- kr=0: cyclic base lr4 (128), masks 1/0/2 ---
    incr                lr6 1;
    sub                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    incr                lr6 1;
    mult.ve             r0 lr4 lr0 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- kr=+1: cyclic base lr5 (128+cols), masks 1/0/2 ---
    incr                lr6 1;
    sub                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    incr                lr6 1;
    mult.ve             r0 lr5 lr0 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr5 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # Advance to next input channel
    incr                lr6 1;
    incr                lr10 128;;

    blt                 lr10 lr11 ch_loop;;

    # Advance kernel offset; check for more channel groups
    incr                lr12 128;
    add                 lr15 cr6 lr0;;

    blt                 lr10 lr15 reload;;

    # All input channels done -- store and advance output filter
    str_acc_reg         lr7 cr2;;

    incr                lr7 512;;

    blt                 lr12 lr13 filter_loop;;

    # All filters done for this chunk -- advance to next chunk
    add                 lr8 lr8 cr6;
    incr                lr9 1;;

    # Restore lr15 = chunk limit (trashed during S2 index computation)
    sub                 lr15 cr5 lr1;;

    blt                 lr9 lr15 row_loop;;

    b                   gN_section;;

reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr11 lr10 cr7;;

    b                   ch_loop;;

# ===========================================================================
# Section 3: Last chunk (bottom border)
# Load S0 and S1 only (skip S2). Use mask slots 3/4/5 for kr=+1 taps.
# ===========================================================================

gN_section:
    set                 lr12 0;;

gN_filter_loop:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    set                 lr10 0;
    reset_acc;;

    add                 lr11 lr10 cr7;;

gN_ch_loop:
    # Compute S1 address = chunk_base + channel_offset
    add                 lr14 lr8 lr10;;

    # Load S0 (prev chunk) at cyclic index 0
    sub                 lr15 lr14 cr6;
    ldr_cyclic_mult_reg lr15 cr0 lr0;;

    # Load S1 (current chunk) at cyclic index 128
    ldr_cyclic_mult_reg lr14 cr0 lr4;;

    # (skip S2 -- stale data will be masked by slots 3/4/5)

    # --- kr=-1: cyclic base lr3 (128-cols), normal masks 1/0/2 ---
    sub                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    incr                lr6 1;
    mult.ve             r0 lr3 lr0 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr3 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- kr=0: cyclic base lr4 (128), normal masks 1/0/2 ---
    incr                lr6 1;
    sub                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr1 lr0 lr6;
    acc;;

    incr                lr6 1;
    mult.ve             r0 lr4 lr0 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr4 lr1;
    mult.ve             r0 lr14 lr2 lr0 lr6;
    acc;;

    # --- kr=+1: cyclic base lr5 (128+cols), BOTTOM BORDER masks 4/3/5 ---
    incr                lr6 1;
    sub                 lr14 lr5 lr1;;

    set                 lr15 4;
    mult.ve             r0 lr14 lr15 lr0 lr6;
    acc;;

    incr                lr6 1;
    set                 lr15 3;;

    mult.ve             r0 lr5 lr15 lr0 lr6;
    acc;;

    incr                lr6 1;
    add                 lr14 lr5 lr1;;

    set                 lr15 5;
    mult.ve             r0 lr14 lr15 lr0 lr6;
    acc;;

    # Advance to next input channel
    incr                lr6 1;
    incr                lr10 128;;

    blt                 lr10 lr11 gN_ch_loop;;

    # Advance kernel offset; check for more channel groups
    incr                lr12 128;
    add                 lr15 cr6 lr0;;

    blt                 lr10 lr15 gN_reload;;

    # All input channels done -- store and advance
    str_acc_reg         lr7 cr2;;

    incr                lr7 512;;

    blt                 lr12 lr13 gN_filter_loop;;

end:
    bkpt;;

gN_reload:
    ldr_mult_reg        r0 lr12 cr1;
    set                 lr6 0;;

    add                 lr11 lr10 cr7;;

    b                   gN_ch_loop;;
