# Standard 3x3 convolution: 8 input channels -> 16 output channels, 64x64.
#
# Since spatial width (64) < SIMD width (128), 2 spatial rows are packed
# per 128-byte chunk.  32 row-groups total.  The cyclic register holds 3
# neighboring chunks (S0=group_{g-1}, S1=group_g, S2=group_{g+1}) so
# that vertical neighbor access is a simple offset into the cyclic register.
#
# Each output channel has 8 input-channel 3x3 kernels (full cross-channel
# mixing).  Kernel per filter: 8 * 9 = 72 bytes, fits in r0 (128 bytes).
# r0 is reloaded per output filter.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (32 groups x 8 ch x 128 bytes = 32768 bytes)
#   cr1 = kernel base  (16 filters x 128 bytes each = 2048 bytes)
#   cr2 = output base  (32 groups x 16 out_ch x 512 bytes = 262144 bytes)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Input layout (interleaved by group, 8 channels):
#   group g, channel ch, local_row lr, col c:
#     offset = (g * 8 + ch) * 128 + lr * 64 + c
#   Group stride = 8 * 128 = 1024 bytes
#
# Output layout (interleaved by group, 32-bit accumulators):
#   group g, filter f: offset = (g * 16 + f) * 512
#
# Kernel layout per filter f (at XMEM offset f * 128 from kernel base):
#   For input channel ch (0..7):
#     byte[ch*9 + dr*3 + dc]   (72 bytes data, padded to 128)
#
# Cyclic register usage (3 chunks loaded per input channel):
#   S0 at index 0:    group_{g-1}, same channel
#   S1 at index 128:  group_g,     same channel
#   S2 at index 256:  group_{g+1}, same channel
#
#   kr=-1: cyclic offset 63/64/65     (kc=-1/0/+1)
#   kr= 0: cyclic offset 127/128/129  (kc=-1/0/+1)
#   kr=+1: cyclic offset 191/192/193  (kc=-1/0/+1)
#
# Mask slots in r_mask:
#   slot 0: all zeros          -> no masking               (kc = 0)
#   slot 1: bits {0, 64} set   -> zero col 0 of each row   (kc = -1)
#   slot 2: bits {63, 127} set -> zero col 63 of each row  (kc = +1)
#   slot 3: bits {64..127}     -> zero bottom row           (last group kr=+1 kc=0)
#   slot 4: bits {0, 64..127}  -> left border + bottom row  (last group kr=+1 kc=-1)
#   slot 5: bits {63..127}     -> right border + bottom row (last group kr=+1 kc=+1)
#
# Register allocation:
#   lr0  = 0     (mask_shift, mask_slot_0, cyclic index for S0)
#   lr1  = temp  (S0/S2 address computation)
#   lr2  = output write offset
#   lr3  = kernel byte index (0-71 per filter)
#   lr4  = temp  (cyclic offset for mult.ve)
#   lr5  = 128   (cyclic index for S1, chunk stride)
#   lr6  = 256   (cyclic index for S2)
#   lr7  = group base offset (g * 1024)
#   lr8  = 1     (mask_slot_1 = kc=-1)
#   lr9  = 2     (mask_slot_2 = kc=+1)
#   lr10 = channel offset counter (0, 128, ..., 896)
#   lr11 = 1024  (group stride = 8 * 128, channel loop end)
#   lr12 = temp  (S1 address)
#   lr13 = 31744 (group loop end = 31 * 1024) / repurposed for mask slots
#   lr14 = filter kernel offset (0, 128, ..., 1920)
#   lr15 = 2048  (filter loop end = 16 * 128)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 128;
    set                 lr6 256;;

    ldr_mult_mask_reg   lr0 cr3;
    set                 lr8 1;;

    set                 lr9 2;
    set                 lr11 1024;;

    set                 lr15 2048;;

# ===========================================================================
# Group 0 (rows 0-1) -- top border
# S0 = zeros (cyclic reg initialised to 0), only load S1 and S2.
# ===========================================================================

    set                 lr14 0;
    set                 lr2 0;;

g0_filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

g0_ch_loop:
    # Load S1 (group 0, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    # Load S2 (group 1, this channel) at cyclic index 256
    add                 lr1 lr10 lr11;
    ldr_cyclic_mult_reg lr1 cr0 lr6;;

    # --- kr=-1 (cyclic base 64): S0=zeros, normal masks ---
    set                 lr4 63;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 64;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 65;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # --- kr=0 (cyclic base 128): normal masks ---
    incr                lr3 1;
    set                 lr4 127;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr5 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 129;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # --- kr=+1 (cyclic base 192): normal masks ---
    incr                lr3 1;
    set                 lr4 191;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 192;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 193;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # Advance to next input channel
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr10 lr11 g0_ch_loop;;

    # Store output, advance filter
    str_acc_reg         lr2 cr2;
    incr                lr14 128;;

    incr                lr2 512;;

    blt                 lr14 lr15 g0_filter_loop;;

# ===========================================================================
# Main loop: groups 1-30
# ===========================================================================

    set                 lr7 1024;
    set                 lr13 31744;;

group_loop:
    set                 lr14 0;;

filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

ch_loop:
    # Compute S1 address = group_base + ch_offset
    add                 lr12 lr7 lr10;;

    # Load S0 (prev group, this channel) at cyclic index 0
    sub                 lr1 lr12 lr11;
    ldr_cyclic_mult_reg lr1 cr0 lr0;;

    # Load S1 (current group, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr12 cr0 lr5;;

    # Load S2 (next group, this channel) at cyclic index 256
    # Overlap with first mult.ve (S2 writes [256:384], mult reads [63])
    add                 lr1 lr12 lr11;
    set                 lr4 63;
    ldr_cyclic_mult_reg lr1 cr0 lr6;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    # --- kr=-1 remaining: kc=0, kc=+1 ---
    incr                lr3 1;
    set                 lr4 64;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 65;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # --- kr=0 (cyclic base 128): kc=-1, kc=0, kc=+1 ---
    incr                lr3 1;
    set                 lr4 127;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr5 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 129;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # --- kr=+1 (cyclic base 192): kc=-1, kc=0, kc=+1 ---
    incr                lr3 1;
    set                 lr4 191;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 192;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 193;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # Advance to next input channel
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr10 lr11 ch_loop;;

    # Store output, advance filter
    str_acc_reg         lr2 cr2;
    incr                lr14 128;;

    incr                lr2 512;;

    blt                 lr14 lr15 filter_loop;;

    # Advance to next group
    incr                lr7 1024;;

    blt                 lr7 lr13 group_loop;;

# ===========================================================================
# Group 31 (rows 62-63) -- bottom border
# Don't load S2.  Use mask slots 3/4/5 for kr=+1 to zero bottom row.
# ===========================================================================

    set                 lr14 0;;

g31_filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

g31_ch_loop:
    # Compute S1 address = group_base + ch_offset
    add                 lr12 lr7 lr10;;

    # Load S0 (group 30, this channel) at cyclic index 0
    sub                 lr1 lr12 lr11;
    ldr_cyclic_mult_reg lr1 cr0 lr0;;

    # Load S1 (group 31, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr12 cr0 lr5;;

    # (skip S2 -- stale data will be masked)

    # --- kr=-1 (cyclic base 64): normal masks ---
    set                 lr4 63;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 64;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 65;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # --- kr=0 (cyclic base 128): normal masks ---
    incr                lr3 1;
    set                 lr4 127;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr5 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 129;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # --- kr=+1 (cyclic base 192): special masks (4/3/5) to zero bottom row ---
    incr                lr3 1;
    set                 lr4 191;;

    set                 lr13 4;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 192;;

    set                 lr13 3;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 193;;

    set                 lr13 5;
    mult.ve             r0 lr4 lr13 lr0 lr3;
    acc;;

    # Advance to next input channel
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr10 lr11 g31_ch_loop;;

    # Store output, advance filter
    str_acc_reg         lr2 cr2;
    incr                lr14 128;;

    incr                lr2 512;;

    blt                 lr14 lr15 g31_filter_loop;;

end:
    bkpt;;
