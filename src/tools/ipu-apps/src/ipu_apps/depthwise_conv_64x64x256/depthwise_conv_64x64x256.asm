# Multi-channel depthwise convolution: 256 channels, 64x64, 3x3 kernel.
#
# Each channel has its own 3x3 kernel (depthwise = no cross-channel mixing).
# Since spatial width (64) < SIMD width (128), 2 spatial rows are packed
# per 128-byte chunk.  32 row-groups total.
#
# The cyclic register holds 3 neighboring chunks (S0, S1, S2) so that
# vertical neighbor access is a simple offset into the cyclic register:
#
#   kr=-1: cyclic offset 63/64/65     (kc=-1/0/+1)
#   kr= 0: cyclic offset 127/128/129  (kc=-1/0/+1)
#   kr=+1: cyclic offset 191/192/193  (kc=-1/0/+1)
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (32 groups x 256 ch x 128 bytes = 1048576 bytes)
#   cr1 = kernel base  (32 kernel groups x 128 bytes = 4096 bytes, padded)
#   cr2 = output base  (32 groups x 256 ch x 512 bytes = 4194304 bytes)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Input layout (interleaved by group, 256 channels):
#   group g, channel ch, local_row lr, col c:
#     offset = (g * 256 + ch) * 128 + lr * 64 + c
#   Group stride = 256 * 128 = 32768 bytes
#
# Output layout (interleaved by group, 32-bit accumulators):
#   group g, channel ch: offset = (g * 256 + ch) * 512
#
# Kernel layout (32 groups of 128 bytes, 8 channels per group):
#   Group kg, channel c (0-7) within group:
#     byte[c*9 + 0..8] = 9 taps for that channel
#   72 bytes of kernel data per group, padded to 128 bytes.
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
#   lr1  = temp  (S0/S2 address, mask slot for bottom border)
#   lr2  = output write offset
#   lr3  = kernel byte index within r0 (0-71 per kernel group)
#   lr4  = temp  (cyclic offset for mult.ve, also 256 for S2 cyclic index)
#   lr5  = 128   (cyclic index for S1, chunk stride)
#   lr6  = row-group counter (main loop)
#   lr7  = row-group base offset (g * 32768)
#   lr8  = 1     (mask_slot_1 = kc=-1)
#   lr9  = 2     (mask_slot_2 = kc=+1)
#   lr10 = 32768 (row-group stride = 256 * 128; also kg loop limit)
#   lr11 = channel offset within row-group (0, 128, ..., 32640)
#   lr12 = temp  (S1 address)
#   lr13 = channel loop end per kernel group (lr11 + 1024)
#   lr14 = kernel memory offset (0, 128, ..., 3968)
#   lr15 = 1024  (channel-group size = 8 ch * 128 bytes)
#
# Structure:
#   32 kernel groups x 8 channels each = 256 channels total.
#   For each row-group section (top border, main, bottom border):
#     Outer loop over kernel groups (load r0, iterate 8 channels)

# ===========================================================================
# Initialization
# ===========================================================================

    ldr_mult_mask_reg   lr0 cr3;
    set                 lr5 128;;

    set                 lr8 1;
    set                 lr9 2;;

    set                 lr10 16384;
    set                 lr15 1024;;

    add                 lr10 lr10 lr10;;

# ===========================================================================
# Row-group 0 (rows 0-1) -- top border
# S0 = zeros (cyclic reg initialised to 0), only load S1 and S2.
# Skip kr=-1 contributions (S0 is all zeros, result is zero anyway --
# but we still execute the taps since S0 *is* zero at init).
# ===========================================================================

    set                 lr11 0;
    set                 lr14 0;;

    set                 lr2 0;;

g0_kg_loop:
    ldr_mult_reg        r0 lr14 cr1;
    set                 lr3 0;
    add                 lr13 lr11 lr15;;

g0_ch_loop:
    # Load S1 (group 0, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr11 cr0 lr5;;

    # Load S2 (group 1, this channel) at cyclic index 256
    add                 lr1 lr11 lr10;
    add                 lr4 lr5 lr5;;

    ldr_cyclic_mult_reg lr1 cr0 lr4;
    reset_acc;;

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

    # Store output and advance
    str_acc_reg         lr2 cr2;
    incr                lr3 1;;

    incr                lr11 128;
    incr                lr2 512;;

    blt                 lr11 lr13 g0_ch_loop;;

    # Advance to next kernel group
    incr                lr14 128;;

    blt                 lr11 lr10 g0_kg_loop;;

# ===========================================================================
# Main loop: row-groups 1 .. 30
# Load all 3 chunks (S0, S1, S2).  All 9 kernel taps active.
# ===========================================================================

    add                 lr7 lr10 lr0;
    set                 lr6 1;;

row_loop:
    set                 lr11 0;
    set                 lr14 0;;

row_kg_loop:
    ldr_mult_reg        r0 lr14 cr1;
    set                 lr3 0;
    add                 lr13 lr11 lr15;;

ch_loop:
    # Compute S1 address = row-group base + channel offset
    add                 lr12 lr7 lr11;;

    # Load S0 (prev group, this channel) at cyclic index 0
    sub                 lr1 lr12 lr10;
    ldr_cyclic_mult_reg lr1 cr0 lr0;;

    # Load S1 (current group, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr12 cr0 lr5;;

    # Load S2 (next group, this channel) at cyclic index 256
    add                 lr1 lr12 lr10;
    add                 lr4 lr5 lr5;;

    ldr_cyclic_mult_reg lr1 cr0 lr4;
    reset_acc;;

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

    # Store and advance to next channel
    str_acc_reg         lr2 cr2;
    incr                lr3 1;;

    incr                lr11 128;
    incr                lr2 512;;

    blt                 lr11 lr13 ch_loop;;

    # Advance to next kernel group
    incr                lr14 128;;

    blt                 lr11 lr10 row_kg_loop;;

    # Advance to next row-group
    add                 lr7 lr7 lr10;
    incr                lr6 1;;

    set                 lr13 31;;

    blt                 lr6 lr13 row_loop;;

# ===========================================================================
# Row-group 31 (rows 62-63) -- bottom border
# Don't load S2.  Use mask slots 3/4/5 for kr=+1 to zero bottom row
# (positions 64-127).  Stale S2 data is harmless because of masking.
# ===========================================================================

    set                 lr11 0;
    set                 lr14 0;;

g31_kg_loop:
    ldr_mult_reg        r0 lr14 cr1;
    set                 lr3 0;
    add                 lr13 lr11 lr15;;

g31_ch_loop:
    # Compute S1 address = row-group base + channel offset
    add                 lr12 lr7 lr11;;

    # Load S0 (group 30, this channel) at cyclic index 0
    sub                 lr1 lr12 lr10;
    ldr_cyclic_mult_reg lr1 cr0 lr0;;

    # Load S1 (group 31, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr12 cr0 lr5;;

    # (skip S2 -- stale data will be masked by slots 3/4/5)
    reset_acc;;

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

    set                 lr1 4;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 192;;

    set                 lr1 3;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 193;;

    set                 lr1 5;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    # Store and advance to next channel
    str_acc_reg         lr2 cr2;
    incr                lr3 1;;

    incr                lr11 128;
    incr                lr2 512;;

    blt                 lr11 lr13 g31_ch_loop;;

    # Advance to next kernel group
    incr                lr14 128;;

    blt                 lr11 lr10 g31_kg_loop;;

end:
    bkpt;;
