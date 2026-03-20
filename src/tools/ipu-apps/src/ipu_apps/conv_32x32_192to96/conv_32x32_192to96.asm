# Standard 3x3 convolution: 192 input -> 96 output channels, 32x32.
#
# Since spatial width (32) < SIMD width (128), 4 spatial rows are packed
# per 128-byte chunk.  8 row-groups total.  The cyclic register holds 3
# neighboring chunks (S0=group_{g-1}, S1=group_g, S2=group_{g+1}) so
# that vertical neighbor access is a simple offset into the cyclic register.
#
# Each output channel has 192 input-channel 3x3 kernels (full cross-channel
# mixing).  Kernel per filter: 192 * 9 = 1728 bytes, which exceeds r0's
# 128 bytes.  Solution: split into 24 input-channel groups of 8, each
# fitting in one r0 load (72 bytes + 56 padding = 128 bytes per block).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (8 groups x 192 ch x 128 bytes = 196608 bytes)
#   cr1 = kernel base  (96 filters x 24 blocks x 128 bytes = 294912 bytes)
#   cr2 = output base  (8 groups x 96 out_ch x 512 bytes = 393216 bytes)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Input layout (interleaved by group, 192 channels):
#   group g, channel ch, local_row lr, col c:
#     offset = (g * 192 + ch) * 128 + lr * 32 + c
#   Group stride = 192 * 128 = 24576 bytes
#
# Output layout (interleaved by group, 32-bit accumulators):
#   group g, filter f: offset = (g * 96 + f) * 512
#
# Kernel layout (96 filters x 24 blocks x 128 bytes = 294912 bytes):
#   Filter f, block b (0..23):
#     At XMEM offset (f * 24 + b) * 128 from kernel base.
#     Block b: input channels b*8..(b+1)*8-1 (72 bytes data + 56 padding)
#     Within a block, for input channel ch (0-7):
#       byte[ch*9 + dr*3 + dc]
#
# Cyclic register usage (3 chunks loaded per input channel):
#   S0 at index 0:    group_{g-1}, same channel
#   S1 at index 128:  group_g,     same channel
#   S2 at index 256:  group_{g+1}, same channel
#
#   kr=-1: cyclic offset 95/96/97    (kc=-1/0/+1)   [128 - 32 - 1/0/+1]
#   kr= 0: cyclic offset 127/128/129 (kc=-1/0/+1)
#   kr=+1: cyclic offset 159/160/161 (kc=-1/0/+1)   [128 + 32 - 1/0/+1]
#
# Mask slots in r_mask (4 rows x 32 cols packed):
#   slot 0: all zeros            -> no masking                   (kc = 0)
#   slot 1: bits {0,32,64,96}    -> zero col 0 of each row       (kc = -1)
#   slot 2: bits {31,63,95,127}  -> zero col 31 of each row      (kc = +1)
#   slot 3: bits {96..127}       -> zero bottom row of chunk      (last group kr=+1 kc=0)
#   slot 4: bits {0,32,64,96..127} -> left + bottom              (last group kr=+1 kc=-1)
#   slot 5: bits {31,63,95..127}   -> right + bottom             (last group kr=+1 kc=+1)
#
# Register allocation:
#   lr0  = 0     (mask_shift, mask_slot_0, cyclic index for S0)
#   lr1  = temp  (S0/S2 address, mask slot for bottom border)
#   lr2  = output write offset
#   lr3  = kernel byte index within r0 (0-71 per input-channel group)
#   lr4  = temp  (cyclic offset for mult.ve, S2 cyclic index computation)
#   lr5  = 128   (cyclic index for S1, chunk stride)
#   lr6  = 294912 (filter loop limit = 96 filters x 3072 bytes/filter)
#   lr7  = row-group base offset (g * 24576)
#   lr8  = 1     (mask_slot_1 = kc=-1)
#   lr9  = 2     (mask_slot_2 = kc=+1)
#   lr10 = channel offset (0, 128, ..., 24448, spans all 24 channel groups)
#   lr11 = 1024  (input-channel group size = 8 ch x 128 bytes)
#   lr12 = temp  (S1 address)
#   lr13 = channel loop end (lr10 + 1024) / temp for group loop limit
#   lr14 = kernel memory offset (0, 128, ..., 294784)
#   lr15 = 24576 (row-group stride = 192 x 128, also total channels x 128)
#
# Structure:
#   Each filter needs 24 r0 loads (8 input channels each).
#   After the first 8 channels, reload r0 and continue accumulating.
#   For each row-group section (top, middle, bottom):
#     Filter loop (96 filters): reset_acc, process 24 channel groups, store.

# ===========================================================================
# Initialization
# ===========================================================================

    ldr_mult_mask_reg   lr0 cr3;
    set                 lr5 128;;

    set                 lr8 1;
    set                 lr9 2;;

    set                 lr11 1024;
    set                 lr15 24576;;

    # lr6 = 96 filters x 3072 bytes/filter = 294912 = 12 x lr15
    add                 lr6 lr15 lr15;;

    add                 lr13 lr6 lr15;;

    add                 lr6 lr13 lr13;;

    add                 lr6 lr6 lr6;;

# ===========================================================================
# Group 0 (rows 0-3) -- top border
# S0 = zeros (cyclic reg initialised to 0), only load S1 and S2.
# ===========================================================================

    set                 lr14 0;
    set                 lr2 0;;

g0_filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

    add                 lr13 lr10 lr11;;

g0_ch_loop:
    # Load S1 (group 0, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    # Load S2 (group 1, this channel) at cyclic index 256
    add                 lr1 lr10 lr15;
    add                 lr4 lr5 lr5;;

    ldr_cyclic_mult_reg lr1 cr0 lr4;;

    # --- kr=-1 (cyclic base 96): S0=zeros, normal masks ---
    set                 lr4 95;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 96;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 97;
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

    # --- kr=+1 (cyclic base 160): normal masks ---
    incr                lr3 1;
    set                 lr4 159;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 160;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 161;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # Advance to next input channel
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr10 lr13 g0_ch_loop;;

    # Check if next channel group needed
    incr                lr14 128;;

    blt                 lr10 lr15 g0_reload;;

    # All groups done -- store output and advance filter
    str_acc_reg         lr2 cr2;;

    incr                lr2 512;;

    blt                 lr14 lr6 g0_filter_loop;;

    b                   main_setup;;

g0_reload:
    ldr_mult_reg        r0 lr14 cr1;
    set                 lr3 0;
    add                 lr13 lr10 lr11;;

    b                   g0_ch_loop;;

# ===========================================================================
# Main loop: groups 1 .. 6
# Load all 3 chunks (S0, S1, S2).  All 9 kernel taps active.
# ===========================================================================

main_setup:
    set                 lr7 24576;;

group_loop:
    set                 lr14 0;;

filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

    add                 lr13 lr10 lr11;;

ch_loop:
    # Compute S1 address = group_base + channel_offset
    add                 lr12 lr7 lr10;;

    # Load S0 (prev group, this channel) at cyclic index 0
    sub                 lr1 lr12 lr15;
    ldr_cyclic_mult_reg lr1 cr0 lr0;;

    # Load S1 (current group, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr12 cr0 lr5;;

    # Load S2 (next group, this channel) at cyclic index 256
    add                 lr1 lr12 lr15;
    add                 lr4 lr5 lr5;;

    ldr_cyclic_mult_reg lr1 cr0 lr4;;

    # --- kr=-1 (cyclic base 96): normal masks ---
    set                 lr4 95;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 96;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 97;
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

    # --- kr=+1 (cyclic base 160): normal masks ---
    incr                lr3 1;
    set                 lr4 159;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 160;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 161;
    mult.ve             r0 lr4 lr9 lr0 lr3;
    acc;;

    # Advance to next input channel
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr10 lr13 ch_loop;;

    # Check if next channel group needed
    incr                lr14 128;;

    blt                 lr10 lr15 reload;;

    # All groups done -- store output and advance filter
    str_acc_reg         lr2 cr2;;

    incr                lr2 512;;

    blt                 lr14 lr6 filter_loop;;

    # Advance to next row-group
    add                 lr7 lr7 lr15;;

    # Middle loop limit: 7 * 24576 = 172032 = 21504 * 8
    set                 lr13 21504;;

    add                 lr13 lr13 lr13;;

    add                 lr13 lr13 lr13;;

    add                 lr13 lr13 lr13;;

    blt                 lr7 lr13 group_loop;;

    b                   g7_section;;

reload:
    ldr_mult_reg        r0 lr14 cr1;
    set                 lr3 0;
    add                 lr13 lr10 lr11;;

    b                   ch_loop;;

# ===========================================================================
# Group 7 (rows 28-31) -- bottom border
# Don't load S2.  Use mask slots 3/4/5 for kr=+1 to zero bottom row
# (positions 96-127).
# ===========================================================================

g7_section:
    set                 lr14 0;;

g7_filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

    add                 lr13 lr10 lr11;;

g7_ch_loop:
    # Compute S1 address = group_base + channel_offset
    add                 lr12 lr7 lr10;;

    # Load S0 (group 6, this channel) at cyclic index 0
    sub                 lr1 lr12 lr15;
    ldr_cyclic_mult_reg lr1 cr0 lr0;;

    # Load S1 (group 7, this channel) at cyclic index 128
    ldr_cyclic_mult_reg lr12 cr0 lr5;;

    # (skip S2 -- stale data will be masked by slots 3/4/5)

    # --- kr=-1 (cyclic base 96): normal masks ---
    set                 lr4 95;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 96;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 97;
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

    # --- kr=+1 (cyclic base 160): special masks (4/3/5) to zero bottom row ---
    incr                lr3 1;
    set                 lr4 159;;

    set                 lr1 4;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 160;;

    set                 lr1 3;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 161;;

    set                 lr1 5;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    # Advance to next input channel
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr10 lr13 g7_ch_loop;;

    # Check if next channel group needed
    incr                lr14 128;;

    blt                 lr10 lr15 g7_reload;;

    # All groups done -- store output and advance filter
    str_acc_reg         lr2 cr2;;

    incr                lr2 512;;

    blt                 lr14 lr6 g7_filter_loop;;

end:
    bkpt;;

g7_reload:
    ldr_mult_reg        r0 lr14 cr1;
    set                 lr3 0;
    add                 lr13 lr10 lr11;;

    b                   g7_ch_loop;;
