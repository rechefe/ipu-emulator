# Standard 3x3 convolution: 64 input -> 64 output channels, 16x16.
#
# Since spatial width (16) < SIMD width (128), 8 spatial rows are packed
# per 128-byte chunk.  2 row-groups total.  The cyclic register holds 3
# neighboring chunks (S0=group_{g-1}, S1=group_g, S2=group_{g+1}) so
# that vertical neighbor access is a simple offset into the cyclic register.
#
# Each output channel has 64 input-channel 3x3 kernels (full cross-channel
# mixing).  Kernel per filter: 64 * 9 = 576 bytes, which exceeds r0's
# 128 bytes.  Solution: split into 8 input-channel groups of 8, each
# fitting in one r0 load (72 bytes + 56 padding = 128 bytes per block).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (2 groups x 64 ch x 128 bytes = 16384 bytes)
#   cr1 = kernel base  (64 filters x 8 blocks x 128 bytes = 65536 bytes)
#   cr2 = output base  (2 groups x 64 out_ch x 512 bytes = 65536 bytes)
#   cr3 = mask   base  (128 bytes: 8 mask slots x 16 bytes each)
#
# Input layout (interleaved by group, 64 channels):
#   group g, channel ch, local_row lr, col c:
#     offset = (g * 64 + ch) * 128 + lr * 16 + c
#   Group stride = 64 * 128 = 8192 bytes
#
# Output layout (interleaved by group, 32-bit accumulators):
#   group g, filter f: offset = (g * 64 + f) * 512
#
# Kernel layout (64 filters x 8 blocks x 128 bytes = 65536 bytes):
#   Filter f, block b (0..7):
#     At XMEM offset (f * 8 + b) * 128 from kernel base.
#     Block b: input channels b*8..(b+1)*8-1 (72 bytes data + 56 padding)
#     Within a block, for input channel ch (0-7):
#       byte[ch*9 + dr*3 + dc]
#
# Cyclic register usage (3 chunks loaded per input channel):
#   S0 at index 0:    group_{g-1}, same channel
#   S1 at index 128:  group_g,     same channel
#   S2 at index 256:  group_{g+1}, same channel
#
#   kr=-1: cyclic offset 111/112/113  (kc=-1/0/+1)   [128 - 16 - 1/0/+1]
#   kr= 0: cyclic offset 127/128/129  (kc=-1/0/+1)
#   kr=+1: cyclic offset 143/144/145  (kc=-1/0/+1)   [128 + 16 - 1/0/+1]
#
# Mask slots in r_mask (8 rows x 16 cols packed):
#   slot 0: all zeros             -> no masking                    (kc = 0)
#   slot 1: bits {0,16,32,...,112} -> zero col 0 of each row       (kc = -1)
#   slot 2: bits {15,31,...,127}   -> zero col 15 of each row      (kc = +1)
#   slot 3: bits {112..127}       -> zero bottom row of chunk      (last group kr=+1 kc=0)
#   slot 4: bits {0,16,...,96,112..127} -> left + bottom           (last group kr=+1 kc=-1)
#   slot 5: bits {15,31,...,111,112..127} -> right + bottom        (last group kr=+1 kc=+1)
#
# Register allocation:
#   lr0  = 0     (mask_shift, mask_slot_0, cyclic index for S0)
#   lr1  = temp  (S0/S2 address, mask slot for bottom border)
#   lr2  = output write offset
#   lr3  = kernel byte index within r0 (0-71 per input-channel group)
#   lr4  = temp  (cyclic offset for mult.ve, S2 cyclic index computation)
#   lr5  = 128   (cyclic index for S1, chunk stride)
#   lr6  = 65536 (filter loop limit = 64 filters x 1024 bytes/filter)
#   lr7  = (unused permanent)
#   lr8  = 1     (mask_slot_1 = kc=-1)
#   lr9  = 2     (mask_slot_2 = kc=+1)
#   lr10 = channel offset (0, 128, ..., 8064, spans all 8 channel groups)
#   lr11 = 1024  (input-channel group size = 8 ch x 128 bytes)
#   lr12 = temp  (S1 address)
#   lr13 = channel loop end (lr10 + 1024)
#   lr14 = kernel memory offset (0, 128, ..., 65408)
#   lr15 = 8192  (row-group stride = 64 x 128, also total channels x 128)
#
# Structure:
#   Each filter needs 8 r0 loads (8 input channels each).
#   After the first 8 channels, reload r0 and continue accumulating.
#   Group 0 (top border): skip S0, load S1+S2, normal masks.
#   Group 1 (bottom border): load S0+S1, skip S2, special masks for kr=+1.

# ===========================================================================
# Initialization
# ===========================================================================

    ldr_mult_mask_reg   lr0 cr3;
    set                 lr5 128;;

    set                 lr8 1;
    set                 lr9 2;;

    set                 lr11 1024;
    set                 lr15 8192;;

    add                 lr6 lr15 lr15;;

    add                 lr6 lr6 lr6;;

    add                 lr6 lr6 lr6;;

# ===========================================================================
# Group 0 (rows 0-7) -- top border
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

    # --- kr=-1 (cyclic base 112): S0=zeros, normal masks ---
    set                 lr4 111;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 112;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 113;
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

    # --- kr=+1 (cyclic base 144): normal masks ---
    incr                lr3 1;
    set                 lr4 143;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 144;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 145;
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

    b                   g1_section;;

g0_reload:
    ldr_mult_reg        r0 lr14 cr1;
    set                 lr3 0;
    add                 lr13 lr10 lr11;;

    b                   g0_ch_loop;;

# ===========================================================================
# Group 1 (rows 8-15) -- bottom border
# Load S0 and S1.  Don't load S2.  Use mask slots 3/4/5 for kr=+1
# to zero bottom row (positions 112-127).
# ===========================================================================

g1_section:
    set                 lr14 0;;

g1_filter_loop:
    ldr_mult_reg        r0 lr14 cr1;
    reset_acc;
    set                 lr3 0;
    set                 lr10 0;;

    add                 lr13 lr10 lr11;;

g1_ch_loop:
    # Load S0 (group 0, this channel) at cyclic index 0
    ldr_cyclic_mult_reg lr10 cr0 lr0;;

    # Load S1 (group 1, this channel) at cyclic index 128
    add                 lr12 lr10 lr15;;

    ldr_cyclic_mult_reg lr12 cr0 lr5;;

    # (skip S2 -- stale data will be masked by slots 3/4/5)

    # --- kr=-1 (cyclic base 112): normal masks ---
    set                 lr4 111;
    mult.ve             r0 lr4 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 112;
    mult.ve             r0 lr4 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 113;
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

    # --- kr=+1 (cyclic base 144): special masks (4/3/5) to zero bottom row ---
    incr                lr3 1;
    set                 lr4 143;;

    set                 lr1 4;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 144;;

    set                 lr1 3;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    incr                lr3 1;
    set                 lr4 145;;

    set                 lr1 5;
    mult.ve             r0 lr4 lr1 lr0 lr3;
    acc;;

    # Advance to next input channel
    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr10 lr13 g1_ch_loop;;

    # Check if next channel group needed
    incr                lr14 128;;

    blt                 lr10 lr15 g1_reload;;

    # All groups done -- store output and advance filter
    str_acc_reg         lr2 cr2;;

    incr                lr2 512;;

    blt                 lr14 lr6 g1_filter_loop;;

end:
    bkpt;;

g1_reload:
    ldr_mult_reg        r0 lr14 cr1;
    set                 lr3 0;
    add                 lr13 lr10 lr11;;

    b                   g1_ch_loop;;
