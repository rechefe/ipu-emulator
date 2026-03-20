# Pointwise (1x1) convolution: 160 input -> 160 output channels, 8x8 spatial.
#
# Paired-output processing: two output channels share one accumulator.
# OC f0 (even) accumulates in lanes 0-63, OC f1 (odd) in lanes 64-127.
# One str_acc_reg per pair stores all 512 bytes (both valid).
#
# 8x8 spatial fits in 64 bytes (half a 128-byte chunk).
# Input: same paired layout as standard conv -- 2 ICs per 128-byte chunk.
#   IC 2k in bytes 0-63, IC 2k+1 in bytes 64-127.
# Loaded at cyclic index 128: IC_even at cyclic[128..191], IC_odd at [192..255].
#
# Since pointwise has no spatial shifts, masks need only 2 slots:
#   Slot 0: bits {64-127} set -> zero lanes 64-127 (for f0, active lanes 0-63)
#   Slot 1: bits {0-63} set  -> zero lanes 0-63   (for f1, active lanes 64-127)
# The mask is loaded once and persists for the entire computation.
#
# For each input chunk (IC pair), 4 mult.ve operations:
#   f0 x IC_even: cyclic_offset=128, mask slot 0  (IC_even in lanes 0-63)
#   f0 x IC_odd:  cyclic_offset=192, mask slot 0  (IC_odd shifted to lanes 0-63)
#   f1 x IC_even: cyclic_offset=64,  mask slot 1  (IC_even shifted to lanes 64-127)
#   f1 x IC_odd:  cyclic_offset=128, mask slot 1  (IC_odd in lanes 64-127)
#
# Kernel layout: interleaved per IC pair, 3 blocks of 128 bytes per OC pair.
#   Per IC pair j (ICs 2j, 2j+1): 4 bytes = [f0[2j], f0[2j+1], f1[2j], f1[2j+1]]
#   Block 0: IC pairs 0-31   (128 bytes)
#   Block 1: IC pairs 32-63  (128 bytes)
#   Block 2: IC pairs 64-79  (64 bytes + 64 padding)
#   Total kernel: 80 OC pairs x 3 blocks x 128 bytes = 30720 bytes.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (80 chunks x 128 bytes = 10240 bytes)
#   cr1 = kernel base  (30720 bytes)
#   cr2 = output base  (80 pairs x 512 bytes = 40960 bytes)
#   cr3 = mask   base  (128 bytes)
#
# Register allocation:
#   lr0  = 0     (mask_shift=0, mask_slot_0 for f0)
#   lr2  = output write offset (0, 512, ..., 40448)
#   lr3  = r0 index (0..127 or 0..63, advances by 4 per chunk)
#   lr5  = 128   (cyclic load index; cyclic_offset for f0xeven & f1xodd)
#   lr6  = 64    (cyclic_offset for f1 x IC_even)
#   lr7  = 192   (cyclic_offset for f0 x IC_odd)
#   lr8  = 1     (mask_slot_1 for f1)
#   lr9  = kernel offset (0..30592, advances 128 per block)
#   lr10 = input chunk offset (0..10112, resets per OC pair)
#   lr11 = 128   (block loop end for blocks 0 & 1)
#   lr12 = 64    (block loop end for block 2)
#   lr15 = 40960 (output end condition)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr5 128;
    set                 lr6 64;;

    set                 lr7 192;
    set                 lr8 1;;

    set                 lr11 128;
    set                 lr12 64;;

    set                 lr15 20480;
    set                 lr9 0;;

    add                 lr15 lr15 lr15;
    set                 lr2 0;;

# Load mask (stays loaded for entire computation -- no spatial shifts)
    ldr_mult_mask_reg   lr0 cr3;;

# ===========================================================================
# OC pair loop (80 pairs of 2 output channels)
# ===========================================================================

oc_pair_loop:
    reset_acc;
    set                 lr10 0;;

# ---- Block 0 (ICs 0-63: 32 input chunks) ----

    ldr_mult_reg        r0 lr9 cr1;
    set                 lr3 0;;

block0_loop:
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    mult.ve             r0 lr5 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr7 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr6 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr5 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr3 lr11 block0_loop;;

# ---- Block 1 (ICs 64-127: 32 input chunks) ----

    incr                lr9 128;;

    ldr_mult_reg        r0 lr9 cr1;
    set                 lr3 0;;

block1_loop:
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    mult.ve             r0 lr5 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr7 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr6 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr5 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr3 lr11 block1_loop;;

# ---- Block 2 (ICs 128-159: 16 input chunks) ----

    incr                lr9 128;;

    ldr_mult_reg        r0 lr9 cr1;
    set                 lr3 0;;

block2_loop:
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    mult.ve             r0 lr5 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr7 lr0 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr6 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    mult.ve             r0 lr5 lr8 lr0 lr3;
    acc;;

    incr                lr3 1;
    incr                lr10 128;;

    blt                 lr3 lr12 block2_loop;;

# ---- Store result, advance to next OC pair ----

    str_acc_reg         lr2 cr2;
    incr                lr9 128;;

    incr                lr2 512;;

    blt                 lr2 lr15 oc_pair_loop;;

end:
    bkpt;;
