# Universal Pointwise (1x1) Convolution: 8x8 spatial, flexible channels.
#
# Paired-output processing: two output channels share one accumulator.
# OC f0 (even) accumulates in lanes 0-63, OC f1 (odd) in lanes 64-127.
# Output: aaq quantizes r_acc to 128 bytes int8 per OC pair.
#
# 8x8 spatial fits in 64 bytes (half a 128-byte chunk).
# Input: 2 ICs per 128-byte chunk (IC_even in bytes 0-63, IC_odd in 64-127).
# Loaded at cyclic index 128: IC_even at cyclic[128..191], IC_odd at [192..255].
#
# For each input chunk (IC pair), 4 mult.ve operations:
#   f0 x IC_even: cyclic_offset=128, mask slot 0  (IC_even in lanes 0-63)
#   f0 x IC_odd:  cyclic_offset=192, mask slot 0  (IC_odd to lanes 0-63)
#   f1 x IC_even: cyclic_offset=64,  mask slot 1  (IC_even to lanes 64-127)
#   f1 x IC_odd:  cyclic_offset=128, mask slot 1  (IC_odd in lanes 64-127)
#
# Kernel layout: interleaved per IC pair, packed into 128-byte blocks.
#   Per IC pair j: 4 bytes = [f0[2j], f0[2j+1], f1[2j], f1[2j+1]]
#   32 IC pairs per block (128 bytes). Last block may be partial (zero-padded).
#
# Mask: 2 slots:
#   Slot 0: bits {64-127} set -> zero lanes 64-127 (for f0)
#   Slot 1: bits {0-63} set  -> zero lanes 0-63   (for f1)
#
# CR registers (set by harness):
#   cr0  = input base address
#   cr1  = kernel base address
#   cr2  = output base address
#   cr3  = mask base address
#   cr4  = kernel_bytes_per_oc_pair (= ceil(ic_pairs/32) * 128)
#   cr5  = total_input_bytes (= ic_pairs * 128)
#   cr6  = total_output_bytes (= oc_pairs * 128)
#   cr12 = 128  (step constant: chunk / kernel block / output advance)
#   cr13 = 64   (cyclic_offset for f1 x IC_even)
#   cr14 = 192  (cyclic_offset for f0 x IC_odd)
#
# LR registers:
#   lr0  = 0     (mask_shift=0; also zero constant for add src_a)
#   lr2  = output write offset (0, 128, ..., total_output_bytes-128)
#   lr3  = r0 byte index (0..127, resets per block)
#   lr5  = 128   (cyclic load index, cyclic_offset for f0xeven & f1xodd, block size)
#   lr6  = 64    (cyclic_offset for f1 x IC_even, from cr13)
#   lr7  = 192   (cyclic_offset for f0 x IC_odd, from cr14)
#   lr9  = kernel offset for start of current OC pair
#   lr10 = input chunk offset (0, 128, ..., total_input_bytes-128)
#   lr11 = total_input_bytes (copy of cr5, ic_loop limit)
#   lr12 = current kernel block read offset (advances during ic_loop)
#   lr15 = total_output_bytes (copy of cr6, oc_pair_loop limit)

# ===========================================================================
# Initialization
# ===========================================================================

    set                 lr0 0;;

    add                 lr5 lr0 cr12;
    add                 lr6 lr0 cr13;;

    add                 lr7 lr0 cr14;;

# Load mask (stays loaded for entire computation)
    ldr_mult_mask_reg   lr0 cr3;;

# Copy CR parameters to LR for use in blt
    add                 lr11 lr0 cr5;
    add                 lr15 lr0 cr6;;

    set                 lr9 0;
    set                 lr2 0;;

# ===========================================================================
# OC pair loop (processes 2 output channels per iteration)
# ===========================================================================

oc_pair_loop:
    reset_acc;
    set                 lr10 0;;

# Load first kernel block for this OC pair
    add                 lr12 lr9 lr0;;

    ldr_mult_reg        r0 lr12 cr1;
    set                 lr3 0;;

# ---------------------------------------------------------------------------
# IC loop: iterate over all IC pairs, reload r0 every 32 IC pairs
# ---------------------------------------------------------------------------

ic_loop:
    ldr_cyclic_mult_reg lr10 cr0 lr5;;

    # f0 x IC_even: cyclic_offset=128, mask=slot0
    mult.ve.cyclic      lr5 0 lr0 lr3;
    acc;;

    # f0 x IC_odd: cyclic_offset=192, mask=slot0
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr7 0 lr0 lr3;
    acc;;

    # f1 x IC_even: cyclic_offset=64, mask=slot1
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr6 1 lr0 lr3;
    acc;;

    # f1 x IC_odd: cyclic_offset=128, mask=slot1
    add                 lr3 lr3 1;
    mult.ve.cyclic      lr5 1 lr0 lr3;
    acc;;

    add                 lr3 lr3 1;
    add                 lr10 lr10 cr12;;

    # Check if we consumed an entire r0 block (lr3 == 128)
    blt                 lr3 lr5 skip_reload;;

    # Reload next kernel block
    add                 lr12 lr12 cr12;;

    ldr_mult_reg        r0 lr12 cr1;
    set                 lr3 0;;

skip_reload:
    blt                 lr10 lr11 ic_loop;;

# ---------------------------------------------------------------------------
# Store result, advance to next OC pair
# ---------------------------------------------------------------------------

    aaq;;

    xmem.store_aaq_result lr2 cr2;;

    add                 lr9 lr9 cr4;
    add                 lr2 lr2 cr12;;

    blt                 lr2 lr15 oc_pair_loop;;

end:
    bkpt;;
