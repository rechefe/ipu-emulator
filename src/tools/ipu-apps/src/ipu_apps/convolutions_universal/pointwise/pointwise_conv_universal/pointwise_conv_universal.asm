# Universal Pointwise (1x1) Convolution
#
# A single binary that handles ANY valid pointwise convolution configuration.
# Parameters are passed via CR registers at runtime — no recompilation needed.
#
# Supported configurations:
#   - Spatial: any power-of-2 rows/cols in [16..128]
#   - in_channels: any value where gcd_pow2(in_channels, 128) >= 4
#     (i.e., in_channels must be divisible by at least 4)
#   - out_channels: must be divisible by 2 * (128 / G),
#     where G = min(largest_pow2_dividing(in_channels), 128)
#
# Two code paths are selected at runtime:
#   - Original path: when num_groups == 1 (in_channels divides 128) — zero overhead
#   - Grouped path:  when num_groups > 1  — processes G channels per group,
#     accumulating across groups without resetting the accumulator
#
# ISA: mult.ve.cyclic (mask_offset immediate), acc.first, no reset_acc+mult+acc triplet.
#
# CR register parameters (set by harness):
#   cr0 = input base address
#   cr1 = kernel base address
#   cr2 = mask base address   (128 bytes of zeros)
#   cr3 = output base address
#   cr4 = oc_per_reg           (= 128 / G)
#   cr5 = row_groups           (= rows * cols / 128)
#   cr6 = pipeline_limit       (= G - 5; for inner loop branch)
#   cr7 = out_channels
#   cr8 = row_group_stride     (= in_channels * 128)
#   cr9 = num_groups * 128     (for group limit; 128 when num_groups == 1)
#   cr10 = G * 128             (input stride per channel group)
#   cr11 = G                   (channel group size)
#   cr12 = 128                 (constant; cyclic slot stride / output stride)
#   cr13 = 256                 (constant; cyclic S2 offset)
#   cr14 = 512                 (constant; cyclic S3 base / kernel group advance)
#
# LR register allocation (original path, num_groups == 1):
#   lr0  = input row-group base address
#   lr1  = output pointer (pre-offset by -128)
#   lr2  = row-group counter
#   lr3  = output channel counter
#   lr4  = kernel byte index (continuous across OCs within a kernel group)
#   lr5  = 128  (output stride constant; loaded from cr12)
#   lr6  = 128  (cyclic S1 offset / channel stride constant; loaded from cr12)
#   lr7  = 0    (cyclic S0 offset / zero constant; default value)
#   lr8  = 256  (cyclic S2 offset / kernel group advance constant; loaded from cr13)
#   lr9  = 384  (cyclic S3 offset; synthesized as cr12+cr13)
#   lr10 = oc_per_reg (from cr4)
#   lr11 = row_groups  (from cr5)
#   lr12 = kernel memory offset (0, 256, 512, ... per kernel group)
#   lr13 = OC limit for current register half (lr3 + lr10)
#   lr14 = temp (address computation)
#   lr15 = per-OC pipeline bound (lr4 + cr6, recomputed each OC)
#
# LR register allocation (grouped path, num_groups > 1):
#   lr0  = input row-group base address (constant per row-group)
#   lr1  = output pointer (pre-offset by -128)
#   lr2  = row-group counter
#   lr3  = output channel counter
#   lr4  = kernel byte index (within register, resets per group)
#   lr5  = 128  (output stride; loaded from cr12)
#   lr6  = 128  (cyclic stride; loaded from cr12)
#   lr7  = 0
#   lr8  = 256  (loaded from cr13)
#   lr9  = 384  (synthesized as cr12+cr13)
#   lr10 = kernel group offset (starts at lr12, advances +128 per group)
#   lr11 = input group base (starts at lr0, advances +G*128 per group)
#   lr12 = batch kernel base offset (advances per batch)
#   lr13 = OC limit for current register half
#   lr14 = temp (address computation)
#   lr15 = temp (group limit / pipeline bound)
#
# Pipeline strategy:
#   r_cyclic has 4 slots of 128 bytes each (512 total, wrapping).
#   First 4 input channels are pre-loaded into S0-S3.
#   A 4-word loop body processes remaining channels:
#     Word A: load->S0, mult from S1
#     Word B: load->S1, mult from S2
#     Word C: load->S2, mult from S3
#     Word D: load->S3, mult from S0 + branch
#   After the loop, a fixed 4-word epilogue multiplies the last 3
#   channels and reloads ich0-3 for the next output channel (or group).
#
# Grouped path addition:
#   When in_channels doesn't divide 128, channels are processed in groups
#   of G. Each group runs the full pipeline for G channels, then advances
#   to the next group's kernel weights and input channels. The accumulator
#   is only reset at the start of a new output channel, not between groups.
#   The epilogue overlaps the next group's (or next OC's) cyclic preload
#   with the current group's last 3 multiplies, maintaining efficiency.

# ===========================================================================
# Initialization
# ===========================================================================

    # Load mask data (all zeros — no masking for pointwise conv)
    SET                 lr7 cr0;
    ldr_mult_mask_reg   lr7 cr2;
    add                 lr5 lr7 cr12;;   # lr5 = 128 (output stride)

    add                 lr6 lr7 cr12;
    add                 lr8 lr7 cr13;;   # lr6=128, lr8=256

    add                 lr9 lr6 lr8;;    # lr9 = 384

    # Copy CR parameters into LR registers (used by original path)
    add                 lr10 lr7 cr4;
    add                 lr11 lr7 cr5;;

    SET                 lr1 cr0;;
    sub                 lr1 lr1 cr12;;   # lr1 = -128 (pre-offset)

# ===========================================================================
# Row-group loop
# ===========================================================================

row_loop:

    # Pre-load first 4 input channels into r_cyclic S0-S3
    ldr_cyclic_mult_reg lr0 cr0 lr7;;

    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;;

    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;;

    add                 lr14 lr0 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    SET                 lr3 cr0;
    SET                 lr12 cr0;;

    # Path selection: if num_groups > 1, use grouped path
    # cr9 = num_groups * 128; lr6 = 128
    # If 128 < cr9 → num_groups > 1 → grouped path
    add                 lr14 lr7 cr9;;

    blt                 lr6 lr14 kernel_group_loop_g;;

# ###########################################################################
# ORIGINAL PATH (num_groups == 1) — unchanged from previous version
# ###########################################################################

# ===========================================================================
# Kernel-group loop (each group loads r0+r1 = 2*oc_per_reg output channels)
# ===========================================================================

kernel_group_loop:

    # Load kernel pair: r0 and r1
    ldr_mult_reg        r0 lr12 cr1;;

    add                 lr14 lr12 lr6;
    ldr_mult_reg        r1 lr14 cr1;;

    SET                 lr4 cr0;
    add                 lr13 lr3 lr10;;

# ---------------------------------------------------------------------------
# Inner loop A: output channels from r0
# ---------------------------------------------------------------------------

LOOP_OCH_A:

    # W1: Reset acc + first mult (ich0 from S0) folded via acc.first
    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc.first;;

    # Guard: skip pipeline if in_channels <= 4 (pipeline_limit < 0)
    # lr15 = lr4 + (in_channels - 5) = per-OC pipeline bound
    # lr14 = lr0 + 384, so first iteration's lr14 += 128 → ich4 address
    add                 lr15 lr4 cr6;
    add                 lr14 lr0 lr9;;

    blt                 lr15 lr4 EPILOGUE_A;;

# Pipeline body: 4-word loop processing 4 input channels per iteration.
# Cyclic slot rotation is hardcoded in the 4 words; loop count is dynamic.

PIPELINE_LOOP_A:

    # load ich(k)   -> S0, mult ich(k-3) from S1
    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve.cyclic      lr6 0 lr7 lr4;
    acc;;

    # load ich(k+1) -> S1, mult ich(k-2) from S2
    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve.cyclic      lr8 0 lr7 lr4;
    acc;;

    # load ich(k+2) -> S2, mult ich(k-1) from S3
    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve.cyclic      lr9 0 lr7 lr4;
    acc;;

    # load ich(k+3) -> S3, mult ich(k) from S0; branch if more
    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc;
    blt                 lr4 lr15 PIPELINE_LOOP_A;;

EPILOGUE_A:

    # E1: reload ich0 -> S0, mult second-to-last-3 from S1
    add                 lr4 lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    mult.ve.cyclic      lr6 0 lr7 lr4;
    acc;;

    # E2: reload ich1 -> S1, mult second-to-last-2 from S2
    add                 lr4 lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve.cyclic      lr8 0 lr7 lr4;
    acc;;

    # E3: reload ich2 -> S2, mult last channel from S3; quantize r_acc -> aaq_result
    add                 lr4 lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve.cyclic      lr9 0 lr7 lr4;
    acc;
    aaq;;

    # E4: reload ich3 -> S3, advance output pointer
    add                 lr4 lr4 1;
    add                 lr14 lr0 lr9;
    add                 lr1 lr1 lr5;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # Store 128-byte quantized output, advance OC counter
    add                 lr3 lr3 1;
    xmem.store_aaq_result lr1 cr3;;

    # Branch back if more OCs in first half
    blt                 lr3 lr13 LOOP_OCH_A;;

# ---------------------------------------------------------------------------
# Transition: reset kernel index for second register half (r1)
# fixed_idx 128..255 selects r1[fixed_idx-128]; seed lr4=128 so loop B
# uses the same cyclic pipeline structure as A with no extra register.
# ---------------------------------------------------------------------------

    add                 lr4 lr7 cr12;   # lr4 = 128
    add                 lr13 lr3 lr10;;

# ---------------------------------------------------------------------------
# Inner loop B: output channels from r1 (identical structure to A)
# ---------------------------------------------------------------------------

LOOP_OCH_B:

    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc.first;;

    add                 lr15 lr4 cr6;
    add                 lr14 lr0 lr9;;

    blt                 lr15 lr4 EPILOGUE_B;;

PIPELINE_LOOP_B:

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve.cyclic      lr6 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve.cyclic      lr8 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve.cyclic      lr9 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc;
    blt                 lr4 lr15 PIPELINE_LOOP_B;;

EPILOGUE_B:

    add                 lr4 lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    mult.ve.cyclic      lr6 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve.cyclic      lr8 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve.cyclic      lr9 0 lr7 lr4;
    acc;
    aaq;;

    add                 lr4 lr4 1;
    add                 lr14 lr0 lr9;
    add                 lr1 lr1 lr5;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    add                 lr3 lr3 1;
    xmem.store_aaq_result lr1 cr3;;

    blt                 lr3 lr13 LOOP_OCH_B;;

# ---------------------------------------------------------------------------
# Advance to next kernel group
# ---------------------------------------------------------------------------

    add                 lr14 lr7 cr7;
    add                 lr12 lr12 lr8;;

    blt                 lr3 lr14 kernel_group_loop;;

# ---------------------------------------------------------------------------
# Advance to next row group
# ---------------------------------------------------------------------------

    add                 lr0 lr0 cr8;
    add                 lr2 lr2 1;;

    blt                 lr2 lr11 row_loop;;

end:
    bkpt;;

# ###########################################################################
# GROUPED PATH (num_groups > 1) — processes G channels per group
# ###########################################################################

# ===========================================================================
# Grouped kernel-group loop
# Each iteration processes 2*oc_per_reg output channels.
# For each OC, all groups are processed before storing.
# ===========================================================================

kernel_group_loop_g:

    SET                 lr4 cr0;
    add                 lr13 lr3 cr4;;

# ---------------------------------------------------------------------------
# Grouped inner loop A: output channels from r0
# ---------------------------------------------------------------------------

LOOP_OCH_A_G:

    # Initialize group loop: lr10 = kernel group offset, lr11 = input group base
    add                 lr10 lr12 lr7;
    add                 lr11 lr0 lr7;;

GROUP_LOOP_A_G:

    # Load r0 for this group
    ldr_mult_reg        r0 lr10 cr1;;

    # First group of a new OC: use acc.first to reset accumulator
    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc.first;;

    # Pipeline guard: lr15 = lr4 + (G - 5)
    add                 lr15 lr4 cr6;
    add                 lr14 lr11 lr9;;

    blt                 lr15 lr4 EPILOGUE_GROUP_A_G;;

PIPELINE_LOOP_GROUP_A_G:

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve.cyclic      lr6 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve.cyclic      lr8 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve.cyclic      lr9 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc;
    blt                 lr4 lr15 PIPELINE_LOOP_GROUP_A_G;;

EPILOGUE_GROUP_A_G:

    # Advance group pointers (before epilogue loads, so epilogue preloads next group)
    add                 lr15 lr12 cr9;
    add                 lr10 lr10 lr6;;

    add                 lr11 lr11 cr10;
    blt                 lr10 lr15 EPILOGUE_LOADS_A_G;;

    # Last group: reset lr11 to lr0 (preload group 0 for next OC)
    add                 lr11 lr0 lr7;;

EPILOGUE_LOADS_A_G:

    # E1: preload ich0 -> S0, mult from S1
    add                 lr4 lr4 1;
    ldr_cyclic_mult_reg lr11 cr0 lr7;
    mult.ve.cyclic      lr6 0 lr7 lr4;
    acc;;

    # E2: preload ich1 -> S1, mult from S2
    add                 lr4 lr4 1;
    add                 lr14 lr11 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve.cyclic      lr8 0 lr7 lr4;
    acc;;

    # E3: preload ich2 -> S2, mult from S3; quantize on last group
    add                 lr4 lr4 1;
    add                 lr14 lr11 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve.cyclic      lr9 0 lr7 lr4;
    acc;
    aaq;;

    # E4: preload ich3 -> S3
    add                 lr4 lr4 1;
    add                 lr14 lr11 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # Check if more groups (lr10, lr15 unchanged during epilogue)
    blt                 lr10 lr15 CONTINUE_GROUP_A_G;;

    # All groups done: advance output pointer and store 128-byte quantized output
    add                 lr1 lr1 lr5;
    add                 lr3 lr3 1;
    xmem.store_aaq_result lr1 cr3;;

    blt                 lr3 lr13 LOOP_OCH_A_G;;

    # Fall through to B half transition
    b                   TRANSITION_B_G;;

CONTINUE_GROUP_A_G:
    # Not last group: restore lr4 to OC's starting byte index, accumulate into existing r_acc
    sub                 lr4 lr4 cr11;;

    # Load r0 for this group
    ldr_mult_reg        r0 lr10 cr1;;

    # Subsequent group: plain acc (do NOT reset accumulator)
    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc;;

    # Pipeline guard
    add                 lr15 lr4 cr6;
    add                 lr14 lr11 lr9;;

    blt                 lr15 lr4 EPILOGUE_GROUP_A_G;;
    b                   PIPELINE_LOOP_GROUP_A_G;;

# ---------------------------------------------------------------------------
# Transition: reset for second register half (r1)
# fixed_idx 128..255 selects r1[fixed_idx-128]; seed lr4=128.
# ---------------------------------------------------------------------------

TRANSITION_B_G:

    add                 lr4 lr7 cr12;   # lr4 = 128
    add                 lr13 lr3 cr4;;

# ---------------------------------------------------------------------------
# Grouped inner loop B: output channels from r1
# ---------------------------------------------------------------------------

LOOP_OCH_B_G:

    # lr10 = kernel offset for r1 groups = lr12 + num_groups * 128
    add                 lr10 lr12 cr9;
    add                 lr11 lr0 lr7;;

GROUP_LOOP_B_G:

    ldr_mult_reg        r1 lr10 cr1;;

    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc.first;;

    add                 lr15 lr4 cr6;
    add                 lr14 lr11 lr9;;

    blt                 lr15 lr4 EPILOGUE_GROUP_B_G;;

PIPELINE_LOOP_GROUP_B_G:

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    mult.ve.cyclic      lr6 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve.cyclic      lr8 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve.cyclic      lr9 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc;
    blt                 lr4 lr15 PIPELINE_LOOP_GROUP_B_G;;

EPILOGUE_GROUP_B_G:

    # Advance group pointers
    add                 lr15 lr12 cr9;
    add                 lr10 lr10 lr6;;

    # Note: For B half, group limit is lr12 + 2*cr9 (r1 groups start at lr12+cr9)
    # But lr10 started at lr12+cr9, so limit is lr12+cr9 + num_groups*128 = lr12+2*cr9
    add                 lr15 lr15 cr9;
    add                 lr11 lr11 cr10;;

    blt                 lr10 lr15 EPILOGUE_LOADS_B_G;;

    # Last group: reset lr11 to lr0
    add                 lr11 lr0 lr7;;

EPILOGUE_LOADS_B_G:

    add                 lr4 lr4 1;
    ldr_cyclic_mult_reg lr11 cr0 lr7;
    mult.ve.cyclic      lr6 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr11 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    mult.ve.cyclic      lr8 0 lr7 lr4;
    acc;;

    add                 lr4 lr4 1;
    add                 lr14 lr11 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    mult.ve.cyclic      lr9 0 lr7 lr4;
    acc;
    aaq;;

    add                 lr4 lr4 1;
    add                 lr14 lr11 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # Check if more groups
    blt                 lr10 lr15 CONTINUE_GROUP_B_G;;

    # All groups done: advance output pointer and store 128-byte quantized output
    add                 lr1 lr1 lr5;
    add                 lr3 lr3 1;
    xmem.store_aaq_result lr1 cr3;;

    blt                 lr3 lr13 LOOP_OCH_B_G;;

    # Fall through to batch advance
    b                   ADVANCE_BATCH_G;;

CONTINUE_GROUP_B_G:
    # Not last group: restore lr4, accumulate into existing r_acc
    sub                 lr4 lr4 cr11;;

    # Load r1 for this group
    ldr_mult_reg        r1 lr10 cr1;;

    # Subsequent group: plain acc (do NOT reset accumulator)
    mult.ve.cyclic      lr7 0 lr7 lr4;
    acc;;

    # Pipeline guard
    add                 lr15 lr4 cr6;
    add                 lr14 lr11 lr9;;

    blt                 lr15 lr4 EPILOGUE_GROUP_B_G;;
    b                   PIPELINE_LOOP_GROUP_B_G;;

# ---------------------------------------------------------------------------
# Advance to next kernel batch (grouped path)
# ---------------------------------------------------------------------------

ADVANCE_BATCH_G:

    # Batch stride = 2 * num_groups * 128 = 2 * cr9.
    # add reads snapshot so split across two words to avoid RAW on lr14.
    add                 lr14 lr7 cr9;;
    add                 lr14 lr14 cr9;;
    add                 lr12 lr12 lr14;;

    add                 lr14 lr7 cr7;;

    blt                 lr3 lr14 kernel_group_loop_g;;

# ---------------------------------------------------------------------------
# Advance to next row group (grouped path)
# ---------------------------------------------------------------------------

    add                 lr0 lr0 cr8;
    add                 lr2 lr2 1;;

    add                 lr14 lr7 cr5;;

    blt                 lr2 lr14 row_loop;;

end_g:
    bkpt;;
