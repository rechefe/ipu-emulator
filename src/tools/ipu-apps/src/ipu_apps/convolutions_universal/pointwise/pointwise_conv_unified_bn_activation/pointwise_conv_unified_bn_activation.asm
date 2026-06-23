# Unified Pointwise (1x1) Convolution + folded BN bias + ReLU
#
# A single code path handling any valid pointwise convolution via a multi-pass
# inner loop. Kernel layout: one OC per 128-byte register (zero-padded).
#
# BN/activation twin of pointwise_conv_unified:
#   * Folded bias — one INT8 bias per output channel, seeded into the
#     accumulator (r_acc = bias) once per OC before the conv taps, via a
#     MULT.EE broadcast of the bias byte (× CR1=1).  Batch-norm is assumed
#     already folded into the conv weights + this bias.
#   * ReLU — applied via ACTIVATE relu (instead of identity) before quantize.
#
# The bias region (cr10) mirrors the kernel layout: one 128-byte block per
# OC-pass, the OC's INT8 bias in byte 0 of its pass-0 block (rest zero).  So
# lr12 (the kernel byte offset) indexes the bias block identically to the
# kernel — no extra pointer/arithmetic needed.  Only pass-0 blocks are read.
# (cr10 was the base app's vestigial "tail_size" param — never read as an
# operand — so it is reused here; CR15 is reserved and not a valid operand.)
#
# See DESIGN.md (sibling file) for the structural overview.
#
# CR register parameters (master ISA: CR0 = read-only 0, CR1 = read-only 1):
#   cr0  = 0 constant AND input/cyclic-load base (INPUT_BASE_ADDR == 0)
#   cr1  = 1 constant (used as the pass-counter decrement)
#   cr2  = mask base address       (128 bytes of 0xFF — keep-all, new polarity)
#   cr3  = output base address
#   cr4  = num_passes              (= ceil(in_ch / 128))
#   cr5  = row_groups              (= rows * cols / 128)
#   cr6  = pipeline_limit_full     (= 128 - 5 = 123)
#   cr7  = out_channels
#   cr8  = row_group_stride        (= in_ch * 128)
#   cr9  = pipeline_limit_tail     (= tail_size - 5; may be neg as 2's comp)
#   cr10 = bias base address       (reused; was the vestigial tail_size param)
#   cr11 = num_passes - 1
#   cr12 = 128
#   cr13 = 16384   (input pass stride = 128 ICs * 128B)
#   cr14 = kernel base address     (relocated off CR1, now reserved)
#
# LR register allocation:
#   lr0  = input row-group base address (constant per row-group)
#   lr1  = output pointer (pre-offset by -128)
#   lr2  = row-group counter
#   lr3  = output channel counter
#   lr4  = fixed_idx within the current register (0..127 for A, 128..255 for B)
#   lr5  = 128 (output stride)
#   lr6  = 128
#   lr7  = 0   (cyclic S0 offset)
#   lr8  = 256 (cyclic S2 offset)
#   lr9  = 384 (cyclic S3 offset)
#   lr10 = pass_counter  (counts down; 0 means we are in the LAST pass)
#   lr11 = row_groups
#   lr12 = kernel byte offset (advances by 128 per pass within an OC)
#   lr13 = input base for the NEXT pass's preload (= lr0 for last-pass epilogue,
#          or lr0 + (p+1)*16384 for non-last-pass epilogue)
#   lr14 = temp (address computation; base for cyclic streaming loads)
#   lr15 = per-pass pipeline bound (lr4 + pipeline_limit)

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 lr7 cr0;
    ldr_mult_mask_reg   lr7 cr2;
    add                 lr5 lr7 cr12;;   # lr5 = 128

    add                 lr6 lr7 cr12;;   # lr6 = 128
    add                 lr8 lr6 lr6;;    # lr8 = 256

    add                 lr9 lr6 lr8;;    # lr9 = 384

    add                 lr11 lr7 cr5;;   # lr11 = row_groups

    SET                 lr1 cr0;;
    sub                 lr1 lr1 cr12;;   # lr1 = -128 (pre-offset)

    SET                 lr2 cr0;;        # lr2 = row-group counter

# ===========================================================================
# Row-group loop
# ===========================================================================

row_loop:

    # Pre-load first 4 input channels into r_cyclic S0..S3
    ldr_cyclic_mult_reg lr0 cr0 lr7;;

    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;;

    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;;

    add                 lr14 lr0 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    SET                 lr3 cr0;
    SET                 lr12 cr0;;       # lr3 = OC counter, lr12 = kernel offset

# ===========================================================================
# OC pair loop — process one OC per half (A then B)
# ===========================================================================

oc_pair_loop:

# ---------------------------------------------------------------------------
# Half A — process OC = lr3 using r0
# ---------------------------------------------------------------------------

    # Folded BN bias seed.  lr12 points at this OC's pass-0 block; cr10 (bias
    # base) indexes the parallel bias region, so the bias block lands in r0
    # with the OC's bias in byte 0.  Broadcast it (× CR1=1) and seed r_acc via
    # acc.first; THEN load the real weights over r0.
    ldr_mult_reg        r0 lr12 cr10;;     # bias block -> r0 (bias at byte 0)

    MULT.EE             lr7 cr1 0 lr7;     # bias = r0[0] * CR1(=1), broadcast
    acc.first;;                            # r_acc = bias (conv taps add on top)

    ldr_mult_reg        r0 lr12 cr14;;     # load OC's pass-0 weights into r0

    SET                 lr4 cr0;
    add                 lr10 lr7 cr11;
    add                 lr13 lr0 lr7;;    # lr13 = lr0 (pass 0 input base)

    # Choose first-pass W1: full vs tail (single-pass case)
    blt                 lr7 lr10 W1_FULL_A;;

    # num_passes == 1 (only the tail pass).  First conv tap is plain acc now —
    # bias was already seeded via acc.first above.
    add                 lr15 lr4 cr9;
    add                 lr14 lr0 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;;

    blt                 lr4 lr15 PIPELINE_BODY_A;;
    b                   POST_BODY_A;;

W1_FULL_A:
    add                 lr15 lr4 cr6;
    add                 lr14 lr0 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;;

# ---------------------------------------------------------------------------
# Pipeline body — 4 cycles per iter, 4 ICs per iter
# ---------------------------------------------------------------------------

PIPELINE_BODY_A:

    INC                 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    MULT.RC.VE          lr6 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    MULT.RC.VE          lr8 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    MULT.RC.VE          lr9 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;
    blt                 lr4 lr15 PIPELINE_BODY_A;;

# ---------------------------------------------------------------------------
# Post-body: decide which epilogue to enter (LAST = end-of-OC, MID = more passes)
# ---------------------------------------------------------------------------

POST_BODY_A:
    blt                 lr7 lr10 MID_EPILOGUE_A;;   # lr10 > 0 → more passes

# ---------------------------------------------------------------------------
# LAST_EPILOGUE_A: this pass is the OC's last. Reload cyclic from lr0
# (so next OC starts with ich0..3). aaq+store, then advance to next OC.
# ---------------------------------------------------------------------------

LAST_EPILOGUE_A:

    INC                 lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    MULT.RC.VE          lr6 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    MULT.RC.VE          lr8 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    MULT.RC.VE          lr9 lr4 0 lr7;
    acc;;

    # ACTIVATE (relu) applies ReLU to the just-finalized r_acc (conv + bias)
    # and copies it -> post_aaq_reg.  It reads the cycle-start snapshot of
    # r_acc, so it must run a cycle AFTER the final acc (above), not fused with
    # it.  Folded into the existing preload word (the aaq slot was free).
    INC                 lr4 1;
    add                 lr14 lr0 lr9;
    add                 lr1 lr1 lr5;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    ACTIVATE            relu 1;;

    # aaq clamps post_aaq_reg -> 128 INT8 bytes; the store (same word, store slot
    # runs after the aaq slot) drains them to XMEM.
    INC                 lr3 1;
    add                 lr12 lr12 cr12;
    aaq                 1;
    STR_POST_AAQ_REG    lr1 cr3;;

    # Fall through to Half B section

# ---------------------------------------------------------------------------
# Half B — process OC = lr3 using r1 (fixed_idx 128..255)
# ---------------------------------------------------------------------------

    # Bounds check: skip B half if no more OCs
    add                 lr14 lr7 cr7;;
    blt                 lr3 lr14 HALF_B_GO;;
    b                   END_ROW_GROUP_CHECK;;

HALF_B_GO:

    # Folded BN bias seed (half B).  Bias block -> r1; the bias byte sits at
    # r1's byte 0 = combined Ra index 128, so MULT.EE reads index lr5 (=128).
    ldr_mult_reg        r1 lr12 cr10;;     # bias block -> r1 (bias at byte 0)

    MULT.EE             lr5 cr1 0 lr7;     # bias = r1[0] * CR1(=1), broadcast
    acc.first;;                            # r_acc = bias

    ldr_mult_reg        r1 lr12 cr14;;

    add                 lr4 lr7 cr12;
    add                 lr10 lr7 cr11;
    add                 lr13 lr0 lr7;;    # lr13 = lr0 (pass 0 input base)

    blt                 lr7 lr10 W1_FULL_B;;

    add                 lr15 lr4 cr9;
    add                 lr14 lr0 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;;

    blt                 lr4 lr15 PIPELINE_BODY_B;;
    b                   POST_BODY_B;;

W1_FULL_B:
    add                 lr15 lr4 cr6;
    add                 lr14 lr0 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;;

PIPELINE_BODY_B:

    INC                 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr7;
    MULT.RC.VE          lr6 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    MULT.RC.VE          lr8 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    MULT.RC.VE          lr9 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr14 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;
    blt                 lr4 lr15 PIPELINE_BODY_B;;

POST_BODY_B:
    blt                 lr7 lr10 MID_EPILOGUE_B;;

LAST_EPILOGUE_B:

    INC                 lr4 1;
    ldr_cyclic_mult_reg lr0 cr0 lr7;
    MULT.RC.VE          lr6 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr0 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    MULT.RC.VE          lr8 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr0 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    MULT.RC.VE          lr9 lr4 0 lr7;
    acc;;

    # ACTIVATE (relu): ReLU the finalized r_acc (conv + bias) -> post_aaq_reg,
    # one cycle after the final acc (reads snapshot).  Folded into the preload.
    INC                 lr4 1;
    add                 lr14 lr0 lr9;
    add                 lr1 lr1 lr5;
    ldr_cyclic_mult_reg lr14 cr0 lr9;
    ACTIVATE            relu 1;;

    # aaq clamps post_aaq_reg; store (same word) drains to XMEM.
    INC                 lr3 1;
    add                 lr12 lr12 cr12;
    aaq                 1;
    STR_POST_AAQ_REG    lr1 cr3;;

# ---------------------------------------------------------------------------
# End of OC-pair: check if more OCs in this row-group
# ---------------------------------------------------------------------------

END_ROW_GROUP_CHECK:

    add                 lr14 lr7 cr7;;
    blt                 lr3 lr14 oc_pair_loop;;

# ---------------------------------------------------------------------------
# Advance to next row-group
# ---------------------------------------------------------------------------

    add                 lr0 lr0 cr8;
    INC                 lr2 1;;

    blt                 lr2 lr11 row_loop;;

end:
    bkpt;;

# ===========================================================================
# MID epilogues (more passes follow): reload cyclic from NEXT-pass base,
# then transition to next pass with kernel reload.
# ===========================================================================

MID_EPILOGUE_A:

    # Compute lr13 = lr0 + (current_pass_index + 1) * 16384
    # We don't track current_pass_index directly; we track lr10 = passes_remaining-1.
    # current_pass_index = num_passes - 1 - lr10.
    # next-pass base offset from lr0 = (current_pass_index + 1) * 16384
    #                                = (num_passes - lr10) * 16384.
    # Compute incrementally: maintain lr13 across passes.
    #
    # Simpler: derive lr13 from how many passes we have YET to do.
    # passes_done_after_this = num_passes - lr10. Next pass's base offset
    # from lr0 = (num_passes - lr10) * 16384.
    # We don't have num_passes*16384 in a register; compute via cr4 * cr13.
    # No mult instruction for LR/CR arithmetic. So:
    #
    # Track via increment: at start of OC, set lr13 = lr0. Each MID_EPILOGUE,
    # advance lr13 += cr13. This requires lr13 init somewhere — we do it
    # below in the kernel-load transition (set lr13 = lr0 + cr13 directly
    # since this is the FIRST mid-transition, where current_pass_index=0).
    #
    # Actually we need lr13 to track CURRENT pass's base so we can advance
    # by cr13 to get next pass's base. The "current pass base" for pass 0
    # is lr0. For pass 1 it's lr0+16384. Etc.
    #
    # Since this is hard to track without a dedicated init, use a separate
    # approach: maintain lr13 = current pass's base, initialized to lr0 at
    # OC start, advanced by cr13 each MID transition AFTER the epilogue uses
    # the previous value... no wait, we need NEXT-pass base for E1-E4.
    #
    # SIMPLEST: just compute lr13 = lr0 + cr13 * (num_passes - lr10) here.
    # But we have no multiply. So track incrementally — initialize at OC start
    # and advance per pass. (Adding init at OC start below in __init__-style.)

    # Advance lr13 to next pass's input base.
    # (lr13 was set to lr0 at OC entry; we add cr13 each mid-transition.)
    add                 lr13 lr13 cr13;;

    # E1-E4 with reloads from lr13 (next pass's ich0..ich3)
    INC                 lr4 1;
    ldr_cyclic_mult_reg lr13 cr0 lr7;
    MULT.RC.VE          lr6 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr13 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    MULT.RC.VE          lr8 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr13 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    MULT.RC.VE          lr9 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr13 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    # Transition: advance kernel pointer + decrement pass counter + load r0
    add                 lr12 lr12 cr12;
    sub                 lr10 lr10 cr1;;     # pass_counter -= 1

    ldr_mult_reg        r0 lr12 cr14;;

    # Reset lr4 = 0
    SET                 lr4 cr0;;

    # Select pipeline_limit: full or tail
    blt                 lr7 lr10 MP_A_FULL;;

    # Last pass for this OC: use tail limit
    add                 lr15 lr4 cr9;
    add                 lr14 lr13 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;;

    blt                 lr4 lr15 PIPELINE_BODY_A;;
    b                   POST_BODY_A;;

MP_A_FULL:
    add                 lr15 lr4 cr6;
    add                 lr14 lr13 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;;

    b                   PIPELINE_BODY_A;;

MID_EPILOGUE_B:

    add                 lr13 lr13 cr13;;

    INC                 lr4 1;
    ldr_cyclic_mult_reg lr13 cr0 lr7;
    MULT.RC.VE          lr6 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr13 lr6;
    ldr_cyclic_mult_reg lr14 cr0 lr6;
    MULT.RC.VE          lr8 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr13 lr8;
    ldr_cyclic_mult_reg lr14 cr0 lr8;
    MULT.RC.VE          lr9 lr4 0 lr7;
    acc;;

    INC                 lr4 1;
    add                 lr14 lr13 lr9;
    ldr_cyclic_mult_reg lr14 cr0 lr9;;

    add                 lr12 lr12 cr12;
    sub                 lr10 lr10 cr1;;

    ldr_mult_reg        r1 lr12 cr14;;

    add                 lr4 lr7 cr12;;       # lr4 = 128

    blt                 lr7 lr10 MP_B_FULL;;

    add                 lr15 lr4 cr9;
    add                 lr14 lr13 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;;

    blt                 lr4 lr15 PIPELINE_BODY_B;;
    b                   POST_BODY_B;;

MP_B_FULL:
    add                 lr15 lr4 cr6;
    add                 lr14 lr13 lr9;
    MULT.RC.VE          lr7 lr4 0 lr7;
    acc;;

    b                   PIPELINE_BODY_B;;
