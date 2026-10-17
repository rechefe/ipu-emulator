{#- pointwise_conv_tpl.asm -- y = act(conv2d(x, w, 1x1) + b): a pointwise convolution in FP32 wide vectors, any in_channels through a multi-pass inner loop, with an optional folded bias and identity or ReLU.

    The template of kernels/convolutions/registry.toml's pointwise_conv_unified
    (has_bias = false, apply_relu = false) and
    pointwise_conv_unified_bn_activation (apply_relu = true; has_bias either: a
    query without one runs with a zero bias). Batch-norm is assumed folded
    into the weights and the bias. The program reads the query's own
    parameters: "seeded" (apply_relu) selects the bias seed, apply_relu the
    activation. The seed follows apply_relu, not has_bias or apply_relu as in
    conv_universal_tpl, because the bias region b and CR10 follow it too (b's
    when, and CR10's: b when seeded, the unread tail_size otherwise); has_bias
    without ReLU has no entry yet (E07-20).

    Pipeline (one output channel per register half)
    - Each row group (one 128-element chunk of every input channel) preloads
      input channels 0..3 into R_CYCLIC slots S0..S3 (element offsets 0, 128,
      256, 384: LR7, LR6, LR8, LR9), then walks the output channels two at a
      time: half A loads the channel's weights into R0 (MULT.RC.VE index
      0..127), half B the next channel's into R1 (128..255).
    - Each pass covers 128 input channels (the last pass the tail):
      MULT.RC.VE multiplies the cyclic slot of channel i by weight i and
      accumulates, while LD.MULT.CYCLIC streams channel i + 4 into the slot
      it frees. pipeline_limit_full (123) and pipeline_limit_tail
      (tail - 5) bound the loop; MID_EPILOGUE_* move to the next pass
      (pass_stride rows on, the next weight row), LAST_EPILOGUE_* activate
      and store the output chunk.
    - Seeded, R_ACC starts from the bias (LD.MULT of the channel's b row,
      MULT.EE of element 0 times CR1, ACC.ADD.FIRST) and the first tap
      accumulates on it (ACC.ADD); unseeded, the first tap starts R_ACC
      (ACC.ADD.FIRST).
    The variants differ in exactly four places, each in both halves: the
    bias seed, ACC.ADD vs ACC.ADD.FIRST on the first tap (both W1 paths),
    the LR5 constant (128, the bias index of R1 for half B's MULT.EE, seeded;
    1, the output row stride, unseeded) with the output-pointer advance that
    reads it (LR1 += CR1 seeded, LR1 += LR5 unseeded), and the activation.

    Data layout (packs "chunked", "conv_weight_groups", "pass0_bias")
    - x and y are padded to cols x rows and chunk-interleaved: cols is the
      next power of two >= width (a divisor of 128, so a spatial row never
      straddles a chunk), and row_groups = ceil(height * cols / 128) chunks
      per channel; chunk g of channel c is row g * channels + c. x is pinned
      at row 0, so CR0 is its base (no headroom: a 1x1 conv has no
      neighbourhood to prefetch).
    - w [out, in, 1, 1]: out_channels * num_passes rows, row o * P + p
      holding input channels p * 128 .. p * 128 + 127 of output channel o
      (the tail zero-padded).
    - b [out] (seeded only): the same grid as w, the bias at element 0 of
      each output channel's pass-0 row, so LR12 (the weight row offset)
      indexes both regions.
    - mask: slot 0 keeps every lane; the program never masks.

    Units
    - x, w, b, mask, y, in_channels (the row-group stride) and pass_stride
      (128 rows) are XMEM rows; one input channel of one chunk is one row,
      so the XMEM unit stride is the hardwired CR1 (chained ADD ... CR1
      words where a +2 or +3 row offset is needed: ADD's src_a reads the
      word-start snapshot).
    - fixed_idx_step (128) and the LR6/LR8/LR9 slot offsets are R_CYCLIC or
      lane ELEMENT offsets: the ring is 512 elements in every format.
    - num_passes and tail_size (unseeded CR10) are written but never read.

    LR allocation
      LR0 input row-group base row, LR1 output row (from -1), LR2 row-group
      counter, LR3 output channel counter, LR4 weight index (0..127 half A,
      128..255 half B), LR5 128 seeded / 1 unseeded, LR6 = 128, LR7 = 0
      (slot S0), LR8 = 256, LR9 = 384, LR10 pass counter (0 in the last
      pass), LR11 row_groups, LR12 weight row offset, LR13 the next pass's
      input base, LR14 scratch (the streaming load address), LR15 the pass's
      pipeline bound.
    - LR0 and LR7 are read before they are written: LR0 is only ever
      advanced (by in_channels per row group), and the first word's LD.MASK
      and ADD LR5 read LR7 in the word whose SET writes it. Both started at
      the reset value 0, so [abi] binds them to 0 (input_row_base,
      cyclic_s0): a run after another kernel starts from the same state.

    Supported configurations: width 1..128, any height >= 1 (padded),
    in_channels a multiple of 8, out_channels a multiple of 4.
-#}
{#- The variants: seeded (apply_relu) seeds R_ACC from the bias row; apply_relu picks the activation. -#}
{%- set seeded = apply_relu -%}
{%- set activation = "relu" if apply_relu else "identity" -%}
# ===========================================================================
# Initialization
# ===========================================================================

    SET                 LR7 CR0;
    LD.MASK   LR7 {{ cr.mask }};
{%- if seeded %}
    ADD                 LR5 LR7 {{ cr.fixed_idx_step }};;  # LR5 = 128 (MULT.EE ra_idx, Half B bias)
{%- else %}
    ADD                 LR5 LR7 CR1;;  # LR5 = 1 (output row stride)
{%- endif %}

    ADD                 LR6 LR7 {{ cr.fixed_idx_step }};;  # LR6 = 128 (R_CYCLIC ELEMENT index)
    ADD                 LR8 LR6 LR6;;  # LR8 = 256 (R_CYCLIC ELEMENT index)

    ADD                 LR9 LR6 LR8;;  # LR9 = 384 (R_CYCLIC ELEMENT index)

    ADD                 LR11 LR7 {{ cr.row_groups }};;  # LR11 = row_groups

    SET                 LR1 CR0;;
    SUB                 LR1 LR1 CR1;;  # LR1 = -1 row (pre-offset)

    SET                 LR2 CR0;;  # LR2 = row-group counter

# ===========================================================================
# Row-group loop
# ===========================================================================

row_loop:

    # Pre-load first 4 input channels into R_CYCLIC S0..S3
    LD.MULT.CYCLIC LR0 CR0 LR7;;

    ADD                 LR14 LR0 CR1;  # LR14 = LR0 + 1 row (channel 1's row)
    LD.MULT.CYCLIC LR14 CR0 LR6;;

    ADD                 LR14 LR14 CR1;;  # LR14 = LR0 + 2 rows (channel 2's row)
    LD.MULT.CYCLIC LR14 CR0 LR8;;

    ADD                 LR14 LR14 CR1;;  # LR14 = LR0 + 3 rows (channel 3's row)
    LD.MULT.CYCLIC LR14 CR0 LR9;;

    SET                 LR3 CR0;
    SET                 LR12 CR0;;  # LR3 = OC counter, LR12 = kernel offset

# ===========================================================================
# OC pair loop — process one OC per half (A then B)
# ===========================================================================

oc_pair_loop:
# ---------------------------------------------------------------------------
# Half A — process OC = LR3 using R0
# ---------------------------------------------------------------------------
{%- if seeded %}

    # Folded BN bias seed.  LR12 points at this OC's pass-0 block; CR10 (bias
    # base) indexes the parallel bias region, so the bias block lands in R0
    # with the OC's bias in element 0.  Broadcast it (× CR1=1) and seed R_ACC via
    # acc.first; THEN load the real weights over R0.
    LD.MULT        R0 LR12 {{ cr.b }};;  # bias block -> R0 (bias at element 0)

    MULT.EE             LR7 CR1 0 LR7 CR15 ;  # bias = R0[0] * CR1(=1), broadcast
    ACC.ADD.FIRST;;  # R_ACC = bias (conv taps add on top)
{%- endif %}

    LD.MULT        R0 LR12 {{ cr.w }};;  # load OC's pass-0 weights into R0

    SET                 LR4 CR0;
    ADD                 LR10 LR7 {{ cr.last_pass }};
    ADD                 LR13 LR0 LR7;;  # LR13 = LR0 (pass 0 input base)

    # Choose first-pass W1: full vs tail (single-pass case)
    BLT.S                 LR7 LR10 W1_FULL_A;;
{%- if seeded %}

    # num_passes == 1 (only the tail pass).  First conv tap is plain acc now —
    # bias was already seeded via acc.first above.
    # LR14 = LR0 + 3 rows; chained CR1 adds since ADD's src_a reads snapshot.
{%- else %}

    # num_passes == 1 (only the tail pass)
    # LR14 = LR0 + 3 rows (channel 3's row); chained CR1 adds since ADD's
    # src_a reads snapshot (can't fold into one word -- see header note).
{%- endif %}
    ADD                 LR14 LR0 CR1;
    ADD                 LR15 LR4 {{ cr.pipeline_limit_tail }};;
    ADD                 LR14 LR14 CR1;;
    ADD                 LR14 LR14 CR1;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
{%- if seeded %}
    ACC.ADD;;
{%- else %}
    ACC.ADD.FIRST;;
{%- endif %}

    BLT.S                 LR4 LR15 PIPELINE_BODY_A;;
    B                   POST_BODY_A;;

W1_FULL_A:
    ADD                 LR14 LR0 CR1;
    ADD                 LR15 LR4 {{ cr.pipeline_limit_full }};;
    ADD                 LR14 LR14 CR1;;
    ADD                 LR14 LR14 CR1;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
{%- if seeded %}
    ACC.ADD;;
{%- else %}
    ACC.ADD.FIRST;;
{%- endif %}

# ---------------------------------------------------------------------------
# Pipeline body — 4 cycles per iter, 4 ICs per iter
# ---------------------------------------------------------------------------

PIPELINE_BODY_A:

    INC                 LR4 1;
    ADD                 LR14 LR14 CR1;
    LD.MULT.CYCLIC LR14 CR0 LR7;
    MULT.RC.VE          LR6 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR14 CR1;
    LD.MULT.CYCLIC LR14 CR0 LR6;
    MULT.RC.VE          LR8 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR14 CR1;
    LD.MULT.CYCLIC LR14 CR0 LR8;
    MULT.RC.VE          LR9 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR14 CR1;
    LD.MULT.CYCLIC LR14 CR0 LR9;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
    ACC.ADD;
    BLT.S                 LR4 LR15 PIPELINE_BODY_A;;

# ---------------------------------------------------------------------------
# Post-body: decide which epilogue to enter (LAST = end-of-OC, MID = more passes)
# ---------------------------------------------------------------------------

POST_BODY_A:
    BLT.S                 LR7 LR10 MID_EPILOGUE_A;;  # LR10 > 0 → more passes

# ---------------------------------------------------------------------------
# LAST_EPILOGUE_A: this pass is the OC's last. Reload cyclic from LR0
# (so next OC starts with ich0..3). aaq+store, then advance to next OC.
# ---------------------------------------------------------------------------

LAST_EPILOGUE_A:

    INC                 LR4 1;
    LD.MULT.CYCLIC LR0 CR0 LR7;
    MULT.RC.VE          LR6 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR0 CR1;  # LR14 = LR0 + 1 row
    LD.MULT.CYCLIC LR14 CR0 LR6;
    MULT.RC.VE          LR8 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    # LR14 = LR0 + 2 rows; needs its own word (ADD src_a reads snapshot).
    ADD                 LR14 LR14 CR1;;
    INC                 LR4 1;
    LD.MULT.CYCLIC LR14 CR0 LR8;
    MULT.RC.VE          LR9 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    # LR14 = LR0 + 3 rows; own word, same reason (this word's 3 LR sub-slots
    # below are already full with the next-OC preload + output-ptr advance).
    ADD                 LR14 LR14 CR1;;
{%- if seeded %}

    # ACTIVATE (relu) applies ReLU to the just-finalized R_ACC (conv + bias)
    # and copies it -> post_aaq_reg.  Reads R_ACC LIVE so
    # it does not NEED a cycle after the final acc, but this word's 3 LR
    # sub-slots are already full with the next-OC preload — rides here
    # (see the unseeded program's note below for why fusing
    # further would need LR-slot reshuffling not attempted here).
{%- else %}

    # ACTIVATE.QUANTIZE (identity) activates the just-finalized
    # R_ACC -> 128 FP32 elements in post_aaq_reg.  It reads R_ACC LIVE,
    # so it does not NEED a cycle after the final acc, but
    # this word's 3 LR sub-slots are already full with the next-OC preload,
    # so ACTIVATE rides here rather than fusing
    # into the final acc's word above (which has only 1 free LR slot — not
    # enough room to also relocate this preload).  Fusing further would need
    # LR-slot reshuffling; not attempted here. ST.AAQ (next word)
    # also reads live, but this word's LR slots are already full, so the
    # store stays separate too.
{%- endif %}
    INC                 LR4 1;
{%- if seeded %}
    ADD                 LR1 LR1 CR1;
{%- else %}
    ADD                 LR1 LR1 LR5;
{%- endif %}
    LD.MULT.CYCLIC LR14 CR0 LR9;
    ACTIVATE.QUANTIZE {{ activation }} CR15;;
{%- if seeded %}

    # Store the 128 activated FP32 elements to XMEM.
{%- else %}

{%- endif %}
    INC                 LR3 1;
    ADD                 LR12 LR12 CR1;
    ST.AAQ    LR1 {{ cr.y }};;

    # Fall through to Half B section

# ---------------------------------------------------------------------------
# Half B — process OC = LR3 using R1 (fixed_idx 128..255)
# ---------------------------------------------------------------------------

    # Bounds check: skip B half if no more OCs
    ADD                 LR14 LR7 {{ cr.out_channels }};;
    BLT.S                 LR3 LR14 HALF_B_GO;;
    B                   END_ROW_GROUP_CHECK;;

HALF_B_GO:
{%- if seeded %}

    # Folded BN bias seed (half B).  Bias block -> R1; the bias element sits at
    # R1's element 0 = combined Ra index 128, so MULT.EE reads index LR5 (=128).
    LD.MULT        R1 LR12 {{ cr.b }};;  # bias block -> R1 (bias at element 0)

    MULT.EE             LR5 CR1 0 LR7 CR15 ;  # bias = R1[0] * CR1(=1), broadcast
    ACC.ADD.FIRST;;  # R_ACC = bias
{%- endif %}

    LD.MULT        R1 LR12 {{ cr.w }};;

    ADD                 LR4 LR7 {{ cr.fixed_idx_step }};
    ADD                 LR10 LR7 {{ cr.last_pass }};
    ADD                 LR13 LR0 LR7;;  # LR13 = LR0 (pass 0 input base)

    BLT.S                 LR7 LR10 W1_FULL_B;;

    ADD                 LR14 LR0 CR1;
    ADD                 LR15 LR4 {{ cr.pipeline_limit_tail }};;
    ADD                 LR14 LR14 CR1;;
    ADD                 LR14 LR14 CR1;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
{%- if seeded %}
    ACC.ADD;;
{%- else %}
    ACC.ADD.FIRST;;
{%- endif %}

    BLT.S                 LR4 LR15 PIPELINE_BODY_B;;
    B                   POST_BODY_B;;

W1_FULL_B:
    ADD                 LR14 LR0 CR1;
    ADD                 LR15 LR4 {{ cr.pipeline_limit_full }};;
    ADD                 LR14 LR14 CR1;;
    ADD                 LR14 LR14 CR1;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
{%- if seeded %}
    ACC.ADD;;
{%- else %}
    ACC.ADD.FIRST;;
{%- endif %}

PIPELINE_BODY_B:

    INC                 LR4 1;
    ADD                 LR14 LR14 CR1;
    LD.MULT.CYCLIC LR14 CR0 LR7;
    MULT.RC.VE          LR6 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR14 CR1;
    LD.MULT.CYCLIC LR14 CR0 LR6;
    MULT.RC.VE          LR8 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR14 CR1;
    LD.MULT.CYCLIC LR14 CR0 LR8;
    MULT.RC.VE          LR9 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR14 CR1;
    LD.MULT.CYCLIC LR14 CR0 LR9;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
    ACC.ADD;
    BLT.S                 LR4 LR15 PIPELINE_BODY_B;;

POST_BODY_B:
    BLT.S                 LR7 LR10 MID_EPILOGUE_B;;

LAST_EPILOGUE_B:

    INC                 LR4 1;
    LD.MULT.CYCLIC LR0 CR0 LR7;
    MULT.RC.VE          LR6 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR0 CR1;  # LR14 = LR0 + 1 row
    LD.MULT.CYCLIC LR14 CR0 LR6;
    MULT.RC.VE          LR8 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    # LR14 = LR0 + 2 rows; own word (ADD src_a reads snapshot).
    ADD                 LR14 LR14 CR1;;
    INC                 LR4 1;
    LD.MULT.CYCLIC LR14 CR0 LR8;
    MULT.RC.VE          LR9 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    # LR14 = LR0 + 3 rows; own word, same reason (the word below is already
    # full with the next-OC preload + output-ptr advance).
    ADD                 LR14 LR14 CR1;;
{%- if seeded %}

    # ACTIVATE (relu): ReLU the finalized R_ACC (conv + bias) -> post_aaq_reg.
    # Reads R_ACC LIVE; rides in the preload word
    # (same LR-slot constraint as Half A's identical note).
{%- else %}

    # ACTIVATE.QUANTIZE (identity): activate the finalized R_ACC ->
    # 128 FP32 elements in post_aaq_reg.  Reads R_ACC LIVE so
    # it does not NEED a cycle after the final acc, but this word's 3 LR
    # sub-slots are already full with the next-OC preload (no room to also
    # relocate that preload into the final acc's word, which has only 1 free
    # LR slot) — rides here.  See Half A's identical note.
{%- endif %}
    INC                 LR4 1;
{%- if seeded %}
    ADD                 LR1 LR1 CR1;
{%- else %}
    ADD                 LR1 LR1 LR5;
{%- endif %}
    LD.MULT.CYCLIC LR14 CR0 LR9;
    ACTIVATE.QUANTIZE {{ activation }} CR15;;

    # Store the 128 activated FP32 elements to XMEM.
    INC                 LR3 1;
    ADD                 LR12 LR12 CR1;
    ST.AAQ    LR1 {{ cr.y }};;

# ---------------------------------------------------------------------------
# End of OC-pair: check if more OCs in this row-group
# ---------------------------------------------------------------------------

END_ROW_GROUP_CHECK:

    ADD                 LR14 LR7 {{ cr.out_channels }};;
    BLT.S                 LR3 LR14 oc_pair_loop;;

# ---------------------------------------------------------------------------
# Advance to next row-group
# ---------------------------------------------------------------------------

    ADD                 LR0 LR0 {{ cr.in_channels }};
    INC                 LR2 1;;

    BLT.S                 LR2 LR11 row_loop;;

end:
    HALT;;

# ===========================================================================
# MID epilogues (more passes follow): reload cyclic from NEXT-pass base,
# then transition to next pass with kernel reload.
# ===========================================================================

MID_EPILOGUE_A:

    # Compute LR13 = LR0 + (current_pass_index + 1) * 16384
    # We don't track current_pass_index directly; we track LR10 = passes_remaining-1.
    # current_pass_index = num_passes - 1 - LR10.
    # next-pass base offset from LR0 = (current_pass_index + 1) * 16384
    #                                = (num_passes - LR10) * 16384.
    # Compute incrementally: maintain LR13 across passes.
    #
    # Simpler: derive LR13 from how many passes we have YET to do.
    # passes_done_after_this = num_passes - LR10. Next pass's base offset
    # from LR0 = (num_passes - LR10) * 16384.
    # We don't have num_passes*16384 in a register; compute via CR4 * CR13.
    # No mult instruction for LR/CR arithmetic. So:
    #
    # Track via increment: at start of OC, set LR13 = LR0. Each MID_EPILOGUE,
    # advance LR13 += CR13. This requires LR13 init somewhere — we do it
    # below in the kernel-load transition (set LR13 = LR0 + CR13 directly
    # since this is the FIRST mid-transition, where current_pass_index=0).
    #
    # Actually we need LR13 to track CURRENT pass's base so we can advance
    # by CR13 to get next pass's base. The "current pass base" for pass 0
    # is LR0. For pass 1 it's LR0+16384. Etc.
    #
    # Since this is hard to track without a dedicated init, use a separate
    # approach: maintain LR13 = current pass's base, initialized to LR0 at
    # OC start, advanced by CR13 each MID transition AFTER the epilogue uses
    # the previous value... no wait, we need NEXT-pass base for E1-E4.
    #
    # SIMPLEST: just compute LR13 = LR0 + CR13 * (num_passes - LR10) here.
    # But we have no multiply. So track incrementally — initialize at OC start
    # and advance per pass. (Adding init at OC start below in __init__-style.)

    # Advance LR13 to next pass's input base.
    # (LR13 was set to LR0 at OC entry; we add CR13 each mid-transition.)
    ADD                 LR13 LR13 {{ cr.pass_stride }};;

    # E1-E4 with reloads from LR13 (next pass's ich0..ich3)
    INC                 LR4 1;
    LD.MULT.CYCLIC LR13 CR0 LR7;
    MULT.RC.VE          LR6 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR13 CR1;  # LR14 = LR13 + 1 row
    LD.MULT.CYCLIC LR14 CR0 LR6;
    MULT.RC.VE          LR8 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    # LR14 = LR13 + 2 rows; own word (ADD src_a reads snapshot).
    ADD                 LR14 LR14 CR1;;
    INC                 LR4 1;
    LD.MULT.CYCLIC LR14 CR0 LR8;
    MULT.RC.VE          LR9 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    # LR14 = LR13 + 3 rows; own word, same reason.
    ADD                 LR14 LR14 CR1;;
    INC                 LR4 1;
    LD.MULT.CYCLIC LR14 CR0 LR9;;

    # Transition: advance kernel pointer + decrement pass counter + load R0
    ADD                 LR12 LR12 CR1;
    SUB                 LR10 LR10 CR1;;  # pass_counter -= 1

    LD.MULT        R0 LR12 {{ cr.w }};;

    # Reset LR4 = 0
    SET                 LR4 CR0;;

    # Select pipeline_limit: full or tail
    BLT.S                 LR7 LR10 MP_A_FULL;;

    # Last pass for this OC: use tail limit
    ADD                 LR14 LR13 CR1;
    ADD                 LR15 LR4 {{ cr.pipeline_limit_tail }};;
    ADD                 LR14 LR14 CR1;;
    ADD                 LR14 LR14 CR1;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    BLT.S                 LR4 LR15 PIPELINE_BODY_A;;
    B                   POST_BODY_A;;

MP_A_FULL:
    ADD                 LR14 LR13 CR1;
    ADD                 LR15 LR4 {{ cr.pipeline_limit_full }};;
    ADD                 LR14 LR14 CR1;;
    ADD                 LR14 LR14 CR1;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    B                   PIPELINE_BODY_A;;

MID_EPILOGUE_B:

    ADD                 LR13 LR13 {{ cr.pass_stride }};;

    INC                 LR4 1;
    LD.MULT.CYCLIC LR13 CR0 LR7;
    MULT.RC.VE          LR6 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    INC                 LR4 1;
    ADD                 LR14 LR13 CR1;  # LR14 = LR13 + 1 row
    LD.MULT.CYCLIC LR14 CR0 LR6;
    MULT.RC.VE          LR8 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    # LR14 = LR13 + 2 rows; own word (ADD src_a reads snapshot).
    ADD                 LR14 LR14 CR1;;
    INC                 LR4 1;
    LD.MULT.CYCLIC LR14 CR0 LR8;
    MULT.RC.VE          LR9 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    # LR14 = LR13 + 3 rows; own word, same reason.
    ADD                 LR14 LR14 CR1;;
    INC                 LR4 1;
    LD.MULT.CYCLIC LR14 CR0 LR9;;

    ADD                 LR12 LR12 CR1;
    SUB                 LR10 LR10 CR1;;

    LD.MULT        R1 LR12 {{ cr.w }};;

    ADD                 LR4 LR7 {{ cr.fixed_idx_step }};;  # LR4 = 128

    BLT.S                 LR7 LR10 MP_B_FULL;;

    ADD                 LR14 LR13 CR1;
    ADD                 LR15 LR4 {{ cr.pipeline_limit_tail }};;
    ADD                 LR14 LR14 CR1;;
    ADD                 LR14 LR14 CR1;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    BLT.S                 LR4 LR15 PIPELINE_BODY_B;;
    B                   POST_BODY_B;;

MP_B_FULL:
    ADD                 LR14 LR13 CR1;
    ADD                 LR15 LR4 {{ cr.pipeline_limit_full }};;
    ADD                 LR14 LR14 CR1;;
    ADD                 LR14 LR14 CR1;
    MULT.RC.VE          LR7 LR4 0 LR7 CR15 ;
    ACC.ADD;;

    B                   PIPELINE_BODY_B;;
