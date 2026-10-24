{#- depthwise_conv_tpl.asm -- y = act(depthwise conv2d(x, w, 3x3, stride 1, padding 1) + b): one 3x3 filter per channel (groups = channels) in FP32 wide vectors, with an optional folded bias and identity or ReLU.

    The template of kernels/convolutions/registry.toml's
    depthwise_conv_universal (has_bias = false, apply_relu = false, fpb = 28),
    depthwise_conv_universal_bn_activation (has_bias = true,
    apply_relu = true, fpb = 25), and, E07-20, depthwise_conv_universal_bias
    (true, false, 25) and depthwise_conv_universal_relu (false, true, 25: a
    zero bias). Batch-norm is assumed folded into the
    weights and the bias. The program reads the query's own parameters:
    "seeded" (has_bias or apply_relu, as conv_universal_tpl) selects the bias
    seed, apply_relu the activation. It is also stage 1 of the
    depthwise_conv_stride2_* kernels (E07-19), so it reads only has_bias,
    apply_relu and the [abi] names those kernels declare too.

    Pipeline per channel (a walking pointer over a rotating cyclic ring)
    - Unseeded, 9 cycles: the 9 taps (MULT.RC.VE from the ring), tap 1
      starting R_ACC (ACC.ADD.FIRST).
    - Seeded, 10 cycles: cycle 0 seeds R_ACC with the channel's bias
      (MULT.EE of its bias element times CR1, ACC.ADD.FIRST), and the 9 taps
      accumulate on it.
    - In both, tap 9's final ACC.ADD and ACTIVATE.EXACT (relu when
      apply_relu, else identity) co-issue in one word (AAQ_EXACT reads R_ACC
      forwarded, after ACC), and the activate stores the channel's row at
      y + LR7: LR7 starts at -1 and tap 2 advances it, so channel K's row
      is y + K.
    The variants differ in the 3 sections (chunk 0, chunks 1..N-2, chunk N-1)
    in the same places: the seed word; ACC.ADD vs ACC.ADD.FIRST at tap 1;
    the cycle that prefetches the channel's kr = -1 chunk (seeded: its own
    cycle 0; unseeded: fused into the previous channel's tap 9, its address
    computed at tap 8 or, in chunk N-1, tap 7); and so the taps where the
    channel counter (LR10) and the read-pointer rotation (LR4) sit, which
    move to keep the LR slots of each word within three.

    Filter super-blocks (pack "depthwise_filters", input filters [channels, 10]: the bias, then the 9 taps)
    - One 256-element super-block (2 XMEM rows: R0 half, R1 half) holds fpb
      channels; LD.MULT of both rows, and MULT's shared index (0..255)
      addresses all of them. LR6 walks the index +1 a cycle.
    - Unseeded: channel s at elements [9s, 9s + 9), its 9 taps; fpb = 28
      (252 <= 256). The bias column is not stored.
    - Seeded: channel s at [10s, 10s + 10), the bias then its 9 taps;
      fpb = 25 (250 <= 256). The bias element holds the bias column only
      when has_bias, else 0.
    - fpb (CR7) is a register value that follows the seed, but a bool cannot
      enter [derived]: each entry fixes it, and the pack refuses an fpb that
      disagrees with the seed. kernel_rows (CR8, ceil(in_channels / fpb) * 2)
      is written but never read.

    Data layout (packs "chunked_guard", "chunked")
    - width pads to cols in {16, 32, 64, 128}: one packed spatial row per
      DS partition, so CR15.L is ds(cols, 128 / cols, 128 / cols) (one
      partition at cols = 128); CR15.H is the case's QF. height pads to rows >= ceil(256 / cols): at least 2
      chunks, for the first and the last chunk section.
    - 128 / cols spatial rows share one 128-element chunk (one XMEM row);
      chunks are grouped by channel: row (chunk * channels + ch).
    - The input region starts with in_channels zero rows (one group stride
      of headroom), so chunk 0's kr = -1 prefetch, at LR2 - in_channels,
      bottoms out at the region base instead of underflowing. LR8 (the
      chunk base) starts at in_channels, and chunk_limit is pre-shifted by
      the same stride: (num_chunks - 1) * in_channels + in_channels.

    Units
    - x, filters, border_mask, y, in_channels (the group stride), fpb,
      kernel_rows and chunk_limit are XMEM rows. One chunk is one row, so
      the row-space chunk stride is the hardwired CR1 (the 2-row
      super-block step is two ADD ... CR1 words).
    - slot_stride (128), half_slot (256), ring_adv (384) and walk_step are
      R_CYCLIC ELEMENT offsets: the ring is 512 elements in every format,
      so they are mode-blind and must not be scaled.
    - Ring rotation: the read pointer LR4 (the kr = 0 slot) rotates +256
      (mod 512) a channel; the write pointer LR5, advanced only by
      INC.MODP2 (mod 512), steps +384, +128 and +256 a channel (net +256),
      and is not recomputed from LR4. walk_step = 256 - 2 * cols - 2 brings
      the walk pointer LR3 from this channel's tap 9 to the next channel's
      tap 1.

    Border masks (scratch border_mask, one row, ../masks.py:border_mask)
    - One R_MASK blob, loaded once at init, carries three 16-byte slots; a
      bit of 1 KEEPS the lane. Slot 0 keeps every lane (interior and kr = 0
      taps), slot 3 zeroes packed row 0 (chunk 0's kr = -1 taps 1-3), slot 6
      zeroes the last packed row (chunk N-1's kr = +1 taps 7-9).
    - Left/right edge columns are zeroed by mask_shift, with
      the DS's 128 / cols partitions: kc = -1 shifts by +1 (LR9, zeroes each
      row's first column), kc = 0 by 0 (LR0), kc = +1 by -1 (LR13, zeroes
      each row's last column).

    LR allocation
      LR0 = 0, LR1 = cols - 2, LR2 this channel's kr = 0 row, LR3 walk
      pointer, LR4 read slot, LR5 write slot, LR6 filter element index, LR7
      output row (from -1), LR8 chunk base row, LR9 = +1 (mask shift), LR10
      channel row counter, LR11 the super-block's channel limit, LR12
      super-block row offset, LR13 = -1 (mask shift), LR14 scratch. LR15 is
      free. No LR is read before it is written.

    Supported configurations: width 1..128, any height >= 1 (padded),
    channels >= 1 (in_channels = out_channels = groups).
    KNOWN LATENT BUG (not fixed here; D3, as conv_universal_tpl):
    num_chunks = rows * cols // 128 rounds down while the input pack writes
    ceil(rows / (128 / cols)) chunks of every channel, so a shape with
    rows * cols not a multiple of 128 (height 17 at width 16) computes only
    num_chunks chunks. No case uses such a shape.
-#}
{#- The variants: seeded (has_bias or apply_relu) seeds R_ACC from the bias
    element; apply_relu picks the activation. -#}
{%- set seeded = has_bias or apply_relu -%}
{%- set activation = "relu" if apply_relu else "identity" -%}
# ===========================================================================
# Initialization
# ===========================================================================

    SET                 LR0 CR0;
    LD.MASK   LR0 {{ cr.border_mask }};;  # load mask blob (slots 0/3/6)

    ADD                 LR1 LR0 {{ cr.cols }};  # LR1 = cols (temp)
    # Guard band: LR8 (chunk base, ROW-space) seeds to group_stride (not 0) so
    # g0's kr=-1 prefetch (LR2 - CR6, computed off LR8) bottoms out at exactly
    # row 0 instead of underflowing; CR10/CR11 are biased to match (harness).
    ADD                 LR8 LR0 {{ cr.in_channels }};
    SET                 LR7 CR0;;

    # LR7 = -1 row so ch 0's tap-2 advance lands at row 0, its output row.
    SUB                 LR7 LR7 CR1;;  # LR7 = -1

    DEC                 LR1 2;  # LR1 = cols - 2 (permanent)
    ADD                 LR9 LR0 CR1;  # LR9 = +1 (mask_shift for kc=-1, permanent)
    SUB                 LR13 LR0 CR1;;  # LR13 = -1 (mask_shift for kc=+1, permanent)

# ===========================================================================
# Section 1: Chunk 0 (top border) — kr=-1 row masked (slots 3/4/5).
# ===========================================================================

    SET                 LR12 CR0;;

g0_kg_loop:
    LD.MULT        R0 LR12 {{ cr.filters }};
    SET                 LR6 CR0;;

    ADD                 LR14 LR12 CR1;
    SET                 LR10 CR0;
    LD.MULT        R1 LR14 {{ cr.filters }};;

    ADD                 LR11 LR10 {{ cr.fpb }};;
    BLT.S                 {{ cr.in_channels }} LR11 g0_clamp;;
    B                   g0_ch_pre;;
g0_clamp:
    ADD                 LR11 LR0 {{ cr.in_channels }};;

g0_ch_pre:
    # Preamble (4 cyc): set up rotating slots for first ch of this kernel-group.

    # Cy 1: LR4 = 0; LR5 = 0; LR2 = LR8+LR10 (THIS-ch kr=0 ext); load into slot 0.
    SET                 LR4 CR0;
    SET                 LR5 CR0;
    ADD                 LR2 LR8 LR10;
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;;

    # Cy 2: LR5 = 128; ext = LR2+CR6 (kr=+1); LR3 = 1 (seed for -255).
    ADD                 LR5 LR5 {{ cr.slot_stride }};
    ADD                 LR14 LR2 {{ cr.in_channels }};
    SET                 LR3 CR1;
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;;

    # Cy 3: LR3 = 1-256 = -255; LR5 = 128+256 = 384.
    SUB                 LR3 LR3 {{ cr.half_slot }};
    ADD                 LR5 LR5 {{ cr.half_slot }};;
{%- if seeded %}

    # Cy 4: walk seed += cols → cols-255.  LR6 = -1 (bias cycle's +1 -> 0 = ch0
    # bias element index; the 10-cycle body then advances LR6 by exactly 10/channel).
{%- else %}

    # Cy 4: walk seed += cols → cols-255.  LR6 = -1 (tap 1's +1 -> 0 = ch0's
    # first tap element index; the 9-cycle body then advances LR6 by exactly
    # 9/channel).
{%- endif %}
    ADD                 LR3 LR3 {{ cr.cols }};
    SUB                 LR6 LR0 CR1;;

    B                   g0_tap_body;;

g0_tap_body:
{%- if seeded %}
    # 10 cyc/ch.  cyc0 = bias seed; taps 1..9 = weights, ACTIVATE.EXACT
    # fused into tap 9's own cycle (see below).  G0: kr=-1 row masked.
    #
    # Store (no extra cycle):
    #   ch K   tap 2: LR7 advance to K
    #   ch K   tap 9: acc (final) + ACTIVATE.EXACT {{ activation }} -> y[K] = act(R_ACC)

    # --- cyc 0: BIAS seed + activate the PREVIOUS channel.
    # LR6 += 1 -> s*10 (this ch's bias element index; the +1 here plus +1 on each of
    # taps 1..9 = +10 total per channel = one 10-element channel slot).  Bias's
    # MULT.EE reads Ra[LR6]*CR1=1, broadcasting the bias element to all lanes.
    # LR3 is NOT walked here; tap 1 keeps the unseeded program's walk_step.
    # Also pre-load this ch's kr=-1 (own valid base LR2) into slot LR5 here, one
    # cycle before tap 1 reads it (snapshot; cyc-0's LOAD slot is free).
    INC                 LR6 1;
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;
    MULT.EE             LR6 CR1 0 LR0 CR15.L CR15.H ;
    ACC.ADD.FIRST;;
{%- endif %}
{%- if seeded %}

    # --- tap 1: kr=-1 kc=-1.  Top row out of bounds: slot 3; kc=-1 shift (LR9)
    #     zeros the left edge column.  kr=-1 (own valid base LR2) pre-loaded in
    #     cyc 0; row-0 lanes masked.  Walk +CR14, as unseeded.
{%- else %}
    # 9 cyc/ch.  Taps 1..9 = weights; ACTIVATE.EXACT fused into tap 9's own
    # cycle (see tap 9 below).  G0: kr=-1 row masked.
    #
    # Store (no extra cycle):
    #   ch K   tap 2: LR7 advance to K
    #   ch K   tap 9: acc (final) + ACTIVATE.EXACT identity -> y[K] = R_ACC

    # --- tap 1: kr=-1 kc=-1.  Top row out of bounds: slot 3; kc=-1 shift (LR9)
    #     zeros the left edge column.  kr=-1 pre-loaded by g0_ch_pre (ch 0) or
    #     the previous channel's tap 9 (see below).  Walk +CR14.
    #     acc.add.first resets R_ACC for this channel (replaces the seeded program's
    #     bias seed).  Loop counter += CR12 moved here (tap 1 has a free LR
    #     slot) to make room at tap 6/8 for the NEXT-ch kr=-1 ext calc.
{%- endif %}
    ADD                 LR3 LR3 {{ cr.walk_step }};
    INC                 LR6 1;
{%- if not seeded %}
    ADD                 LR10 LR10 CR1;
{%- endif %}
    MULT.RC.VE          LR3 LR6 3 LR9 CR15.L CR15.H ;
{%- if seeded %}
    ACC.ADD;;
{%- else %}
    ACC.ADD.FIRST;;
{%- endif %}

    # --- tap 2: kr=-1 kc=0.  slot 3 = top row only, no shift.  LR7 advance: this channel's output row.
    INC                 LR3 1;
    INC                 LR6 1;
    ADD                 LR7 LR7 CR1;
    MULT.RC.VE          LR3 LR6 3 LR0 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 3: kr=-1 kc=+1.  slot 3 + kc=+1 shift (LR13).  Advance LR2 → NEXT-ch kr=0 ext.
    INC                 LR3 1;
    INC                 LR6 1;
    ADD                 LR2 LR2 CR1;
    MULT.RC.VE          LR3 LR6 3 LR13 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 4: kr=0 kc=-1.  slot 0 + kc=-1 shift.  LR5 += 384 (mod 512) → R+256;
    #     load NEXT-ch kr=0 → slot LR5.
    ADD                 LR3 LR3 LR1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.ring_adv }} 9;
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;
    MULT.RC.VE          LR3 LR6 0 LR9 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 5: kr=0 kc=0.  slot 0, no shift.  LR5 += 128 (mod 512) → R+384.
    INC                 LR3 1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.slot_stride }} 9;
    MULT.RC.VE          LR3 LR6 0 LR0 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 6: kr=0 kc=+1.  slot 0 + kc=+1 shift.  Loop counter += 128.
{%- else %}

    # --- tap 6: kr=0 kc=+1.  slot 0 + kc=+1 shift.  Rotate lr_read (moved
    #     from tap 8 to free its LR slot for the NEXT-ch kr=-1 ext calc).
{%- endif %}
    INC                 LR3 1;
    INC                 LR6 1;
{%- if seeded %}
    ADD                 LR10 LR10 CR1;
{%- else %}
    INC.MODP2       LR4 {{ cr.half_slot }} 9;
{%- endif %}
    MULT.RC.VE          LR3 LR6 0 LR13 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 7: kr=+1 kc=-1.  slot 0 + kc=-1 shift.  Load NEXT-ch kr=+1 → slot LR5 (= R+384).
    ADD                 LR3 LR3 LR1;
    INC                 LR6 1;
    ADD                 LR14 LR2 {{ cr.in_channels }};
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;
    MULT.RC.VE          LR3 LR6 0 LR9 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 8: kr=+1 kc=0.  slot 0, no shift.  Rotate lr_read.
{%- else %}

    # --- tap 8: kr=+1 kc=0.  slot 0, no shift.  Precompute LR14 = NEXT-ch
    #     kr=-1 ext (= LR2-CR6), overwriting tap 7's now-dead LR14 value, so
    #     tap 9's fused load can issue it directly.
{%- endif %}
    INC                 LR3 1;
    INC                 LR6 1;
{%- if seeded %}
    INC.MODP2       LR4 {{ cr.half_slot }} 9;
{%- else %}
    SUB                 LR14 LR2 {{ cr.in_channels }};
{%- endif %}
    MULT.RC.VE          LR3 LR6 0 LR0 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 9: kr=+1 kc=+1.  slot 0 + kc=+1 shift.  Final acc, fused with
    #     ACTIVATE.EXACT in the SAME cycle (reads R_ACC LIVE; ACC
    #     dispatches before AAQ_EXACT) — no standalone cycle needed.  LR5 += 256
    #     (mod 512) → R+128 = next ch's R'-128 (its tap-1 kr=-1 slot; already
    #     pre-loaded every channel by the NEXT channel's cyc0).  LR6 now =
    #     s*10 + 9; +1 at next cyc0 -> (s+1)*10.  Loop branch lives here too.
{%- else %}

    # --- tap 9: kr=+1 kc=+1.  slot 0 + kc=+1 shift.  Final acc, fused with
    #     ACTIVATE.EXACT in the SAME cycle (reads R_ACC LIVE; ACC
    #     dispatches before AAQ_EXACT) — no standalone cycle needed.  LR5 += 256
    #     (mod 512) → R+128 = next ch's R'-128 (its tap-1 kr=-1 slot); load
    #     NEXT-ch kr=-1 (LR14, from tap 8) there in the same word (XMEM reads
    #     LR5 LIVE).  LR6 now = s*9 + 8; +1 at next ch's tap 1 -> (s+1)*9.
    #     Loop branch lives here too.
{%- endif %}
    INC                 LR3 1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.half_slot }} 9;
{%- if not seeded %}
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;
{%- endif %}
    MULT.RC.VE          LR3 LR6 0 LR13 CR15.L CR15.H ;
    ACC.ADD;
    ACTIVATE.EXACT {{ activation }} CR15.L {{ cr.y }} LR7;
    BLT.S                 LR10 LR11 g0_tap_body;;

    # All channels in this kernel-group done; the last channel's ACTIVATE.EXACT
    # has stored its row.
    # SUPER_BLOCK_ROWS=2: advance LR12 (kernel super-block XMEM row) by 2
    # rows via CR1 (row-space); CR13 is R_CYCLIC ELEMENT-space (256) and
    # would double-count under row addressing if reused here. ADD's src_a
    # reads snapshot, so the second +CR1 needs its own cycle (this is a
    # group-boundary event, not the hot per-tap loop -- one extra cycle here
    # is not on the per-channel critical path).
    ADD                 LR12 LR12 CR1;;
    ADD                 LR12 LR12 CR1;
    BLT.S                 LR10 {{ cr.in_channels }} g0_reload;;

    B                   main_setup;;

g0_reload:
    LD.MULT        R0 LR12 {{ cr.filters }};
    SET                 LR6 CR0;;

    ADD                 LR14 LR12 CR1;
    LD.MULT        R1 LR14 {{ cr.filters }};;

    ADD                 LR11 LR10 {{ cr.fpb }};;
    BLT.S                 {{ cr.in_channels }} LR11 g0_reload_clamp;;
    B                   g0_ch_pre;;
g0_reload_clamp:
    ADD                 LR11 LR0 {{ cr.in_channels }};
    B                   g0_ch_pre;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop) — all rows real from CR10, no masks.
# ===========================================================================

main_setup:
    # LR8 already carries the guard-band group_stride from init; chunk 1's
    # true (biased) base is 2*group_stride, so ADD onto LR8, don't reset from LR0.
    ADD                 LR8 LR8 {{ cr.in_channels }};;
    BLT.S                 LR8 {{ cr.chunk_limit }} row_loop;;

    B                   gN_section;;

row_loop:
    SET                 LR12 CR0;;

kg_loop:
    LD.MULT        R0 LR12 {{ cr.filters }};
    SET                 LR6 CR0;;

    ADD                 LR14 LR12 CR1;
    SET                 LR10 CR0;
    LD.MULT        R1 LR14 {{ cr.filters }};;

    ADD                 LR11 LR10 {{ cr.fpb }};;
    BLT.S                 {{ cr.in_channels }} LR11 mn_clamp;;
    B                   ch_pre;;
mn_clamp:
    ADD                 LR11 LR0 {{ cr.in_channels }};;

ch_pre:
    # Cy 1: LR2 = LR8+LR10; load ch kr=0 → slot 0.
    SET                 LR4 CR0;
    SET                 LR5 CR0;
    ADD                 LR2 LR8 LR10;
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;;

    # Cy 2: LR5 = 128; LR3 = 1 (seed for -255); load ch kr=+1 → slot 128.
    ADD                 LR5 LR5 {{ cr.slot_stride }};
    ADD                 LR14 LR2 {{ cr.in_channels }};
    SET                 LR3 CR1;
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;;

    # Cy 3: LR3 = 1-256 = -255; LR5 = 128+256 = 384.
    SUB                 LR3 LR3 {{ cr.half_slot }};
    ADD                 LR5 LR5 {{ cr.half_slot }};;
{%- if seeded %}

    # Cy 4: walk seed += cols; LR6 = -1 (bias cycle +1 -> 0).
{%- else %}

    # Cy 4: walk seed += cols; LR6 = -1 (tap 1's +1 -> 0).  Prefetch THIS ch's
    # kr=-1 (ext = LR2-CR6) into its slot LR5 here, one cycle before tap 1
    # reads it — under snapshot a same-cycle LDR is not visible to the mult,
    # and this cycle's LOAD slot is free.
{%- endif %}
    ADD                 LR3 LR3 {{ cr.cols }};
{%- if seeded %}
    SUB                 LR6 LR0 CR1;;
{%- else %}
    SUB                 LR6 LR0 CR1;
    SUB                 LR14 LR2 {{ cr.in_channels }};
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;;
{%- endif %}

    B                   mn_tap_body;;

mn_tap_body:
{%- if seeded %}
    # 10 cyc/ch.  All loads use CR10; no border masks (slot 0 + shift).
    # ACTIVATE.EXACT fused into tap 9's own cycle (see below).

    # --- cyc 0: BIAS seed + activate the PREVIOUS channel (LR3 not walked here).
    #     Also pre-load THIS-ch kr=-1 (ext = LR2-CR6) into its slot LR5 here, one
    #     cycle before tap 1 reads it — under snapshot a same-cycle LDR is not
    #     visible to the mult, and cyc-0's LOAD slot is free.  LR5 is already the
    #     kr=-1 slot; LR2 already holds THIS-ch kr=0 ext.
    INC                 LR6 1;
    SUB                 LR14 LR2 {{ cr.in_channels }};
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;
    MULT.EE             LR6 CR1 0 LR0 CR15.L CR15.H ;
    ACC.ADD.FIRST;;
{%- endif %}
{%- if seeded %}

    # --- tap 1: kr=-1 kc=-1.  slot 0 + kc=-1 shift (LR9).  kr=-1 pre-loaded in
    #     cyc 0.  Walk +CR14.
{%- else %}
    # 10 cyc/ch.  All loads use CR10; no border masks (slot 0 + shift).

    # --- tap 1: kr=-1 kc=-1.  slot 0 + kc=-1 shift (LR9).  kr=-1 pre-loaded in
    #     the previous channel's tap 9 (see below — ACTIVATE.EXACT now
    #     reads R_ACC LIVE, so it fuses into tap 9's own cycle).  Walk +CR14.
    #     acc.add.first resets R_ACC.  Loop counter += CR12 moved here (tap 1
    #     has a free LR slot) to make room at tap 6/8 for the NEXT-ch kr=-1
    #     ext address calc + lr_read rotation swap (see tap 9).
{%- endif %}
    ADD                 LR3 LR3 {{ cr.walk_step }};
    INC                 LR6 1;
{%- if not seeded %}
    ADD                 LR10 LR10 CR1;
{%- endif %}
    MULT.RC.VE          LR3 LR6 0 LR9 CR15.L CR15.H ;
{%- if seeded %}
    ACC.ADD;;
{%- else %}
    ACC.ADD.FIRST;;
{%- endif %}

    # --- tap 2: kc=0, no shift.  LR7 advance: this channel's output row.
    INC                 LR3 1;
    INC                 LR6 1;
    ADD                 LR7 LR7 CR1;
    MULT.RC.VE          LR3 LR6 0 LR0 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 3: kc=+1 shift (LR13).  Advance LR2 → NEXT-ch kr=0 ext.
    INC                 LR3 1;
    INC                 LR6 1;
    ADD                 LR2 LR2 CR1;
    MULT.RC.VE          LR3 LR6 0 LR13 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 4: kr=0 kc=-1 shift.  NEXT-ch kr=0 ext = LR2 LIVE.
    ADD                 LR3 LR3 LR1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.ring_adv }} 9;
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;
    MULT.RC.VE          LR3 LR6 0 LR9 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 5: kc=0, no shift.
    INC                 LR3 1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.slot_stride }} 9;
    MULT.RC.VE          LR3 LR6 0 LR0 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 6: kc=+1 shift.
{%- else %}

    # --- tap 6: kc=+1 shift.  Rotate lr_read (moved from tap 8 to free its
    #     LR slot for the NEXT-ch kr=-1 ext calc below).
{%- endif %}
    INC                 LR3 1;
    INC                 LR6 1;
{%- if seeded %}
    ADD                 LR10 LR10 CR1;
{%- else %}
    INC.MODP2       LR4 {{ cr.half_slot }} 9;
{%- endif %}
    MULT.RC.VE          LR3 LR6 0 LR13 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 7: kr=+1 kc=-1 shift.  NEXT-ch kr=+1 ext = LR2+CR6.
    ADD                 LR3 LR3 LR1;
    INC                 LR6 1;
    ADD                 LR14 LR2 {{ cr.in_channels }};
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;
    MULT.RC.VE          LR3 LR6 0 LR9 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 8: kc=0, no shift.  rotate lr_read.
{%- else %}

    # --- tap 8: kc=0, no shift.  Precompute LR14 = NEXT-ch kr=-1 ext
    #     (= LR2-CR6; LR2 already advanced to NEXT-ch kr=0 ext at tap 3),
    #     overwriting tap 7's now-dead LR14 value, so tap 9's fused load can
    #     issue it directly.
{%- endif %}
    INC                 LR3 1;
    INC                 LR6 1;
{%- if seeded %}
    INC.MODP2       LR4 {{ cr.half_slot }} 9;
{%- else %}
    SUB                 LR14 LR2 {{ cr.in_channels }};
{%- endif %}
    MULT.RC.VE          LR3 LR6 0 LR0 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 9: kc=+1 shift.  Final acc, fused with ACTIVATE.EXACT in the
    #     SAME cycle (reads R_ACC LIVE; ACC dispatches before AAQ_EXACT) — no
    #     standalone cycle needed.  Prep LR5 for next iter tap 1.  Loop branch
    #     lives here too.
{%- else %}

    # --- tap 9: kc=+1 shift.  Final acc, fused with ACTIVATE.EXACT in the
    #     SAME cycle (ACTIVATE now reads R_ACC LIVE, and ACC dispatches
    #     before AAQ in the per-cycle order, so it sees this word's acc.add
    #     result) — no standalone cycle needed.  LR5 += 256 (mod 512) ->
    #     NEXT-ch kr=-1 slot; load NEXT-ch kr=-1 (LR14, computed at tap 8)
    #     there in the same word (XMEM reads LR5 LIVE, post-LR-phase), so
    #     it's ready for NEXT ch's tap 1.  Loop branch lives here too.
{%- endif %}
    INC                 LR3 1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.half_slot }} 9;
{%- if not seeded %}
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;
{%- endif %}
    MULT.RC.VE          LR3 LR6 0 LR13 CR15.L CR15.H ;
    ACC.ADD;
    ACTIVATE.EXACT {{ activation }} CR15.L {{ cr.y }} LR7;
    BLT.S                 LR10 LR11 mn_tap_body;;

    # SUPER_BLOCK_ROWS=2: advance LR12 (kernel super-block XMEM row) by 2
    # rows via CR1 (row-space); CR13 is R_CYCLIC ELEMENT-space (256) and
    # would double-count under row addressing if reused here. ADD's src_a
    # reads snapshot, so the second +CR1 needs its own cycle (this is a
    # group-boundary event, not the hot per-tap loop -- one extra cycle here
    # is not on the per-channel critical path).
    ADD                 LR12 LR12 CR1;;
    ADD                 LR12 LR12 CR1;
    BLT.S                 LR10 {{ cr.in_channels }} mn_reload;;

    ADD                 LR8 LR8 {{ cr.in_channels }};;
    BLT.S                 LR8 {{ cr.chunk_limit }} row_loop;;

    B                   gN_section;;

mn_reload:
    LD.MULT        R0 LR12 {{ cr.filters }};
    SET                 LR6 CR0;;

    ADD                 LR14 LR12 CR1;
    LD.MULT        R1 LR14 {{ cr.filters }};;

    ADD                 LR11 LR10 {{ cr.fpb }};;
    BLT.S                 {{ cr.in_channels }} LR11 mn_reload_clamp;;
    B                   ch_pre;;
mn_reload_clamp:
    ADD                 LR11 LR0 {{ cr.in_channels }};
    B                   ch_pre;;

# ===========================================================================
# Section 3: Last chunk (bottom border) — kr=+1 row masked (slots 3/4/5).
# ===========================================================================

gN_section:
    # No R_MASK reload: the single blob already carries slot 6 (bottom-row zero),
    # selected by the kr=+1 taps below.
    SET                 LR12 CR0;;

gN_kg_loop:
    LD.MULT        R0 LR12 {{ cr.filters }};
    SET                 LR6 CR0;;

    ADD                 LR14 LR12 CR1;
    SET                 LR10 CR0;
    LD.MULT        R1 LR14 {{ cr.filters }};;

    ADD                 LR11 LR10 {{ cr.fpb }};;
    BLT.S                 {{ cr.in_channels }} LR11 gN_clamp;;
    B                   gN_ch_pre;;
gN_clamp:
    ADD                 LR11 LR0 {{ cr.in_channels }};;

gN_ch_pre:
    # Cy 1: LR2 = LR8+LR10; load ch kr=0 → slot 0.
    SET                 LR4 CR0;
    SET                 LR5 CR0;
    ADD                 LR2 LR8 LR10;
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;;

    # Cy 2: LR5 = 128; LR3 = 1 (seed for -255); load THIS-ch kr=+1 with its own
    # valid base (LR2); bottom row's lanes are masked on the kr=+1 taps.
    ADD                 LR5 LR5 {{ cr.slot_stride }};
    SET                 LR3 CR1;
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;;

    # Cy 3: LR3 = 1-256 = -255; LR5 = 128+256 = 384.
    SUB                 LR3 LR3 {{ cr.half_slot }};
    ADD                 LR5 LR5 {{ cr.half_slot }};;
{%- if seeded %}

    # Cy 4: walk seed += cols; LR6 = -1.
{%- else %}

    # Cy 4: walk seed += cols; LR6 = -1.  Prefetch THIS ch's kr=-1 (ext =
    # LR2-CR6) into its slot LR5 here, one cycle before tap 1 reads it.
{%- endif %}
    ADD                 LR3 LR3 {{ cr.cols }};
{%- if seeded %}
    SUB                 LR6 LR0 CR1;;
{%- else %}
    SUB                 LR6 LR0 CR1;
    SUB                 LR14 LR2 {{ cr.in_channels }};
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;;
{%- endif %}

    B                   gN_tap_body;;

gN_tap_body:
{%- if seeded %}
    # 10 cyc/ch.  NEXT-ch kr=-1/kr=0 from CR10; kr=+1 taps (7/8/9) masked
    # (bottom).  ACTIVATE.EXACT fused into tap 9's own cycle (see below).

    # --- cyc 0: BIAS seed + activate the PREVIOUS channel (LR3 not walked here).
    #     Also pre-load THIS-ch kr=-1 (ext = LR2-CR6) into its slot LR5 here, one
    #     cycle before tap 1 reads it (snapshot; cyc-0's LOAD slot is free).
    INC                 LR6 1;
    SUB                 LR14 LR2 {{ cr.in_channels }};
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;
    MULT.EE             LR6 CR1 0 LR0 CR15.L CR15.H ;
    ACC.ADD.FIRST;;
{%- endif %}
{%- if seeded %}

    # --- tap 1: kr=-1 kc=-1.  slot 0 + kc=-1 shift (LR9).  kr=-1 pre-loaded in
    #     cyc 0.  Walk +CR14.
{%- else %}
    # 9 cyc/ch.  NEXT-ch kr=-1/kr=0 from CR10; kr=+1 taps (7/8/9) masked
    # (bottom).  ACTIVATE.EXACT fused into tap 9's own cycle (see below).

    # --- tap 1: kr=-1 kc=-1.  slot 0 + kc=-1 shift (LR9).  kr=-1 pre-loaded in
    #     ch_pre/prev-iter tap 9.  Walk +CR14.  acc.add.first resets R_ACC.
    #     Loop counter += CR12 moved here (tap 1 has a free LR slot) to make
    #     room at tap 7 for the NEXT-ch kr=-1 ext calc.
{%- endif %}
    ADD                 LR3 LR3 {{ cr.walk_step }};
    INC                 LR6 1;
{%- if not seeded %}
    ADD                 LR10 LR10 CR1;
{%- endif %}
    MULT.RC.VE          LR3 LR6 0 LR9 CR15.L CR15.H ;
{%- if seeded %}
    ACC.ADD;;
{%- else %}
    ACC.ADD.FIRST;;
{%- endif %}

    # --- tap 2: kc=0, no shift.  LR7 advance: this channel's output row.
    INC                 LR3 1;
    INC                 LR6 1;
    ADD                 LR7 LR7 CR1;
    MULT.RC.VE          LR3 LR6 0 LR0 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 3: kc=+1 shift (LR13).  Advance LR2 → NEXT-ch kr=0 ext.
    INC                 LR3 1;
    INC                 LR6 1;
    ADD                 LR2 LR2 CR1;
    MULT.RC.VE          LR3 LR6 0 LR13 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 4: kr=0 kc=-1 shift.  NEXT-ch kr=0 from CR10.
    ADD                 LR3 LR3 LR1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.ring_adv }} 9;
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;
    MULT.RC.VE          LR3 LR6 0 LR9 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 5: kc=0, no shift.
    INC                 LR3 1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.slot_stride }} 9;
    MULT.RC.VE          LR3 LR6 0 LR0 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 6: kc=+1 shift.  Also pre-load the kr=+1 slot (consumed by tap 7)
    #     here, one cycle ahead (snapshot; tap 6's LOAD slot is free).  LR5 is
    #     unchanged between here and tap 7, so it targets the same slot.  Bottom
    #     border: load this ch's own valid base (LR2); bottom row's lanes masked.
{%- else %}

    # --- tap 6: kc=+1 shift.  Also pre-load the kr=+1 slot (consumed by tap 7)
    #     here, one cycle ahead (snapshot; this cycle's LOAD slot is free).
    #     LR5 is unchanged between here and tap 7, so it targets the same
    #     slot.  Bottom border: load this ch's own valid base (LR2); bottom
    #     row's lanes masked.
{%- endif %}
    INC                 LR3 1;
    INC                 LR6 1;
{%- if seeded %}
    ADD                 LR10 LR10 CR1;
{%- endif %}
    LD.MULT.CYCLIC LR2 {{ cr.x }} LR5;
    MULT.RC.VE          LR3 LR6 0 LR13 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 7: kr=+1 kc=-1.  Bottom row out of bounds: slot 6; kc=-1 shift (LR9)
    #     zeros the left edge column.  kr=+1 chunk pre-loaded in tap 6's word.
{%- else %}

    # --- tap 7: kr=+1 kc=-1.  Bottom row out of bounds: slot 6; kc=-1 shift (LR9)
    #     zeros the left edge column.  kr=+1 chunk pre-loaded in tap 6's word.
    #     Precompute LR14 = NEXT-ch kr=-1 ext (= LR2-CR6; LR2 already advanced
    #     to NEXT-ch kr=0 ext at tap 3) here (this tap has a free LR slot,
    #     since kr=+1 was pre-loaded at tap 6), so tap 9's fused load can
    #     issue it directly.
{%- endif %}
    ADD                 LR3 LR3 LR1;
    INC                 LR6 1;
{%- if not seeded %}
    SUB                 LR14 LR2 {{ cr.in_channels }};
{%- endif %}
    MULT.RC.VE          LR3 LR6 6 LR9 CR15.L CR15.H ;
    ACC.ADD;;

    # --- tap 8: kr=+1 kc=0.  slot 6 = bottom row only, no shift.  Rotate lr_read.
    INC                 LR3 1;
    INC                 LR6 1;
    INC.MODP2       LR4 {{ cr.half_slot }} 9;
    MULT.RC.VE          LR3 LR6 6 LR0 CR15.L CR15.H ;
    ACC.ADD;;
{%- if seeded %}

    # --- tap 9: kr=+1 kc=+1.  slot 6 + kc=+1 shift (LR13).  Final acc, fused
    #     with ACTIVATE.EXACT in the SAME cycle (reads R_ACC LIVE; ACC
    #     dispatches before AAQ_EXACT) — no standalone cycle needed.  Prep LR5 for
    #     next iter tap 1.  Loop branch lives here too.
{%- else %}

    # --- tap 9: kr=+1 kc=+1.  slot 6 + kc=+1 shift (LR13).  Final acc, fused
    #     with ACTIVATE.EXACT in the SAME cycle (reads R_ACC LIVE; ACC
    #     dispatches before AAQ_EXACT) — no standalone cycle needed.  LR5 += 256
    #     (mod 512) -> NEXT-ch kr=-1 slot; load NEXT-ch kr=-1 (LR14, from tap
    #     7) there in the same word (XMEM reads LR5 LIVE).  Loop branch lives
    #     here too.
{%- endif %}
    INC                 LR3 1;
    INC                 LR6 1;
    INC.MODP2       LR5 {{ cr.half_slot }} 9;
{%- if not seeded %}
    LD.MULT.CYCLIC LR14 {{ cr.x }} LR5;
{%- endif %}
    MULT.RC.VE          LR3 LR6 6 LR13 CR15.L CR15.H ;
    ACC.ADD;
    ACTIVATE.EXACT {{ activation }} CR15.L {{ cr.y }} LR7;
    BLT.S                 LR10 LR11 gN_tap_body;;

    # SUPER_BLOCK_ROWS=2: advance LR12 (kernel super-block XMEM row) by 2
    # rows via CR1 (row-space); CR13 is R_CYCLIC ELEMENT-space (256) and
    # would double-count under row addressing if reused here. ADD's src_a
    # reads snapshot, so the second +CR1 needs its own cycle (this is a
    # group-boundary event, not the hot per-tap loop -- one extra cycle here
    # is not on the per-channel critical path).
    ADD                 LR12 LR12 CR1;;
    ADD                 LR12 LR12 CR1;
    BLT.S                 LR10 {{ cr.in_channels }} gN_reload;;

end:
    # Epilogue: the very last channel's ACTIVATE.EXACT has stored its row.

    HALT;;

gN_reload:
    LD.MULT        R0 LR12 {{ cr.filters }};
    SET                 LR6 CR0;;

    ADD                 LR14 LR12 CR1;
    LD.MULT        R1 LR14 {{ cr.filters }};;

    ADD                 LR11 LR10 {{ cr.fpb }};;
    BLT.S                 {{ cr.in_channels }} LR11 gN_reload_clamp;;
    B                   gN_ch_pre;;
gN_reload_clamp:
    ADD                 LR11 LR0 {{ cr.in_channels }};
    B                   gN_ch_pre;;
