{#- ==========================================================================
    unfold_16x16x192.asm -- splits a 16x16x192 tensor into its four stride-2 phases

    Rearranges a 16x16x192 spatial tensor (NHCW striped input) into 4
    stride-2 decimated streams of 8x8 tokens, channel-major. TL, TR, BL and
    BR name the PHASE within each 2x2 block (stream s takes phase
    (s // 2, s % 2)), NOT a quadrant of the image.
    Related: L4 variant of unfold_32x32x144 (L3) with re-derived geometry,
    NOT a direct port (2 stripes not 8, elements_in_row=16 not 32);
    unfold_8x8x240 is the L5 sibling; fold_16x16x192 is its inverse; feeds
    layernorm_64x192.

    The geometry differs from L3: L3 holds 4 spatial rows of 32 cols per row
    and has 8 stripes; L4 holds 8 rows of 16 cols and has only 2 stripes.
    The ACC.STRIDE decimation still yields 32 elements per call (4 rows x 8
    cols), so the four-slot structure carries over, but a stream fills only
    2 of R_ACC's 4 slots (L3 fills all 4, twice per channel).

    Input x (NHCW striped):
      2 stripes x 192 channels; each row = 8 spatial rows x 16 cols = 128
      elements. Row (stripe, ch) at x + stripe x 192 + ch.

    Output y (one channel per row, stream after stream, HALF-ROW PADDED):
      Stream s, channel c: row c of stream s.
      Each row carries 64 valid FP32 tokens in lanes 0..63; lanes 64..127
      are STALE R_ACC lanes and must be ignored by consumers. A stream
      contributes 2 x 32 = 64 tokens, which half-fills R_ACC;
      ACTIVATE.QUANTIZE identity stages all 128 lanes of R_ACC (CR15 carries
      the default valid_elements=128) and ST.AAQ writes the whole row. This
      is the deliberate per-stream layout; a packed [k][p*n] variant that
      would use the full row is not implemented.
      Lanes 0..31 hold the channel's stripe-0 tokens and lanes 32..63 its
      stripe-1 tokens -- the OPPOSITE of the textual load order in each
      instruction word (see MULT CYCLE-START CONTRACT); fold_16x16x192
      depends on this.

    Pass-through multiply: MULT.RC.VV with R_CYCLIC holding 1.0 in every lane
    (loaded once from the ones row at startup, never overwritten) and the
    stripe row in R0 (LD.MULT R0):
      MULT.RC.VV LR0 R0 0 LR0: R0[i] x R_CYCLIC[i] = stripe[i] x 1.0 = stripe[i].

    Streams (ACC.STRIDE with elements_in_row=16):
      128 elements from one stripe = 8 rows x 16 cols
      TL (stream 0): ACC.STRIDE 16 on     on     -> even cols, even rows
      TR (stream 1): ACC.STRIDE 16 on_inv on     -> odd  cols, even rows
      BL (stream 2): ACC.STRIDE 16 on     on_inv -> even cols, odd  rows
      BR (stream 3): ACC.STRIDE 16 on_inv on_inv -> odd  cols, odd  rows
      Each selects 32 elements (4 rows x 8 cols) with the expected even/odd
      row and column split, checked against the emulator.
    ACC.STRIDE encodings:
      elements_in_row: "16" (enum 0)
      horizontal: on=1 (even cols), on_inv=2 (odd cols)
      vertical:   on=1 (even rows), on_inv=2 (odd rows)

    ABI: kernel.toml's [abi] binds the CRs and LRs below.
      CR0          x's stripe 0, read through the hardwired 0: x must sit at
                   row 0
      CR13         x's stripe 1 (CR1 is read-only, so not CR1)
      CR8          the ones row (R_CYCLIC's 1.0 multiplier)
      CR9..CR12    streams TL, TR, BL, BR of y
      LR1          ACC.STRIDE R_ACC slot 1 (lanes 32..63)
      LR5, LR6     source and destination row stride per channel (1)
      LR11         the channel loop's bound (192)
    LR0 (R_CYCLIC slot 0, mask offset and shift 0, ACC.STRIDE slot 0), LR4
    (source row offset within each stripe), LR8 (destination row offset)
    and LR10 (channel counter) start at 0, as kernel.toml's [abi]
    binds them (slot0, src_row, dst_row, channel). CR1 (hardwired
    1) is the channel loop's increment. CR0 and CR1 are read-only: a host
    write to either raises.

    ST.AAQ reads its offset as the forwarded value, after the word's LR
    writes -- never co-issue an ADD on the offset LR with the store, or the
    row lands one row late. Here the destination offset LR8 advances in its
    own instruction word at the end of the channel loop, well clear of every
    store.

    MULT CYCLE-START CONTRACT (issue #157): MULT.RC.VV reads its R0 data as it
    was at the start of the cycle, so a MULT cannot consume a row loaded by an
    LD.MULT in its own instruction word -- it would see the PREVIOUS
    contents of R0. `;;` ends one instruction word = one cycle, so a load and
    a MULT in the same word always execute in the same cycle regardless of
    textual order; co-issuing them is fine, consuming the same-cycle load is
    not. Therefore the loads here run one instruction word AHEAD of the MULT
    that consumes them: the stripe-0 load for the first stream is primed
    before ch_loop, and each word's load prefetches the row the NEXT word
    will multiply. The eight loads per channel form a fixed stripe-0/stripe-1
    alternation, and LR4 (the source offset, read as the forwarded value by
    the load) advances only once per channel, so the rotation is a pure
    one-word shift: the last load of channel ch fetches stripe 0 of channel
    ch+1, which is why LR4's ADD must precede it.
========================================================================== -#}

    LD.MULT.CYCLIC LR0 CR8 LR0;;       # r_cyclic[0..127] = 1.0

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..191)
# ---------------------------------------------------------------------------

    LD.MULT        R0 LR4 CR0;;        # prime: stripe 0 of ch 0 (TL slot 0)

ch_loop:

    # Each word multiplies the row loaded by the PREVIOUS word and loads
    # the row the NEXT word will multiply (MULT cycle-start contract, see header).

    # -- Stream TL  (h=on=even cols, v=on=even rows) -------------------------
    LD.MULT R0 LR4 CR13;
    MULT.RC.VV LR0 R0 0 LR0 CR15;
    ACC.STRIDE 16 on on LR0;;
    LD.MULT R0 LR4 CR0;
    MULT.RC.VV LR0 R0 0 LR0 CR15;
    ACC.STRIDE 16 on on LR1;;
    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ         LR8 CR9;;  # TL → DST_TL + ch (lanes 0..63 valid)

    # -- Stream TR  (h=on_inv=odd cols, v=on=even rows) ----------------------
    LD.MULT R0 LR4 CR13;
    MULT.RC.VV LR0 R0 0 LR0 CR15;
    ACC.STRIDE 16 on_inv on LR0;;
    LD.MULT R0 LR4 CR0;
    MULT.RC.VV LR0 R0 0 LR0 CR15;
    ACC.STRIDE 16 on_inv on LR1;;
    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ         LR8 CR10;;  # TR

    # -- Stream BL  (h=on=even cols, v=on_inv=odd rows) ----------------------
    LD.MULT R0 LR4 CR13;
    MULT.RC.VV LR0 R0 0 LR0 CR15;
    ACC.STRIDE 16 on on_inv LR0;;
    LD.MULT R0 LR4 CR0;
    MULT.RC.VV LR0 R0 0 LR0 CR15;
    ACC.STRIDE 16 on on_inv LR1;;
    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ         LR8 CR11;;  # BL

    # -- Stream BR  (h=on_inv=odd cols, v=on_inv=odd rows) -------------------
    LD.MULT R0 LR4 CR13;
    MULT.RC.VV LR0 R0 0 LR0 CR15;
    ACC.STRIDE 16 on_inv on_inv LR0;;
    # lr4 must advance BEFORE the load that uses the next channel's offset: LR
    # sub-slots run ahead of LOAD within a word, so this ADD gets its own word.
    ADD                 LR4 LR4 LR5;;       # src offset: next channel (+1 row)
    LD.MULT R0 LR4 CR0;
    MULT.RC.VV LR0 R0 0 LR0 CR15;
    ACC.STRIDE 16 on_inv on_inv LR1;;
    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ         LR8 CR12;;  # BR

    # -- Advance pointers; loop ----------------------------------------------
    ADD                 LR8 LR8 LR6;;       # dst offset: next channel (+1 row)
    ADD                 LR10 LR10 CR1;;
    BLT.S                 LR10 LR11 ch_loop;; # loop while ch < 192

end:
    HALT;;
