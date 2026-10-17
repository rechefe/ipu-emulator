{#- ==========================================================================
    fold_16x16x192.asm -- rebuilds a 16x16x192 tensor from its four stride-2 phases

    The exact inverse of unfold_16x16x192: takes the 4 channel-major streams
    that kernel produces (TL, TR, BL, BR -- the phase within each 2x2 block,
    NOT image quadrants) and writes the 16x16x192 spatial tensor NHCW
    striped, exactly unfold_16x16x192's INPUT layout.
    fold(unfold(x)) == x bit for bit: the x1.0 pass-through multiply is
    exact, and the fold/unfold round-trip test checks it.

    Input x (unfold_16x16x192's output; one row per channel per stream):
      Stream s, channel c: row c of stream s.
      Only lanes 0..63 of a row are valid; lanes 64..127 are unfold's stale
      R_ACC padding and MUST be ignored -- no ACC.RESHAPE call reads past
      lane 63 of a loaded stream row.
      Row layout: lanes 0..31 = that channel's STRIPE 0 tokens, lanes 32..63
      = its STRIPE 1 tokens (see CRITICAL LAYOUT DETAIL for why this is the
      OPPOSITE of unfold's textual load order), each in the decimated order
      unfold's ACC.STRIDE produced (row-major within the 8x8 phase grid's
      half: 4 sub-rows x 8 sub-cols).

    Output y (NHCW striped, unfold_16x16x192's input layout):
      2 stripes x 192 channels; each row = 8 spatial rows x 16 cols = 128
      elements. Row (stripe, ch) at y + stripe x 192 + ch.

    CRITICAL LAYOUT DETAIL -- textual load order is NOT data order, because
    of the MULT cycle-start contract (see below): unfold_16x16x192's ch_loop
    TEXTUALLY loads STRIPE 1 (CR13) first and STRIPE 0 (CR0) second for
    every stream, but MULT.RC.VV reads its Ra data as it was at the START of
    the cycle, so each word's MULT consumes the PREVIOUS word's load, not
    the one issued in its own word (unfold primes this with a stripe-0 load
    before ch_loop). Net effect, confirmed by running a spatial tensor with
    distinct per-lane/per-stripe markers through the REAL unfold kernel and
    inspecting its output: ACC.STRIDE slot 0 (offset LR0=0, R_ACC lanes
    0..31) receives STRIPE 0's data (the primed load), and slot 1 (offset
    LR1=1, lanes 32..63) receives STRIPE 1's data -- the OPPOSITE of what the
    textual load order suggests. A first version of this kernel assumed the
    textual-order mapping and got a scrambled result that was NOT obviously
    wrong (every value present, just permuted) until checked against the
    round trip -- the same class of bug unfold_8x8x240's paired-row input
    order warns about. Trust the emulator, not the textual load order, when
    a kernel's loads run one instruction word ahead of the MULT that
    consumes them.

    Consequently: source window [0..31] of a loaded stream row feeds the
    STRIPE 0 destination row, and source window [32..63] feeds the STRIPE 1
    destination row.

    ACC.RESHAPE MAPPING (unfold's ACC.STRIDE selection with
    elements_in_row=16, inverted; checked in Python before this kernel was
    written and against the emulator by the round trip):
      For ONE destination stripe row (128 lanes), each stream's 32 valid
      source lanes (of the right half of its loaded row) scatter via 4
      ACC.RESHAPE calls of 8 elements each:
        source (the same for every stream): [8c, 8c+1, ..., 8c+7]
          -- stripe-0 pass (first): window offset +0;
             stripe-1 pass (second): +32.
        destination, stream s, call c: base(s) + 32*c + 2*i for i in 0..7,
          where base(TL)=0, base(TR)=1, base(BL)=16, base(BR)=17.
      These 16 (stream, call) destination tables cover all 128 destination
      lanes exactly once each -- confirmed by brute-force enumeration (128
      distinct indices, 0..127, no overlap).

    INDEX TABLES. SET copies a CR into an LR, so each table is held in two
    CRs, four little-endian lane indices per CR, and copied into an LR pair
    (LRDn = LRn, LRn+1) when needed:
      SOURCE table LRD2 (LR2, LR3) is SET from CR2, CR3 ([0,1,2,3] /
      [4,5,6,7]) at the start of every stream's 4-call block, then walked +8
      per call with ADDBI. For the stripe-1 pass (window base 32 instead of
      0), the freshly-SET [0..7] table is walked once more with ADDBI +32
      before call 0, reaching [32..39] in one step (ADDBI is a per-lane
      saturating byte add in [0, 255]; 7 + 32 = 39, far from saturating).
      DESTINATION table LRD4 (LR4, LR5) is SET from CR4, CR5 (TL's call-0
      table, [0,2,4,6] / [8,10,12,14]); TR/BL/BR then walk it to their base
      with one ADDBI (+1, +16, +17 -- TR = TL+1, BL = TL+16, BR = TL+17 on
      every lane at once). Across a stream's 4 calls it walks +32 per call
      (the same step as the source table's window, coincidentally, but a
      different register). It does NOT depend on the stripe pass; it is
      rebuilt once per stream per pass for auditability, and a cycle-tuned
      version could keep it across the two passes.

    Questions from the design draft, resolved against the emulator:
      1. ACTIVATE.QUANTIZE identity after ACC.RESHAPE (not ACC.STRIDE) is
         fine: it reads the forwarded R_ACC with no ACC.STRIDE-specific
         assumption, over the active-element count CR15's dstructure gives
         (128, the default).
      2. R_ACC needs no reset before the first ACC.RESHAPE of a stripe pass:
         the 16 destination tables cover all 128 lanes exactly once, so
         every lane is written every pass whatever R_ACC held. ACC.RESHAPE
         updates only R_ACC[dest[i]] for participating i (unlike
         ACC.STRIDE's whole-slot overwrite), which is exactly why the
         partition property matters. The round-trip test also runs with
         R_ACC poisoned by a non-zero sentinel before fold starts.
      3. ADDBI's per-lane saturating byte arithmetic gives the intended
         tables at every step; no index comes near 255 (the largest used is
         127).

    ABI: kernel.toml's [abi] binds the CRs and LRs below.
      CR2, CR3     source table [0,1,2,3] / [4,5,6,7]
      CR4, CR5     TL's destination table [0,2,4,6] / [8,10,12,14]
      CR8          the ones row (R_CYCLIC's 1.0 multiplier)
      CR9..CR12    streams TL, TR, BL, BR of x
      CR14         y's stripe 0
      CR13         y's stripe 1
      LR7, LR9     source and destination row stride per channel (1)
      LR11         the channel loop's bound (192)
    LR0 (R_CYCLIC slot 0), LR6 (source row offset within a stream), LR8
    (destination row offset) and LR10 (channel counter) start at their reset
    value, 0. CR1 (hardwired 1) is the channel loop's increment.

    CR0 IS NOT A FREE BASE: CR0 (0) and CR1 (1) are hardwired, and a host
    write to either raises. unfold_16x16x192 reads its stripe-0 source base
    through CR0 only because that base is row 0. Fold's stripe-0 destination
    base is not 0, so it cannot use CR0 -- a real bug the direct-reference
    test caught (stripe 0's rows silently went to XMEM row 0). CR14 is used
    instead.

    MULT CYCLE-START CONTRACT (issue #157, as in unfold_16x16x192): MULT.RC.VV
    reads its Ra data (R0) as it was at the start of the cycle, so it cannot
    consume a row loaded by an LD.MULT in the same instruction word.
    Co-issuing a load and a MULT is fine only when the MULT consumes a row
    loaded earlier. This kernel primes the first load before ch_loop and
    keeps every later load one instruction word ahead of the MULT that
    consumes it, exactly like unfold_16x16x192.

    ST.AAQ reads its offset operand as the forwarded value, after the word's
    LR writes -- never co-issue an ADD on the offset LR (LR8) with the store,
    or the row lands one row late. Here LR8 advances only in its own
    instruction word at the very end of the channel loop, well clear of both
    stripe stores.
========================================================================== -#}

    LD.MULT.CYCLIC LR0 CR8 LR0;;       # r_cyclic[0..127] = 1.0

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..191)
# ---------------------------------------------------------------------------

    LD.MULT        R0 LR6 CR9;;        # prime: stream TL row of ch 0

ch_loop:

    # ===== Destination STRIPE 0 pass (source window = lanes [0..31]) =======

    # -- Stream TL: dest table = TL base [0,2,4,6,8,10,12,14] ----------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;           # source table <- [0,1,2,3,4,5,6,7]
    SET                 LR4 CR4;
    SET                 LR5 CR5;;           # dest table   <- TL base
    LD.MULT R0 LR6 CR10;
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply TL row (prefetch TR next)
    ACC.RESHAPE LRD2 LRD4 0;;               # call0: src[0..7]   -> dest[0,2,..,14]
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;               # call1: src[8..15]  -> dest[32,34,..,46]
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;               # call2: src[16..23] -> dest[64,66,..,78]
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;               # call3: src[24..31] -> dest[96,98,..,110]

    # -- Stream TR: dest table = TL base + 1 ---------------------------------
    # SET and a same-pair ADDBI cannot share an instruction word -- both would target
    # LR4/LR5, and the emulator's LR-slot conflict check rejects any word
    # writing the same LR twice regardless of instruction identity -- so the
    # SET-reset and the ADDBI-offset each get their own word.
    SET                 LR2 CR2;
    SET                 LR3 CR3;;
    SET                 LR4 CR4;
    SET                 LR5 CR5;;           # dest table <- TL base
    ADDBI               LRD4 1;;            # dest table <- TR base
    LD.MULT R0 LR6 CR11;
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply TR row (prefetch BL next)
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;

    # -- Stream BL: dest table = TL base + 16 --------------------------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;
    SET                 LR4 CR4;
    SET                 LR5 CR5;;           # dest table <- TL base
    ADDBI               LRD4 16;;           # dest table <- BL base
    LD.MULT R0 LR6 CR12;
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply BL row (prefetch BR next)
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;

    # -- Stream BR: dest table = TL base + 17 --------------------------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;
    SET                 LR4 CR4;
    SET                 LR5 CR5;;           # dest table <- TL base
    ADDBI               LRD4 17;;           # dest table <- BR base
    # Prefetch the FIRST stripe-0-pass load (stream TL, same ch): lr6 does
    # NOT advance here -- both stripe passes read the SAME 4 stream rows for
    # this channel; only the source-table window (rebuilt explicitly per
    # stream below) differs between the two passes.
    LD.MULT R0 LR6 CR9;
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply BR row (prefetch TL next)
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;

    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ         LR8 CR14;;     # stripe 0 <- DST_BASE + ch (current channel)

    # ===== Destination STRIPE 1 pass (source window = lanes [32..63]) ======

    # -- Stream TL: dest table = TL base --------------------------------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;           # source table <- [0,1,2,3,4,5,6,7]
    SET                 LR4 CR4;
    SET                 LR5 CR5;;           # dest table <- TL base
    ADDBI               LRD2 32;;           # source table <- [32..39] (stripe-1 window)
    LD.MULT R0 LR6 CR10;
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply TL row (prefetch TR next)
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;

    # -- Stream TR: dest table = TL base + 1 ---------------------------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;
    SET                 LR4 CR4;
    SET                 LR5 CR5;
    ADDBI               LRD2 32;;           # source table <- [32..39]; dest table <- TL base
    ADDBI               LRD4 1;;            # dest table <- TR base
    LD.MULT R0 LR6 CR11;
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply TR row (prefetch BL next)
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;

    # -- Stream BL: dest table = TL base + 16 --------------------------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;
    SET                 LR4 CR4;
    SET                 LR5 CR5;
    ADDBI               LRD2 32;;           # source table <- [32..39]; dest table <- TL base
    ADDBI               LRD4 16;;           # dest table <- BL base
    LD.MULT R0 LR6 CR12;
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply BL row (prefetch BR next)
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;

    # -- Stream BR: dest table = TL base + 17 --------------------------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;
    SET                 LR4 CR4;
    SET                 LR5 CR5;
    ADDBI               LRD2 32;;           # source table <- [32..39]; dest table <- TL base
    ADDBI               LRD4 17;;           # dest table <- BR base
    # lr6 (src offset) must advance BEFORE the load that uses next channel's
    # offset: LR sub-slots run ahead of LOAD within a word (same hazard
    # unfold_16x16x192 documents for its lr4), so this ADD gets its own word.
    # lr8 (dst offset) is deliberately NOT advanced here -- the stripe-0
    # store below still targets the CURRENT channel's row; it advances only
    # once, in the loop tail after both stripe stores (mirrors unfold's lr8).
    ADD                 LR6 LR6 LR7;;       # src offset: next channel
    LD.MULT R0 LR6 CR9;
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply BR row (prefetch ch+1's TL)
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;

    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ         LR8 CR13;;     # stripe 1 <- DST_BASE + 192 + ch

    # -- Advance pointers; loop -------------------------------------------------
    ADD                 LR8 LR8 LR9;;       # dst offset: next channel (+1 row)
    ADD                 LR10 LR10 CR1;;
    BLT.S                 LR10 LR11 ch_loop;; # loop while ch < 192

end:
    HALT;;
