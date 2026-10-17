{#- ==========================================================================
    fold_32x32x144.asm -- rebuilds a 32x32x144 tensor from its four stride-2 phases

    The exact inverse of unfold_32x32x144: takes the 4 channel-major streams
    that kernel produces (TL, TR, BL, BR -- the phase within each 2x2 block,
    NOT image quadrants), each split into 2 token groups per channel, and
    writes the 32x32x144 spatial tensor NHCW striped, exactly
    unfold_32x32x144's INPUT layout.
    fold(unfold(x)) == x bit for bit: the x1.0 pass-through multiply is
    exact, and the fold/unfold round-trip test checks it.

    Input x (unfold_32x32x144's output):
      Stream s, channel c, token group t (0 or 1): row c*2 + t of stream s.
      UNLIKE fold_16x16x192, every one of the 128 lanes of every stream row
      is VALID: each (channel, token group) pair fills a whole row, with no
      stale padding tail.
      Row layout: 128 tokens = one token group's half of the 16x16 decimated
      grid for that stream and channel, row-major (8 rows x 16 cols).

    Output y (NHCW striped, unfold_32x32x144's input layout):
      8 stripes x 144 channels; each row = 4 spatial rows x 32 cols = 128
      elements. Row (stripe, ch) at y + stripe x 144 + ch.

    STRIPE / TOKEN-GROUP MAPPING -- derived and verified, not assumed:
      stream s (phase r_ph = s//2, c_ph = s%2) holds x[:, r_ph::2, c_ph::2]
      flattened row-major over the 16x16 decimated grid; tg=0 holds
      decimated rows 0..7, tg=1 decimated rows 8..15. This was re-confirmed
      by running the REAL unfold_32x32x144 with a per-lane/per-channel
      marker tensor and inspecting its output (not just trusting its
      reference).

      Decimated row dr (0..15) <-> spatial row (r_ph + 2*dr) <-> stripe
      (spatial_row // 4, since a stripe is 4 rows). Enumerating every
      (r_ph, tg, dr_local) shows a CLEAN split with nothing straddling:
        tg=0 (decimated rows 0..7)  <-> stripes 0..3 exactly
        tg=1 (decimated rows 8..15) <-> stripes 4..7 exactly
      i.e. stripe s's token group is tg = s // 4, and within that token
      group's row, stripe s draws from a 32-lane window at
      src_base = (s % 4) * 32 -- exactly 2 of the token group's 8 decimated
      rows (16 lanes each), matching the stripe's 4 spatial rows (2 per
      stream, since each stream covers only even OR odd spatial rows).
      This is cleaner than fold_16x16x192 (where each destination stripe row
      mixed both halves of a stream row within one pass): here one
      (stripe, stream) pair maps to one contiguous 32-lane window of one
      token group's row. Each stripe's 128 destination lanes are covered
      exactly once by the 4 streams x 4 ACC.RESHAPE calls = 16 calls
      (brute-force partition check, before this kernel was written).

    ACC.RESHAPE MAPPING (unfold's ACC.STRIDE selection with
    elements_in_row=32, inverted; checked in Python before this kernel was
    written):
      For ONE destination stripe row (128 lanes), each stream's 32-lane
      source window scatters via 4 ACC.RESHAPE calls of 8 elements each:
        source (the same for every stream and stripe): [src_base+8c, ...,
          src_base+8c+7] for call c in 0..3, src_base = (stripe % 4) * 32.
        destination, stream s, call c: base(s) + table, where base(TL)=0,
          base(TR)=1, base(BL)=32, base(BR)=33, and the 4 calls' tables
          (relative to base) are IDENTICAL for every stream and stripe:
            call0 = base + [0,2,4,6,8,10,12,14]
            call1 = base + [16,18,20,22,24,26,28,30]   (call0 + 16)
            call2 = base + [64,66,68,70,72,74,76,78]   (call1 + 48)
            call3 = base + [80,82,84,86,88,90,92,94]   (call2 + 16)
      These 16 (stream, call) tables cover all 128 destination lanes exactly
      once for EVERY stripe (128 distinct indices, 0..127, no overlap, for
      all 8 stripes).

      Because the destination table and its step sequence (+16, +48, +16)
      are IDENTICAL across all 8 stripes, the stripe loop is unrolled with
      Jinja (as unfold_32x32x144 unrolls its 8 (stream, h, v, tg) groups)
      purely to bake each stripe's source window base and token-group row
      as literal ADDBI/CR values -- the CHANNEL loop (0..143) inside each
      stripe block is a real runtime loop.

    PER-STRIPE LR6 RESET -- a real bug caught during derivation, not an
    optimisation note: LR6 (the source row offset) must equal ch*2 + tg for
    the CURRENT stripe's channel. Every stripe's channel loop restarts ch at
    0, so LR6 must be reset to tg (0 or 1) at the START of every stripe
    block, INCLUDING transitions that keep the same tg (stripe 0->1, 1->2,
    2->3, 4->5, 5->6, 6->7) -- resetting only when tg flips (stripe 3->4) is
    not enough. A first draft let LR6 keep counting across stripe boundaries
    (as if reading a flat ch*8+stripe stream); that is wrong because every
    stream's row index is ch*2 + tg, independent of the stripe -- stripes
    0..3, then 4..7, reuse the exact same stream rows.

    INDEX TABLES. SET copies a CR into an LR, so each table is held in two
    CRs, four little-endian lane indices per CR, and copied into an LR pair
    (LRDn = LRn, LRn+1) when needed:
      SOURCE table LRD2 (LR2, LR3) is SET from CR2, CR3 ([0,1,2,3] /
      [4,5,6,7]) at the start of every stream's 4-call block, then walked +8
      per call with ADDBI. Where the window base is nonzero
      ((stripe % 4) != 0) the freshly-SET [0..7] table is walked once more,
      by ADDBI +(stripe % 4)*32, before call 0. ADDBI is a per-lane
      saturating byte add in [0, 255]; no index comes near saturation (the
      largest used is 127).
      DESTINATION table LRD4 (LR4, LR5) is SET from CR4, CR5 (TL's call-0
      table, [0,2,4,6] / [8,10,12,14]); TR/BL/BR then walk it to their base
      with one ADDBI (+1, +32, +33 -- TR = TL+1, BL = TL+32, BR = TL+33 on
      every lane at once). Across a stream's 4 calls it walks +16, +48, +16
      (NOT a constant step, unlike fold_16x16x192's uniform +32: L3's stripe
      geometry gives 2 spatial rows per stream per stripe instead of L4's
      contiguous half-row). It does NOT depend on the stripe; it is rebuilt
      once per stream per stripe for auditability, and a cycle-tuned version
      could keep it across all 8 stripes.

    PER-STRIPE DESTINATION ROW: ST.AAQ's base operand must be a CR, not an LR
    (a first draft tried an LR base and the assembler rejected it: that
    operand accepts only CR0..CR15). So the destination row is CR13 (y,
    fixed) + LR8, and LR8 is the running offset stripe*144 + ch across the
    WHOLE kernel (0 .. 8*144-1 = 1151) -- it is NEVER reset at a stripe
    boundary (unlike LR6 and LR10, which restart with every stripe's loop).
    That is simpler than 8 per-stripe destination-base CRs, which would not
    fit in the few free CRs left.

    ABI: kernel.toml's [abi] binds the CRs and LRs below.
      CR2, CR3     source table [0,1,2,3] / [4,5,6,7]
      CR4, CR5     TL's destination table [0,2,4,6] / [8,10,12,14]
      CR8          the ones row (R_CYCLIC's 1.0 multiplier)
      CR9..CR12    streams TL, TR, BL, BR of x
      CR13         y (stripe 0's base; every stripe's row is CR13 + LR8)
      LR7          source row stride per channel (2: tg0, tg1)
      LR9          destination row stride per channel (1)
      LR11         the channel loop's bound (144)
    LR0 (R_CYCLIC slot 0), LR6 (source row offset = ch*2 + tg), LR8
    (destination row offset) and LR10 (channel counter) start at their reset
    value, 0; LR6 and LR10 are SET again at every later stripe.
    CR0 (0) and CR1 (1) are hardwired: CR1 is the channel loop's increment,
    and both are deliberately reused as LR6's reset value (tg 0 or 1) and
    CR0 as LR10's.

    CR0 IS NOT A FREE BASE: a host write to CR0 or CR1 raises.
    unfold_32x32x144 reads its stripe-0 source base through CR0 only because
    that base is row 0. Fold's stripe-0 destination base is not 0 (y sits
    after x), so it cannot use CR0 -- the trap fold_16x16x192 also notes.
    CR13 is used instead.

    MULT CYCLE-START CONTRACT (issue #157, as in fold_16x16x192): MULT.RC.VV
    reads its Ra data (R0) as it was at the start of the cycle, so it cannot
    consume a row loaded by an LD.MULT in the same instruction word.
    Co-issuing a load and a MULT is fine only when the MULT consumes a row
    loaded earlier. This kernel primes the first load of EVERY stripe block
    (not just the kernel's first) before that stripe's channel loop, because
    LR6 -- and so the row loaded -- resets at every stripe boundary: the
    rotation cannot carry across a stripe boundary the way it carries across
    a channel boundary. Within one stripe's channel loop the rotation is
    fold_16x16x192's TL -> TR -> BL -> BR -> (next channel's TL), with only
    ONE pass per stripe, so BR's prefetch always targets the NEXT channel's
    TL.

    ST.AAQ reads its offset operand as the forwarded value, after the word's
    LR writes -- never co-issue an ADD on the offset LR (LR8) with the store,
    or the row lands one row late. Here LR8 advances only in its own
    instruction word at the very end of the channel loop, well clear of the
    stripe's single store.
========================================================================== -#}

{% set groups = [
     ("TL", 0, "CR9", "CR10"),
     ("TR", 1, "CR10", "CR11"),
     ("BL", 32, "CR11", "CR12"),
     ("BR", 33, "CR12", "CR9"),
   ] %}

    LD.MULT.CYCLIC LR0 CR8 LR0;;       # r_cyclic[0..127] = 1.0

{% for stripe in range(8) %}
{%- set tg = stripe // 4 %}
{%- set src_base = (stripe % 4) * 32 %}
{%- set tg_reset_cr = "CR1" if tg == 1 else "CR0" %}
# ===========================================================================
# Stripe {{ stripe }}  (token group tg={{ tg }}, source window base={{ src_base }})
# ===========================================================================
{% if stripe > 0 %}
    # Reset per-stripe state: ch counter, src row offset (lr6 <- tg -- see
    # the "PER-STRIPE LR6 RESET" header note). lr8 (dst row offset) is NOT
    # reset -- it runs continuously stripe*144+ch across the whole kernel
    # (see the "PER-STRIPE DESTINATION ROW" header note).
    SET                 LR10 CR0;;
    SET                 LR6 {{ tg_reset_cr }};;
{%- endif %}
    # (Re-)prime: stream TL row of this stripe's ch 0. Required at every
    # stripe boundary because lr6 just reset -- the cycle-start contract
    # rotation cannot carry a stale prefetch across the boundary.
    LD.MULT        R0 LR6 CR9;;

1:                                      # this stripe's channel loop (local label)

{% for g in groups %}
{%- set name = g[0] %}
{%- set dest_base = g[1] %}
{%- set next_cr = g[3] %}
    # -- Stream {{ name }}: dest table = TL base{% if dest_base != 0 %} + {{ dest_base }}{% endif %} --------------------------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;           # source table <- [0,1,2,3,4,5,6,7]
    SET                 LR4 CR4;
    SET                 LR5 CR5;;           # dest table   <- TL base
{%- if src_base != 0 %}
    ADDBI               LRD2 {{ src_base }};;       # source table <- [{{ src_base }}..{{ src_base+7 }}] (this stripe's window)
{%- endif %}
{%- if dest_base != 0 %}
    ADDBI               LRD4 {{ dest_base }};;      # dest table <- {{ name }} base
{%- endif %}
{%- if name == "BR" %}
    # lr6 (src offset) must advance BEFORE the load that uses next channel's
    # offset: LR sub-slots run ahead of LOAD within a word (same hazard
    # fold_16x16x192 documents for its lr6), so this ADD gets its own word.
    ADD                 LR6 LR6 LR7;;       # src offset: next channel (+2 rows: tg0,tg1)
{%- endif %}
    LD.MULT R0 LR6 {{ next_cr }};
    MULT.RC.VV LR0 R0 0 LR0 CR15;;          # multiply {{ name }} row (prefetch next stream)
    ACC.RESHAPE LRD2 LRD4 0;;               # call0: src[{{ src_base }}..{{ src_base+7 }}] -> dest[{{ dest_base }},{{ dest_base+2 }},..,{{ dest_base+14 }}]
    ADDBI               LRD2 8;
    ADDBI               LRD4 16;;
    ACC.RESHAPE LRD2 LRD4 0;;               # call1
    ADDBI               LRD2 8;
    ADDBI               LRD4 48;;
    ACC.RESHAPE LRD2 LRD4 0;;               # call2
    ADDBI               LRD2 8;
    ADDBI               LRD4 16;;
    ACC.RESHAPE LRD2 LRD4 0;;               # call3
{% endfor %}
    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ         LR8 CR13;;     # stripe {{ stripe }} <- DST_BASE_ROW + (stripe*144 + ch)

    # -- Advance pointers; loop within this stripe --------------------------
    ADD                 LR8 LR8 LR9;;       # dst offset: next row (running across the whole kernel)
    ADD                 LR10 LR10 CR1;;
    BLT.S                 LR10 LR11 1b;;                  # loop while ch < 144
{% endfor %}

end:
    HALT;;
