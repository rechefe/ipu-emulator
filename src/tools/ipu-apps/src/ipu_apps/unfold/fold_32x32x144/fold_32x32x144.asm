# Fold 32x32x144: 4 channel-major streams -> spatial 32x32x144   (L3, inverse of unfold)
#
# Layer:   L3
# Scope:   single-stream (per-stripe; consumes the 4 spatial streams)
# Layout:  unpacked
# Shape:   4 streams x [288 rows (144ch x 2tg), 128chan-lanes] -> 32x32x144 spatial output
# Status:  new (L3 sibling of fold_16x16x192; geometry re-derived, not ported)
# Related: exact algebraic inverse of unfold_32x32x144 -- same (H, W, C), same
#          NHCW-striped output layout unfold's OWN INPUT used. Verified
#          fold(unfold(x)) == x bit-for-bit (mod FP32 rounding from the x1.0
#          pass-through multiply, which is exact) in
#          test_fold_32x32x144_wide.py.
# Tests:   test_fold_32x32x144_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Rearranges 4 channel-major streams (TL, TR, BL, BR -- the 2x2 stride
# decimation unfold_32x32x144 produces) back into a 32x32x144 spatial tensor,
# NHCW striped (8 stripes x 144 channels x 128-element row), exactly matching
# unfold_32x32x144's INPUT format.
#
# Input (per-stream channel-major FP32, matches unfold_32x32x144's OUTPUT):
#   4 streams x 288 rows x 512 bytes, stream s based at SRC_s.
#   Stream s, ch c, token-group t (0 or 1): at SRC_s + (c*2 + t) row.
#   *** UNLIKE fold_16x16x192, every one of the 128 lanes of every loaded
#   *** stream row is VALID data here -- unfold_32x32x144's stream rows have
#   *** no stale padding tail (each (ch, tg) pair fills a whole 128-lane row).
#   Row layout: 128 tokens = one token-group's half of the 16x16 decimated
#   grid for that stream/channel, in row-major order (8 rows x 16 cols).
#
# Output (NHCW striped, identical layout to unfold_32x32x144's INPUT):
#   8 stripes x 144 channels; each row = 4 spatial_rows x 32 cols = 128
#   elements (512 bytes in wide-vector debug mode).
#   Row (stripe, ch) at DST_BASE + (stripe x 144 + ch) row.
#
# STRIPE / TOKEN-GROUP MAPPING -- derived and verified BEFORE writing this
# kernel (per the task brief, not assumed):
#   unfold_32x32x144's own test (test_unfold_32x32x144_wide.py) establishes
#   the ground truth: stream s (phase r_ph=s//2, c_ph=s%2) contains
#   x[:, r_ph::2, c_ph::2] flattened row-major over the 16x16 decimated grid;
#   tg=0 holds decimated rows 0..7, tg=1 holds decimated rows 8..15. This was
#   re-confirmed empirically here by running the REAL unfold_32x32x144
#   kernel with a per-lane/per-channel marker tensor and inspecting its
#   actual output (not just trusting the test file) -- see the derivation
#   script used to build this kernel's index tables.
#
#   Decimated row dr (0..15) <-> spatial row (r_ph + 2*dr) <-> stripe
#   (spatial_row // 4, since stripe_h=4). Enumerating all (r_ph, tg, dr_local)
#   combinations shows a CLEAN split with zero straddling:
#     tg=0 (decimated rows 0..7)  <-> stripes 0..3 exactly
#     tg=1 (decimated rows 8..15) <-> stripes 4..7 exactly
#   i.e. stripe s's token group is tg = s // 4, and within that tg-half of a
#   stream's 128-lane row, stripe s draws from a 32-lane window at
#   src_base = (s % 4) * 32 -- exactly 2 of that tg's 8 decimated rows (16
#   lanes each), matching this stripe's 4 physical rows (2 contributed per
#   stream, since each stream only covers even OR odd physical rows).
#   This is a strictly cleaner structure than fold_16x16x192's (there each
#   destination stripe row mixed both "halves" of a stream row within one
#   pass; here, one (stripe, stream) pair maps to exactly one contiguous
#   32-lane source window of exactly one tg's row -- no dest-lane straddling
#   between token groups at all). Verified by brute-force partition check in
#   Python (each stripe's 128 destination lanes are covered exactly once
#   across the 4 streams x 4 ACC.RESHAPE calls = 16 calls) before writing
#   this .asm.
#
# ACC.RESHAPE mapping (derived by inverting unfold's execute_acc_stride
# selection with elements_in_row=32; verified in Python before writing this
# kernel):
#   For ONE destination stripe row (128 lanes), each stream's 32-lane source
#   window (source base = (stripe % 4) * 32 within the stripe's token
#   group's row) scatters via 4 ACC.RESHAPE calls of 8 elements each:
#     source (same shape for every stream/stripe): [src_base+8c, ..., +8c+7]
#       for call c in 0..3, where src_base = (stripe % 4) * 32.
#     dest, stream s, call c: base(s) + step, where base(TL)=0, base(TR)=1,
#       base(BL)=32, base(BR)=33, and the 4 calls' dest tables (relative to
#       base) are IDENTICAL for every stream and every stripe:
#         call0 = base + [0,2,4,6,8,10,12,14]
#         call1 = base + [16,18,20,22,24,26,28,30]   (call0 + 16)
#         call2 = base + [64,66,68,70,72,74,76,78]   (call1 + 48)
#         call3 = base + [80,82,84,86,88,90,92,94]   (call2 + 16)
#   These 16 (stream, call) dest tables partition all 128 destination lanes
#   exactly once each for EVERY stripe -- confirmed by brute-force
#   enumeration in Python before writing this kernel (128 unique destination
#   indices, range 0..127, zero overlap, for all 8 stripes).
#
#   Because the dest table and its per-call step sequence (+16, +48, +16) are
#   IDENTICAL across all 8 stripes, the stripe loop is unrolled here via
#   Jinja (mirroring unfold_32x32x144's own use of Jinja for its 8
#   (stream,h,v,tg) groups) purely to bake each stripe's constant SOURCE
#   window base and token-group row selector as literal ADDBI/CR values --
#   the runtime CHANNEL loop (0..143) inside each stripe block is a real
#   loop, not unrolled.
#
# PER-STRIPE lr6 (src row offset) RESET -- a real bug caught during
# derivation, not just an optimisation note: lr6 must equal ch*2 + tg for
# the CURRENT stripe's channel. Since every stripe's channel loop restarts
# ch at 0, lr6 must be reset to tg (0 or 1) at the START of every stripe
# block, INCLUDING stripe-to-stripe transitions that keep the same tg
# (stripe0->1, 1->2, 2->3, 4->5, 5->6, 6->7) -- it is NOT sufficient to only
# reset lr6 when tg flips (stripe3->4). A first draft of this derivation
# assumed lr6 could just keep incrementing monotonically across stripe
# boundaries (as if reading a flat ch*8+stripe-style stream); that is wrong
# because all 4 streams' row indexing is ch*2+tg, independent of stripe --
# multiple stripes (0..3, then 4..7) reuse the exact same stream rows.
#
# Building the index tables (per user directive: never assume an LR holds
# anything but garbage at kernel start -- unlike CRs, which setup()
# legitimately initializes via state.regfile.set_cr):
#   SOURCE table (LRD2 = LR3:LR2) is rebuilt via two SETs from cr2/cr3
#   (packed bytes [0,1,2,3] / [4,5,6,7]) at the start of every stream's
#   4-call block, then walked forward +8 per call via ADDBI. For stripes
#   where the source window base is nonzero ((stripe % 4) != 0), the
#   freshly-reset [0..7] table is walked forward once more via ADDBI
#   +(stripe%4)*32 before call 0 -- confirmed ADDBI's per-lane saturating
#   byte add in [0, 255] never approaches saturation (max index used here is
#   127).
#
#   DEST table (LRD4 = LR5:LR4) is rebuilt via two SETs from cr4/cr5 (TL's
#   call-0 table, packed bytes [0,2,4,6] / [8,10,12,14]) then, for TR/BL/BR,
#   walked to that stream's base via a single ADDBI (+1, +32, +33
#   respectively -- TR = TL+1, BL = TL+32, BR = TL+33 on every lane
#   simultaneously). Across the 4 calls of one stream, the table is walked
#   +16, +48, +16 (NOT a constant step, unlike fold_16x16x192's uniform +32 --
#   see the derivation above for why: L3's stripe geometry produces 2
#   physical rows per stream per stripe instead of L4's contiguous half-row).
#   The dest table does NOT depend on which stripe is active -- it is
#   rebuilt once per stream per stripe purely for auditability
#   (correctness-first per the task brief; a cycle-tuned version could cache
#   it across all 8 stripes).
#
# Per-stripe DESTINATION row: STR_POST_AAQ_REG's base operand must be a CR,
# not an LR (found when a first draft tried an LR base and the assembler
# rejected it: "CrRegField" only accepts cr0..cr15). So the destination row
# is CR13 (DST_BASE_ROW, fixed) + LR8, and LR8 is the running offset
# stripe*C + ch across the WHOLE kernel (0 .. 8*144-1 = 1151) -- it is NEVER
# reset at a stripe boundary (unlike lr6/lr10, which reset because their
# corresponding CRs/loop structure restart every stripe). This is simpler
# than baking 8 separate per-stripe dest-base CRs, which would not fit in
# the handful of free CRs left (see the CR list below).
#
# CRs:
#   cr13 = DST_BASE_ROW           (stripe-0 output base, ch 0..143)
#   -- NOTE: CR0 is ALSO read-only hardwired to 0 (not just CR1 -- see
#      CR_READ_ONLY_INITIAL_VALUES in ipu_config.py); unfold_32x32x144 gets
#      away with naming its stripe-0 SOURCE base cr0 only because that base
#      happens to equal 0 anyway. Fold's stripe-0 DESTINATION base is
#      DST_BASE_ROW (nonzero, sitting after the 4-stream source region), so
#      it cannot use cr0 -- same trap fold_16x16x192 documents. cr13 is used
#      instead (arbitrary choice among the free CRs). cr0(=0)/cr1(=1) ARE
#      reused deliberately below as the tg-reset values for lr6 (0 and 1
#      respectively), which is exactly what they already hold.
#   cr8  = ONES_BASE               (128 elements of dtype 1.0, for r_cyclic init)
#   cr9  = SRC_TL                  (stream TL input base, row 0)
#   cr10 = SRC_TR                  (stream TR input base, row 288)
#   cr11 = SRC_BL                  (stream BL input base, row 576)
#   cr12 = SRC_BR                  (stream BR input base, row 864)
#   cr2  = source table lo  bytes [0,1,2,3]     (LRD2 low word,  via SET)
#   cr3  = source table hi  bytes [4,5,6,7]     (LRD2 high word, via SET)
#   cr4  = TL dest table lo bytes [0,2,4,6]     (LRD4 low word,  via SET)
#   cr5  = TL dest table hi bytes [8,10,12,14]  (LRD4 high word, via SET)
#   cr1  = 1    (read-only hardwired constant; channel-loop increment AND
#                lr6 reset value for tg=1 stripes)
#   cr0  = 0    (read-only hardwired constant; lr6/lr10 reset value)
#
# LRs (preset by harness):
#   lr0  = 0    (const: r_cyclic slot 0)
#   lrd2 = LR2:LR3  (working source-lane-index table for ACC.RESHAPE)
#   lrd4 = LR4:LR5  (working dest-lane-index table for ACC.RESHAPE)
#   lr6  = 0    (src row offset within a stream = ch*2+tg; reset every stripe)
#   lr7  = 2    (src stride per channel, in rows -- 2 rows: tg0, tg1)
#   lr8  = 0    (dst row offset = stripe*144 + ch, relative to DST_BASE_ROW (cr13); += 1 per channel, NEVER reset -- runs 0..1151 across the whole kernel)
#   lr9  = 1    (dst stride per channel, in rows)
#   lr10 = 0    (ch counter, 0..143, reset at the start of every stripe block)
#   lr11 = 144  (loop limit = C)
#
# Memory layout (ROW numbers -- .asm XMEM operands are rows, issue #179):
#   SRC:  4 streams x 288 rows                      (TL, TR, BL, BR)
#   ONES: 1 row
#   DST:  8 stripes x 144 rows
#
# MULT SNAPSHOT CONTRACT (issue #157, inherited from fold_16x16x192): a
# MULT.RC.VV instruction reads its Ra DATA (r0) from the start-of-cycle
# snapshot, so it cannot consume a chunk loaded by an LDR_MULT_REG issued in
# the very same bundle -- co-issuing a load and a MULT in one bundle is fine
# only when the MULT consumes a PREVIOUSLY loaded chunk. This kernel primes
# the first load of EVERY stripe block (not just the very first, kernel-wide
# prime) before that stripe's channel loop, because lr6 (and hence which row
# is loaded) resets at every stripe boundary -- the rotation cannot carry
# across a stripe boundary the way it carries across an ordinary channel
# boundary. Within one stripe's channel loop, the rotation is exactly
# fold_16x16x192's TL->TR->BL->BR->(next ch's TL) pattern, but with only ONE
# pass per stripe (not two), so BR's prefetch always targets the NEXT
# CHANNEL's TL (never "the same channel again", since L3 has no second
# stripe pass reusing the same 4 rows).
#
# STR_POST_AAQ_REG reads its offset operand LIVE -- never co-issue an ADD on
# the offset LR (lr8) in the same bundle as the store, or the row lands one
# slot late. Here lr8 only advances in its own bundle at the very end of the
# channel loop, well clear of the stripe's single store.
{% set groups = [
     ("TL", 0, "cr9", "cr10"),
     ("TR", 1, "cr10", "cr11"),
     ("BL", 32, "cr11", "cr12"),
     ("BR", 33, "cr12", "cr9"),
   ] %}

    LDR_CYCLIC_MULT_REG lr0 cr8 lr0;;       # r_cyclic[0..127] = 1.0 (dtype-specific)

{% for stripe in range(8) %}
{%- set tg = stripe // 4 %}
{%- set src_base = (stripe % 4) * 32 %}
{%- set tg_reset_cr = "cr1" if tg == 1 else "cr0" %}
# ===========================================================================
# Stripe {{ stripe }}  (token group tg={{ tg }}, source window base={{ src_base }})
# ===========================================================================
{% if stripe > 0 %}
    # Reset per-stripe state: ch counter, src row offset (lr6 <- tg -- see
    # the "PER-STRIPE lr6 RESET" header note). lr8 (dst row offset) is NOT
    # reset -- it runs continuously stripe*144+ch across the whole kernel
    # (see the "Per-stripe DESTINATION row" header note).
    SET                 lr10 cr0;;
    SET                 lr6 {{ tg_reset_cr }};;
{%- endif %}
    # (Re-)prime: stream TL row of this stripe's ch 0. Required at every
    # stripe boundary because lr6 just reset -- the snapshot-contract
    # rotation cannot carry a stale prefetch across the boundary.
    LDR_MULT_REG        r0 lr6 cr9;;

ch_loop_stripe{{ stripe }}:

{% for g in groups %}
{%- set name = g[0] %}
{%- set dest_base = g[1] %}
{%- set next_cr = g[3] %}
    # -- Stream {{ name }}: dest table = TL base{% if dest_base != 0 %} + {{ dest_base }}{% endif %} --------------------------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;           # source table <- [0,1,2,3,4,5,6,7]
    SET                 lr4 cr4;
    SET                 lr5 cr5;;           # dest table   <- TL base
{%- if src_base != 0 %}
    ADDBI               lrd2 {{ src_base }};;       # source table <- [{{ src_base }}..{{ src_base+7 }}] (this stripe's window)
{%- endif %}
{%- if dest_base != 0 %}
    ADDBI               lrd4 {{ dest_base }};;      # dest table <- {{ name }} base
{%- endif %}
{%- if name == "BR" %}
    # lr6 (src offset) must advance BEFORE the load that uses next channel's
    # offset: LR sub-slots run ahead of LOAD within a word (same hazard
    # fold_16x16x192 documents for its lr6), so this ADD gets its own word.
    ADD                 lr6 lr6 lr7;;       # src offset: next channel (+2 rows: tg0,tg1)
{%- endif %}
    LDR_MULT_REG r0 lr6 {{ next_cr }};
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply {{ name }} row (prefetch next stream)
    ACC.RESHAPE lrd2 lrd4 0;;               # call0: src[{{ src_base }}..{{ src_base+7 }}] -> dest[{{ dest_base }},{{ dest_base+2 }},..,{{ dest_base+14 }}]
    ADDBI               lrd2 8;
    ADDBI               lrd4 16;;
    ACC.RESHAPE lrd2 lrd4 0;;               # call1
    ADDBI               lrd2 8;
    ADDBI               lrd4 48;;
    ACC.RESHAPE lrd2 lrd4 0;;               # call2
    ADDBI               lrd2 8;
    ADDBI               lrd4 16;;
    ACC.RESHAPE lrd2 lrd4 0;;               # call3
{% endfor %}
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG         lr8 cr13;;     # stripe {{ stripe }} <- DST_BASE_ROW + (stripe*144 + ch)

    # -- Advance pointers; loop within this stripe --------------------------
    ADD                 lr8 lr8 lr9;;       # dst offset: next row (running across the whole kernel)
    ADD                 lr10 lr10 cr1;;
    BLT                 lr10 lr11 ch_loop_stripe{{ stripe }};;   # loop while ch < 144
{% endfor %}

end:
    BKPT;;
