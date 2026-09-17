# Fold 16x16x192: 4 channel-major streams -> spatial 16x16x192   (L4, inverse of unfold)
#
# Layer:   L4
# Scope:   single-stream (per-stripe; consumes the 4 spatial streams)
# Layout:  unpacked
# Shape:   4 streams x [64tok, 192chan] -> 16x16x192 spatial output
# Status:  new (first ACC.RESHAPE-based kernel in this codebase)
# Related: exact algebraic inverse of unfold_16x16x192 -- same (H, W, C),
#          same NHCW-striped output layout unfold's OWN INPUT used. Verified
#          fold(unfold(x)) == x bit-for-bit (mod FP32 rounding from the x1.0
#          pass-through multiply, which is exact) in
#          test_fold_16x16x192_wide.py.
# Tests:   test_fold_16x16x192_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Rearranges 4 channel-major streams (TL, TR, BL, BR -- the 2x2 stride
# decimation unfold_16x16x192 produces) back into a 16x16x192 spatial tensor,
# NHCW striped (2 stripes x 192 channels x 128B row), exactly matching
# unfold_16x16x192's INPUT format.
#
# Input (per-stream channel-major FP32, matches unfold_16x16x192's OUTPUT):
#   4 streams x 192 rows x 512 bytes, stream s based at SRC_s.
#   Stream s, ch c: at SRC_s + c row.
#   *** Only the first 64 FP32 lanes (256 bytes) of each row are valid data;
#   *** lanes 64..127 are unfold's stale r_acc padding and MUST be ignored --
#   this kernel never reads past lane 63 of a loaded stream row.
#   Stream s row layout: lanes [0..31] = that channel's STRIPE 0 tokens,
#   lanes [32..63] = STRIPE 1 tokens (see "CRITICAL LAYOUT" note below for
#   why this is the OPPOSITE of unfold's textual load order), in the same
#   decimated spatial order unfold's ACC.STRIDE produced them (row-major
#   within the 8x8 phase sub-grid: 4 sub-rows x 8 sub-cols).
#
# Output (NHCW striped, identical layout to unfold_16x16x192's INPUT):
#   2 stripes x 192 channels; each row = 8 spatial_rows x 16 cols = 128
#   elements (512 bytes in wide-vector debug mode).
#   Row (stripe, ch) at DST_BASE + (stripe x 192 + ch) row.
#
# CRITICAL LAYOUT DETAIL -- textual load order is NOT data order, because of
# the MULT snapshot contract (see below): unfold_16x16x192's ch_loop TEXTUALLY
# loads STRIPE 1 (cr13) first and STRIPE 0 (cr0) second for every stream, but
# MULT.RC.VV reads its Ra data from the START-OF-CYCLE snapshot, so each
# bundle's MULT actually consumes the PREVIOUS bundle's load, not the one
# issued in its own bundle (unfold primes this with a stripe-0 load before
# ch_loop). Net effect, confirmed empirically by loading a spatial tensor
# with distinct per-lane/per-stripe markers through the REAL unfold kernel
# and inspecting its output directly: ACC.STRIDE slot 0 (offset lr0=0, r_acc
# lanes 0..31) actually receives STRIPE 0's data (the primed load), and slot
# 1 (offset lr1=1, r_acc lanes 32..63) receives STRIPE 1's data -- the
# OPPOSITE of what the textual LDR order suggests. A first version of this
# kernel assumed the naive (textual-order) mapping and got a scrambled
# result that was NOT obviously wrong (every value present, just permuted)
# until checked against the round-trip test -- exactly the class of bug
# unfold_8x8x240's _ROW_PACK_ORDER note warns about. Trust the emulator, not
# the textual load order, when a kernel's loads run one bundle ahead of the
# MULT that consumes them.
#
# Consequently: fold's source window [0..31] of a loaded stream row feeds
# the STRIPE 0 destination row, and source window [32..63] feeds the
# STRIPE 1 destination row.
#
# ACC.RESHAPE mapping (derived by inverting unfold's execute_acc_stride
# selection with elements_in_row=16; verified in Python before writing this
# kernel and cross-checked against the emulator via the round-trip test):
#   For ONE destination stripe row (128 lanes), each stream's 32 valid
#   source lanes (of the correct half of its loaded row) scatter via 4
#   ACC.RESHAPE calls of 8 elements each:
#     source (same shape for every stream): [8c, 8c+1, ..., 8c+7]
#       -- stripe-0 pass (first): c's window is offset +0;
#          stripe-1 pass (second): +32.
#     dest, stream s, call c: base(s) + 32*c + 2*i for i in 0..7, where
#       base(TL)=0, base(TR)=1, base(BL)=16, base(BR)=17.
#   These 16 (stream, call) dest tables partition all 128 destination lanes
#   exactly once each -- confirmed by brute-force enumeration in Python
#   before writing this kernel (128 unique destination indices, range
#   0..127, zero overlap).
#
# Building the index tables (per user directive: never assume an LR holds
# anything but garbage at kernel start -- unlike CRs, which setup()
# legitimately initializes via state.regfile.set_cr):
#   SOURCE table (LRD2 = LR3:LR2) is rebuilt via two SETs from cr2/cr3
#   (packed bytes [0,1,2,3] / [4,5,6,7]) at the start of every stream's
#   4-call block, then walked forward +8 per call via ADDBI. For the
#   stripe-1 pass (source window base 32 instead of 0), the freshly-reset
#   [0..7] table is walked forward once more via ADDBI +32 before call 0 --
#   this reaches [32..39] in one step (confirmed: ADDBI is a per-lane
#   saturating byte add in [0, 255]; 7 + 32 = 39, far from saturating).
#
#   DEST table (LRD4 = LR5:LR4) is rebuilt via two SETs from cr4/cr5 (TL's
#   call-0 table, packed bytes [0,2,4,6] / [8,10,12,14]) then, for TR/BL/BR,
#   walked to that stream's base via a single ADDBI (+1, +16, +17
#   respectively -- TR = TL+1, BL = TL+16, BR = TL+17 on every lane
#   simultaneously, confirmed by direct computation). Across the 4 calls of
#   one stream, the SAME table is walked forward +32 per call (identical
#   step to the source table, coincidentally, but a different register).
#   The dest table does NOT depend on which stripe pass is active -- it is
#   rebuilt once per stream per stripe pass purely for auditability
#   (correctness-first per the task brief; a cycle-tuned version could cache
#   it across the two stripe passes for the same stream).
#
# Open questions from the design draft, resolved empirically against this
# emulator before trusting them:
#   1. ACTIVATE.QUANTIZE identity after ACC.RESHAPE (not ACC.STRIDE): fine.
#      execute_activate_quantize reads r_acc LIVE and has no ACC.STRIDE-
#      specific assumption; it just reads whatever is in the live r_acc
#      register file for the active-element count cr15's dstructure gives
#      (128, the default) -- confirmed by reading ipu.py directly.
#   2. r_acc does not need an explicit reset/zero before the first
#      ACC.RESHAPE of a stripe pass: the 16 (stream, call) dest tables
#      partition all 128 lanes exactly once (verified above), so every lane
#      is freshly written every stripe pass regardless of prior r_acc
#      contents. execute_acc_reshape only updates R_ACC[dest[i]] for
#      participating i (unlike ACC.STRIDE's unconditional block overwrite),
#      which is exactly why the partition property matters here. Verified
#      empirically too: test_fold_16x16x192_wide poisons r_acc with a
#      non-zero sentinel before the program runs and asserts the output is
#      still bit-exact.
#   3. ADDBI's per-lane saturating byte arithmetic produces the intended
#      tables at every step -- confirmed by direct byte-level computation
#      (see comments above); no lane ever approaches the [0, 255]
#      saturation boundary in this design (max index used is 127).
#
# MULT SNAPSHOT CONTRACT (issue #157, inherited from unfold_16x16x192): a
# MULT.RC.VV instruction reads its Ra DATA (r0) from the start-of-cycle
# snapshot, so it cannot consume a chunk loaded by an LDR_MULT_REG issued in
# the very same bundle -- co-issuing a load and a MULT in one bundle is fine
# only when the MULT consumes a PREVIOUSLY loaded chunk. This kernel primes
# the first load before ch_loop and keeps every subsequent LDR one bundle
# ahead of the MULT that consumes it, exactly like unfold_16x16x192.
#
# STR_POST_AAQ_REG reads its offset operand LIVE -- never co-issue an ADD on
# the offset LR (lr8) in the same bundle as the store, or the row lands one
# slot late. Here lr8 only advances in its own bundle at the very end of the
# channel loop, well clear of both stripe stores.
#
# CRs:
#   cr14 = DST_BASE + 0x192x128   (stripe 0 output base, ch 0..191)
#   cr13 = DST_BASE + 1x192x128   (stripe 1 output base)
#   -- NOTE: CR0 is ALSO read-only hardwired to 0 (not just CR1 -- see
#      CR_READ_ONLY_INITIAL_VALUES in ipu_config.py); unfold_16x16x192 gets
#      away with naming its stripe-0 SOURCE base cr0 only because that base
#      happens to equal 0 anyway. Fold's stripe-0 DESTINATION base is
#      DST_BASE_ROW (nonzero), so it cannot use cr0 -- this was a real bug
#      caught by the direct-reference test (silently wrote stripe 0's rows
#      to XMEM row 0 instead of DST_BASE_ROW). cr14 is used instead.
#   cr8  = ONES_BASE               (128 elements of dtype 1.0, for r_cyclic init)
#   cr9  = SRC_TL                  (stream TL input base)
#   cr10 = SRC_TR                  (stream TR input base)
#   cr11 = SRC_BL                  (stream BL input base)
#   cr12 = SRC_BR                  (stream BR input base)
#   cr2  = source table lo  bytes [0,1,2,3]     (LRD2 low word,  via SET)
#   cr3  = source table hi  bytes [4,5,6,7]     (LRD2 high word, via SET)
#   cr4  = TL dest table lo bytes [0,2,4,6]     (LRD4 low word,  via SET)
#   cr5  = TL dest table hi bytes [8,10,12,14]  (LRD4 high word, via SET)
#   cr1  = 1      (read-only hardwired constant)
#
# LRs (preset by harness):
#   lr0  = 0    (const: r_cyclic slot 0)
#   lrd2 = LR2:LR3  (working source-lane-index table for ACC.RESHAPE)
#   lrd4 = LR4:LR5  (working dest-lane-index table for ACC.RESHAPE)
#   lr6  = 0    (src row offset within a stream; += 1 per channel)
#   lr7  = 1    (src stride per channel, in rows)
#   lr8  = 0    (dst row offset = ch; += 1 per channel)
#   lr9  = 1    (dst stride per channel, in rows)
#   lr10 = 0    (ch counter, 0..191)
#   lr11 = 192  (loop limit = C)
#
# Memory layout (ROW numbers -- .asm XMEM operands are rows, issue #179):
#   SRC:  4 streams x 192 rows                      (TL, TR, BL, BR)
#   ONES: 1 row
#   DST:  2 stripes x 192 rows

    LDR_CYCLIC_MULT_REG lr0 cr8 lr0;;       # r_cyclic[0..127] = 1.0 (dtype-specific)

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..191)
# ---------------------------------------------------------------------------

    LDR_MULT_REG        r0 lr6 cr9;;        # prime: stream TL row of ch 0

ch_loop:

    # ===== Destination STRIPE 0 pass (source window = lanes [0..31]) =======

    # -- Stream TL: dest table = TL base [0,2,4,6,8,10,12,14] ----------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;           # source table <- [0,1,2,3,4,5,6,7]
    SET                 lr4 cr4;
    SET                 lr5 cr5;;           # dest table   <- TL base
    LDR_MULT_REG r0 lr6 cr10;
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply TL row (prefetch TR next)
    ACC.RESHAPE lrd2 lrd4 0;;               # call0: src[0..7]   -> dest[0,2,..,14]
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;               # call1: src[8..15]  -> dest[32,34,..,46]
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;               # call2: src[16..23] -> dest[64,66,..,78]
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;               # call3: src[24..31] -> dest[96,98,..,110]

    # -- Stream TR: dest table = TL base + 1 ---------------------------------
    # SET and a same-pair ADDBI cannot share a bundle -- both would target
    # LR4/LR5, and the emulator's LR-slot conflict check rejects any bundle
    # writing the same LR twice regardless of instruction identity -- so the
    # SET-reset and the ADDBI-offset each get their own word.
    SET                 lr2 cr2;
    SET                 lr3 cr3;;
    SET                 lr4 cr4;
    SET                 lr5 cr5;;           # dest table <- TL base
    ADDBI               lrd4 1;;            # dest table <- TR base
    LDR_MULT_REG r0 lr6 cr11;
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply TR row (prefetch BL next)
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;

    # -- Stream BL: dest table = TL base + 16 --------------------------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;
    SET                 lr4 cr4;
    SET                 lr5 cr5;;           # dest table <- TL base
    ADDBI               lrd4 16;;           # dest table <- BL base
    LDR_MULT_REG r0 lr6 cr12;
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply BL row (prefetch BR next)
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;

    # -- Stream BR: dest table = TL base + 17 --------------------------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;
    SET                 lr4 cr4;
    SET                 lr5 cr5;;           # dest table <- TL base
    ADDBI               lrd4 17;;           # dest table <- BR base
    # Prefetch the FIRST stripe-0-pass load (stream TL, same ch): lr6 does
    # NOT advance here -- both stripe passes read the SAME 4 stream rows for
    # this channel; only the source-table window (rebuilt explicitly per
    # stream below) differs between the two passes.
    LDR_MULT_REG r0 lr6 cr9;
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply BR row (prefetch TL next)
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;

    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG         lr8 cr14;;     # stripe 0 <- DST_BASE + ch (current channel)

    # ===== Destination STRIPE 1 pass (source window = lanes [32..63]) ======

    # -- Stream TL: dest table = TL base --------------------------------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;           # source table <- [0,1,2,3,4,5,6,7]
    SET                 lr4 cr4;
    SET                 lr5 cr5;;           # dest table <- TL base
    ADDBI               lrd2 32;;           # source table <- [32..39] (stripe-1 window)
    LDR_MULT_REG r0 lr6 cr10;
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply TL row (prefetch TR next)
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;

    # -- Stream TR: dest table = TL base + 1 ---------------------------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;
    SET                 lr4 cr4;
    SET                 lr5 cr5;
    ADDBI               lrd2 32;;           # source table <- [32..39]; dest table <- TL base
    ADDBI               lrd4 1;;            # dest table <- TR base
    LDR_MULT_REG r0 lr6 cr11;
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply TR row (prefetch BL next)
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;

    # -- Stream BL: dest table = TL base + 16 --------------------------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;
    SET                 lr4 cr4;
    SET                 lr5 cr5;
    ADDBI               lrd2 32;;           # source table <- [32..39]; dest table <- TL base
    ADDBI               lrd4 16;;           # dest table <- BL base
    LDR_MULT_REG r0 lr6 cr12;
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply BL row (prefetch BR next)
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;

    # -- Stream BR: dest table = TL base + 17 --------------------------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;
    SET                 lr4 cr4;
    SET                 lr5 cr5;
    ADDBI               lrd2 32;;           # source table <- [32..39]; dest table <- TL base
    ADDBI               lrd4 17;;           # dest table <- BR base
    # lr6 (src offset) must advance BEFORE the load that uses next channel's
    # offset: LR sub-slots run ahead of LOAD within a word (same hazard
    # unfold_16x16x192 documents for its lr4), so this ADD gets its own word.
    # lr8 (dst offset) is deliberately NOT advanced here -- the stripe-0
    # store below still targets the CURRENT channel's row; it advances only
    # once, in the loop tail after both stripe stores (mirrors unfold's lr8).
    ADD                 lr6 lr6 lr7;;       # src offset: next channel
    LDR_MULT_REG r0 lr6 cr9;
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply BR row (prefetch ch+1's TL)
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;

    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG         lr8 cr13;;     # stripe 1 <- DST_BASE + 192 + ch

    # -- Advance pointers; loop -------------------------------------------------
    ADD                 lr8 lr8 lr9;;       # dst offset: next channel (+1 row)
    ADD                 lr10 lr10 cr1;;
    BLT                 lr10 lr11 ch_loop;; # loop while ch < 192

end:
    BKPT;;
