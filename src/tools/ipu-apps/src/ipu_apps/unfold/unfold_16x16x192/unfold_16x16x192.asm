# Unfold 16×16×192 → 4 streams of [64, 192] channel-major FP32   (L4)
#
# Layer:   L4
# Scope:   single-stream (per-stripe; emits the 4 spatial streams)
# Layout:  unpacked
# Shape:   16x16x192 spatial input -> 4 streams x [64tok, 192chan]
# Status:  validated
# Related: L4 port of unfold_32x32x144 (L3) with re-derived geometry (NOT a
#          direct port -- 2 stripes not 8, elements_in_row=16 not 32);
#          unfold_8x8x240 is the L5 sibling; feeds layernorm_64x192
# Tests:   test_unfold_16x16x192_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Rearranges a 16×16×192 spatial tensor (NHCW striped input) into
# 4 streams (TL, TR, BL, BR) of 8×8 sub-grids, channel-major.
#
# NOT a port of unfold_32x32x144 — the geometry differs. L3 loads 4 spatial
# rows of 32 cols per 128-byte row and has 8 stripes; L4 loads 8 rows of 16
# cols and has only 2 stripes. The ACC.STRIDE decimation still yields 32
# elements per call (4 rows × 8 cols), so the four-slot structure carries over,
# but a stream fills only 2 slots instead of 8.
#
# Input (NHCW striped):
#   2 stripes × 192 channels; each row = 8 spatial_rows × 16 cols = 128 bytes.
#   Row (stripe, ch) at SRC_BASE + (stripe × 192 + ch) × 128.
#
# Output (per-stream channel-major FP32, HALF-ROW PADDED):
#   4 streams × 192 rows × 512 bytes, stream s based at DST_s.
#   Stream s, ch c: at DST_s + c × 512.
#
#   *** Each row carries 64 valid FP32 tokens (256 bytes); the upper 256 bytes
#   *** are STALE r_acc lanes 64..127 and must be ignored by consumers.
#   A stream contributes 2 × 32 = 64 tokens, which half-fills r_acc, and
#   STR_POST_AAQ_REG unconditionally writes all 512 bytes of post_aaq_reg, and
#   ACTIVATE.QUANTIZE identity stages all 128 lanes of r_acc into it (DSTRUCT
#   carries the default valid_elements=128), so lanes 64..127 carry the same
#   stale r_acc values the old STR_ACC_REG path wrote.
#   This is the deliberate per-stream layout (plan §4); the packed [k][p·n]
#   variant that would use the full row is the deferred §9 experiment.
#
# Pass-through multiply: MULT.RC.VV with r_cyclic preloaded to dtype-1.0,
#   stripe row in r0 (via LDR_MULT_REG r0):
#     MULT.RC.VV lr0 r0 0 lr0: r0[i] × r_cyclic[i] = stripe[i] × 1.0 = stripe[i].
#   r_cyclic[0..127] = 1.0 (loaded once at startup, never overwritten).
#
# Stream definitions (acc.stride mode with elements_in_row=16):
#   128 elements from one stripe = 8 rows × 16 cols
#   TL (stream 0): acc.stride 16 on     on     → even cols, even rows
#   TR (stream 1): acc.stride 16 on_inv on     → odd  cols, even rows
#   BL (stream 2): acc.stride 16 on     on_inv → even cols, odd  rows
#   BR (stream 3): acc.stride 16 on_inv on_inv → odd  cols, odd  rows
#   Verified against execute_acc_stride: each selects 32 elements
#   (4 rows × 8 cols) with the expected even/odd row and column split.
#
# acc.stride enum encoding (from acc_stride_enums.py):
#   elements_in_row: "16" (enum 0)
#   horizontal: on=1 (even cols), on_inv=2 (odd cols)
#   vertical:   on=1 (even rows), on_inv=2 (odd rows)
#
# STR_POST_AAQ_REG reads its offset LIVE — never co-issue an ADD on the offset LR in
# the same bundle as the store, or the row lands one slot late (Phase 0, §Q-3).
# Here the dst pointer lr8 advances in its own bundle at the end of the channel
# loop, well clear of every store.
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VV reads its r0 DATA from the
# start-of-cycle snapshot, so a MULT cannot consume a chunk loaded by an
# LDR_MULT_REG in its own bundle — it would see the PREVIOUS contents of r0.
# `;;` ends one VLIW word = one cycle = one snapshot, so a load and a MULT in
# the same bundle always execute in the same cycle regardless of textual order;
# co-issuing them is fine, consuming the same-cycle load is not.
# Therefore the loads here run one bundle AHEAD of the MULT that consumes them:
# the stripe-0 load for the first stream is primed before ch_loop, and each
# bundle's LDR prefetches the chunk the NEXT bundle will multiply. The eight
# loads per channel form a fixed stripe-0/stripe-1 alternation, and lr4 (the
# src offset, read LIVE by LDR) only advances once per channel, so the rotation
# is a pure one-bundle shift: the last load of channel ch fetches stripe 0 of
# channel ch+1, which is why lr4's ADD must precede it.
#
# CRs:
#   cr0  = SRC_BASE + 0×192×128   (stripe 0 base, ch 0..191)
#   cr13 = SRC_BASE + 1×192×128   (stripe 1 base; CR1 is read-only, so use cr13)
#   cr8  = ONES_BASE              (128 bytes of dtype 1.0, for r_cyclic init)
#   cr9  = DST_TL                 (stream TL output base)
#   cr10 = DST_TR                 (stream TR output base)
#   cr11 = DST_BL                 (stream BL output base)
#   cr12 = DST_BR                 (stream BR output base)
#   cr1  = 1                      (read-only hardwired constant)
#
# LRs (preset by harness):
#   lr0  = 0    (const: r_cyclic slot 0; mask_offset=0; mask_shift=0; acc.stride slot 0)
#   lr1  = 1    (const: acc.stride r_acc slot 1 → [32..63])
#   lr4  = 0    (src byte offset within each stripe; += 128 per ch)
#   lr5  = 128  (src stride per channel)
#   lr6  = 512  (dst stride per channel)
#   lr8  = 0    (dst byte offset = ch × 512)
#   lr10 = 0    (ch counter, 0..191)
#   lr11 = 192  (loop limit = C)
#
# Memory layout:
#   SRC:  2 × 192 × 128 B =  49,152 B  (0x00000..0x0BFFF)
#   ONES: 128 B                        (0x0C000..0x0C07F)
#   DST:  4 × 192 × 512 B = 393,216 B  (0x30000..0x8FFFF)
#         stream0 0x30000  stream1 0x48000  stream2 0x60000  stream3 0x78000

    LDR_CYCLIC_MULT_REG lr0 cr8 lr0;;       # r_cyclic[0..127] = 1.0 (dtype-specific)

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..191)
# ---------------------------------------------------------------------------

    LDR_MULT_REG        r0 lr4 cr0;;        # prime: stripe 0 of ch 0 (TL slot 0)

ch_loop:

    # Each bundle multiplies the chunk loaded by the PREVIOUS bundle and loads
    # the chunk the NEXT bundle will multiply (snapshot contract, see header).

    # -- Stream TL  (h=on=even cols, v=on=even rows) -------------------------
    LDR_MULT_REG r0 lr4 cr13;
    MULT.RC.VV lr0 r0 0 lr0 cr15;
    ACC.STRIDE 16 on on lr0;;
    LDR_MULT_REG r0 lr4 cr0;
    MULT.RC.VV lr0 r0 0 lr0 cr15;
    ACC.STRIDE 16 on on lr1;;
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG         lr8 cr9;;  # TL → DST_TL + ch×512 (lanes 0..63 valid)

    # -- Stream TR  (h=on_inv=odd cols, v=on=even rows) ----------------------
    LDR_MULT_REG r0 lr4 cr13;
    MULT.RC.VV lr0 r0 0 lr0 cr15;
    ACC.STRIDE 16 on_inv on lr0;;
    LDR_MULT_REG r0 lr4 cr0;
    MULT.RC.VV lr0 r0 0 lr0 cr15;
    ACC.STRIDE 16 on_inv on lr1;;
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG         lr8 cr10;;  # TR

    # -- Stream BL  (h=on=even cols, v=on_inv=odd rows) ----------------------
    LDR_MULT_REG r0 lr4 cr13;
    MULT.RC.VV lr0 r0 0 lr0 cr15;
    ACC.STRIDE 16 on on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr0;
    MULT.RC.VV lr0 r0 0 lr0 cr15;
    ACC.STRIDE 16 on on_inv lr1;;
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG         lr8 cr11;;  # BL

    # -- Stream BR  (h=on_inv=odd cols, v=on_inv=odd rows) -------------------
    LDR_MULT_REG r0 lr4 cr13;
    MULT.RC.VV lr0 r0 0 lr0 cr15;
    ACC.STRIDE 16 on_inv on_inv lr0;;
    # lr4 must advance BEFORE the load that uses the next channel's offset: LR
    # sub-slots run ahead of LOAD within a word, so this ADD gets its own word.
    ADD                 lr4 lr4 lr5;;       # src offset: next channel (+128)
    LDR_MULT_REG r0 lr4 cr0;
    MULT.RC.VV lr0 r0 0 lr0 cr15;
    ACC.STRIDE 16 on_inv on_inv lr1;;
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG         lr8 cr12;;  # BR

    # -- Advance pointers; loop ----------------------------------------------
    ADD                 lr8 lr8 lr6;;       # dst offset: next channel (+512)
    ADD                 lr10 lr10 cr1;;
    BLT                 lr10 lr11 ch_loop;; # loop while ch < 192

end:
    BKPT;;
