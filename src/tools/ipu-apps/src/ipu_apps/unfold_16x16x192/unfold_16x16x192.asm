# Unfold 16×16×192 → 4 streams of [64, 192] channel-major FP32   (L4)
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
#   STR_ACC_REG unconditionally writes all 512 bytes of r_acc (ipu.py:446-455).
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
# STR_ACC_REG reads its offset LIVE — never co-issue an ADD on the offset LR in
# the same bundle as the store, or the row lands one slot late (Phase 0, §Q-3).
# Here the dst pointer lr8 advances in its own bundle at the end of the channel
# loop, well clear of every store.
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

ch_loop:

    # -- Stream TL  (h=on=even cols, v=on=even rows) -------------------------
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0 cr15; ACC.STRIDE 16 on on lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0 cr15; ACC.STRIDE 16 on on lr1;;
    STR_ACC_REG         lr8 cr9;;           # TL → DST_TL + ch×512 (lanes 0..63 valid)

    # -- Stream TR  (h=on_inv=odd cols, v=on=even rows) ----------------------
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0 cr15; ACC.STRIDE 16 on_inv on lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0 cr15; ACC.STRIDE 16 on_inv on lr1;;
    STR_ACC_REG         lr8 cr10;;          # TR

    # -- Stream BL  (h=on=even cols, v=on_inv=odd rows) ----------------------
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0 cr15; ACC.STRIDE 16 on on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0 cr15; ACC.STRIDE 16 on on_inv lr1;;
    STR_ACC_REG         lr8 cr11;;          # BL

    # -- Stream BR  (h=on_inv=odd cols, v=on_inv=odd rows) -------------------
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0 cr15; ACC.STRIDE 16 on_inv on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0 cr15; ACC.STRIDE 16 on_inv on_inv lr1;;
    STR_ACC_REG         lr8 cr12;;          # BR

    # -- Advance pointers; loop ----------------------------------------------
    ADD                 lr4 lr4 lr5;;       # src offset: next channel (+128)
    ADD                 lr8 lr8 lr6;;       # dst offset: next channel (+512)
    ADD                 lr10 lr10 cr1;;
    BLT                 lr10 lr11 ch_loop;; # loop while ch < 192

end:
    BKPT;;
