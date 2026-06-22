# Unfold 32×32×144 → 4 streams of [256, 144] channel-major FP32
#
# Rearranges a 32×32×144 spatial tensor (NHCW striped input) into
# 4 streams (TL, TR, BL, BR) of 16×16 sub-grids, channel-major.
#
# Input (NHCW striped):
#   8 stripes × 144 channels; each row = 4 spatial_rows × 32 cols = 128 bytes.
#   Row (stripe, ch) at SRC_BASE + (stripe × 144 + ch) × 128.
#
# Output (interleaved channel-major FP32):
#   4 streams × 288 rows × 512 bytes (128 FP32 words per row).
#   Stream s, ch c, tg t: at DST_s + (c×2 + t) × 512.
#
# Pass-through multiply: MULT.RC.VV (post-merge; was MULT.EE, itself replacing
#   MULT.EV mem_bypass removed in PR #69).
#   stripe row in r0 (via LDR_MULT_REG r0), ones preloaded into r_cyclic once.
#              MULT.RC.VV lr0 r0 0 lr0: r0[i] × r_cyclic[i] = stripe[i] × 1.0 = stripe[i].
#   r_cyclic[0..127] = 1.0 (loaded once at startup, never overwritten in the loop).
#   RESET_ACC removed: the 4 ACC.STRIDE calls direct-write all 4 r_acc slots.
#
# Stream definitions (acc.stride mode with elements_in_row=32):
#   128 elements from one stripe = 4 rows × 32 cols
#   TL (stream 0): acc.stride 32 on     on     → even cols, even rows
#   TR (stream 1): acc.stride 32 on_inv on     → odd  cols, even rows
#   BL (stream 2): acc.stride 32 on     on_inv → even cols, odd  rows
#   BR (stream 3): acc.stride 32 on_inv on_inv → odd  cols, odd  rows
#
# acc.stride enum encoding (from acc_stride_enums.py):
#   horizontal: on=1 (even cols), on_inv=2 (odd cols)
#   vertical:   on=1 (even rows), on_inv=2 (odd rows)
#
# Each acc.stride call selects 32 of 128 mult_result elements (2 rows × 16 cols)
# and accumulates them into one 32-element r_acc slot (slot 0..3).
# Four acc.stride calls fill the full 128-element accumulator per tg.
#
# CRs:
#   cr0  = SRC_BASE + 0×144×128   (stripe 0 tg=0 base, ch 0..143)
#   cr13 = SRC_BASE + 1×144×128   (stripe 1 tg=0 base; moved off read-only CR1)
#   cr2  = SRC_BASE + 2×144×128   (stripe 2 tg=0 base)
#   cr3  = SRC_BASE + 3×144×128   (stripe 3 tg=0 base)
#   cr4  = SRC_BASE + 4×144×128   (stripe 0 tg=1 base)
#   cr5  = SRC_BASE + 5×144×128   (stripe 1 tg=1 base)
#   cr6  = SRC_BASE + 6×144×128   (stripe 2 tg=1 base)
#   cr7  = SRC_BASE + 7×144×128   (stripe 3 tg=1 base)
#   cr8  = ONES_BASE               (128 bytes of dtype 1.0, for r_cyclic init)
#   cr9  = DST_TL                  (stream TL output base)
#   cr10 = DST_TR                  (stream TR output base)
#   cr11 = DST_BL                  (stream BL output base)
#   cr12 = DST_BR                  (stream BR output base)
#
# LRs:
#   lr0  = 0    (const: r_cyclic slot 0; mask_offset immediate=0; mask_shift=0; acc.stride slot 0)
#   lr1  = 1    (const: acc.stride r_acc slot 1 → [32..63])
#   lr2  = 2    (const: acc.stride r_acc slot 2 → [64..95])
#   lr3  = 3    (const: acc.stride r_acc slot 3 → [96..127])
#   lr4  = ch × 128   (src byte offset within each stripe; starts 0, +128 per ch)
#   lr5  = 128  (src stride per channel)
#   lr6  = 1024 (dst stride per channel = 2 rows × 512B)
#   lr8  = tg=0 dst byte offset = ch × 1024            (starts 0, +1024 per ch)
#   lr9  = tg=1 dst byte offset = ch×1024 + 512       (starts 512, +1024 per ch)
#   lr10 = ch counter (0..143)
#   lr11 = 144  (loop limit)
#
# NOTE: All `set` immediates fit signed 16-bit range.
# Stripe base offsets (0..129024) are encoded in CRs, not LR immediates.
#
# Memory layout:
#   SRC:  8 × 144 × 128 B = 147,456 B  (0x00000..0x23FFF)
#   ONES: 128 B                         (0x24000..0x2407F)
#   DST:  4 × 288 × 512 B = 589,824 B  (0x30000..0xBBFFF)

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
# Constant LRs are preset by the harness (set_lr):
#   lr0=0, lr1=1, lr2=2, lr3=3, lr4=0, lr5=128, lr6=1024,
#   lr8=0, lr9=512, lr10=0, lr11=144

    LDR_CYCLIC_MULT_REG lr0 cr8 lr0;;       # r_cyclic[0..127] = 1.0 (dtype-specific)

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..143)
# ---------------------------------------------------------------------------

ch_loop:

    # -----------------------------------------------------------------------
    # Stream TL  (h=on=even cols,  v=on=even rows)
    # -----------------------------------------------------------------------
    # tg=0: stripes 0..3 with cr0..cr3
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr1;;
    LDR_MULT_REG r0 lr4 cr2; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr2;;
    LDR_MULT_REG r0 lr4 cr3; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr3;;
    STR_ACC_REG         lr8 cr9;;           # TL tg=0 → DST_TL + ch×2×512

    # tg=1: stripes 4..7 with cr4..cr7
    LDR_MULT_REG r0 lr4 cr4; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr0;;
    LDR_MULT_REG r0 lr4 cr5; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr1;;
    LDR_MULT_REG r0 lr4 cr6; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr2;;
    LDR_MULT_REG r0 lr4 cr7; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on lr3;;
    STR_ACC_REG         lr9 cr9;;           # TL tg=1 → DST_TL + ch×1024 + 512

    # -----------------------------------------------------------------------
    # Stream TR  (h=on_inv=odd cols,  v=on=even rows)
    # -----------------------------------------------------------------------
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr1;;
    LDR_MULT_REG r0 lr4 cr2; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr2;;
    LDR_MULT_REG r0 lr4 cr3; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr3;;
    STR_ACC_REG         lr8 cr10;;          # TR tg=0

    LDR_MULT_REG r0 lr4 cr4; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr0;;
    LDR_MULT_REG r0 lr4 cr5; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr1;;
    LDR_MULT_REG r0 lr4 cr6; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr2;;
    LDR_MULT_REG r0 lr4 cr7; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on lr3;;
    STR_ACC_REG         lr9 cr10;;          # TR tg=1

    # -----------------------------------------------------------------------
    # Stream BL  (h=on=even cols,  v=on_inv=odd rows)
    # -----------------------------------------------------------------------
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr1;;
    LDR_MULT_REG r0 lr4 cr2; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr2;;
    LDR_MULT_REG r0 lr4 cr3; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr3;;
    STR_ACC_REG         lr8 cr11;;          # BL tg=0

    LDR_MULT_REG r0 lr4 cr4; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr5; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr1;;
    LDR_MULT_REG r0 lr4 cr6; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr2;;
    LDR_MULT_REG r0 lr4 cr7; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on on_inv lr3;;
    STR_ACC_REG         lr9 cr11;;          # BL tg=1

    # -----------------------------------------------------------------------
    # Stream BR  (h=on_inv=odd cols,  v=on_inv=odd rows)
    # -----------------------------------------------------------------------
    LDR_MULT_REG r0 lr4 cr0; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr13;MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr1;;
    LDR_MULT_REG r0 lr4 cr2; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr2;;
    LDR_MULT_REG r0 lr4 cr3; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr3;;
    STR_ACC_REG         lr8 cr12;;          # BR tg=0

    LDR_MULT_REG r0 lr4 cr4; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr0;;
    LDR_MULT_REG r0 lr4 cr5; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr1;;
    LDR_MULT_REG r0 lr4 cr6; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr2;;
    LDR_MULT_REG r0 lr4 cr7; MULT.RC.VV lr0 r0 0 lr0; ACC.STRIDE 32 on_inv on_inv lr3;;
    STR_ACC_REG         lr9 cr12;;          # BR tg=1

    # -----------------------------------------------------------------------
    # Advance pointers; loop
    # -----------------------------------------------------------------------
    ADD                 lr4 lr4 lr5;;            # src offset: next channel (+128)
    ADD                 lr8 lr8 lr6; ADD lr9 lr9 lr6;;  # dst offsets: +1024 per channel
    ADD                 lr10 lr10 cr1;;
    BLT                 lr10 lr11 ch_loop;;      # loop while ch < 144

end:
    BKPT;;
