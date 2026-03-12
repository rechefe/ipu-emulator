# Unfold 32×32×144 → 4 streams of [256, 144] channel-major FP32
#
# Rearranges a 32×32×144 spatial tensor (NHCW striped input) into
# 4 streams (TL, TR, BL, BR) of 16×16 sub-grids, channel-major.
#
# Input (NHCW striped):
#   8 stripes × 144 channels; each row = 4 spatial_rows × 32 cols = 128 bytes.
#   Row (stripe, ch) at SRC_BASE + (stripe × 144 + ch) × 128.
#
# Output (channel-major FP32):
#   4 streams × 288 rows × 512 bytes (128 FP32 words per row).
#   Stream s, ch c, tg t: at DST_s + (c×2 + t) × 512.
#
# Stream definitions (acc.stride mode with elements_in_row=32):
#   128 elements from one stripe = 4 rows × 32 cols
#   TL (stream 0): h=enabled  (even cols, rows 0/2),  v=enabled  (even rows)
#   TR (stream 1): h=inverted (odd  cols, rows 0/2),  v=enabled  (even rows)
#   BL (stream 2): h=enabled  (even cols, rows 1/3),  v=inverted (odd  rows)
#   BR (stream 3): h=inverted (odd  cols, rows 1/3),  v=inverted (odd  rows)
#
# Each acc.stride call selects 32 of 128 mult_result elements (2 rows × 16 cols)
# and accumulates them into one 32-element r_acc slot (slot 0..3).
# Four acc.stride calls fill the full 128-element accumulator per tg.
#
# CRs:
#   cr0  = SRC_BASE + 0×144×128   (stripe 0 tg=0 base, ch 0..143)
#   cr1  = SRC_BASE + 1×144×128   (stripe 1 tg=0 base)
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
#   lr0  = 0   (const: r_cyclic slot index 0; acc.stride r_acc slot 0 → [0..31])
#   lr1  = 1   (const: acc.stride r_acc slot 1 → [32..63])
#   lr2  = 2   (const: acc.stride r_acc slot 2 → [64..95])
#   lr3  = 3   (const: acc.stride r_acc slot 3 → [96..127])
#   lr4  = ch × 128   (src byte offset within each stripe; starts 0, +128 per ch)
#   lr8  = tg=0 dst byte offset = ch × 1024          (starts 0, +1024 per ch)
#   lr9  = tg=1 dst byte offset = ch × 1024 + 512    (starts 512, +1024 per ch)
#   lr10 = ch counter (0..143)
#   lr11 = 144 (loop limit)
#
# NOTE: All `set` immediates are ≤ 512 (within signed 16-bit range ±32767).
# Stripe base offsets (0..129024) are encoded in CRs, not LR immediates,
# because `set` sign-extends the 16-bit immediate and values > 32767 would
# become negative.
#
# Memory layout:
#   SRC:  8 × 144 × 128 B = 147,456 B  (0x00000..0x23FFF)
#   ONES: 128 B                         (0x24000..0x2407F)
#   DST:  4 × 288 × 512 B = 589,824 B  (0x30000..0xBBFFF)

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

    set                 lr0 0;;
    ldr_cyclic_mult_reg lr0 cr8 lr0;;       # r_cyclic[0..127] = 1.0 (dtype-specific)

    set lr1 1; set lr2 2;;
    set lr3 3; set lr4 0;;
    set lr8 0; set lr9 512;;
    set lr10 0; set lr11 144;;

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..143)
# ---------------------------------------------------------------------------

ch_loop:

    # -----------------------------------------------------------------------
    # Stream TL  (h=enabled=even cols,  v=enabled=even rows)
    # -----------------------------------------------------------------------
    # tg=0: stripes 0..3 with cr0..cr3
    reset_acc;;
    ldr_mult_reg mem_bypass lr4 cr0; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on on lr0;;
    ldr_mult_reg mem_bypass lr4 cr1; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on on lr1;;
    ldr_mult_reg mem_bypass lr4 cr2; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on on lr2;;
    ldr_mult_reg mem_bypass lr4 cr3; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on on lr3;;
    str_acc_reg         lr8 cr9;;           # TL tg=0 → DST_TL + ch×1024

    # tg=1: stripes 4..7 with cr4..cr7
    reset_acc;;
    ldr_mult_reg mem_bypass lr4 cr4; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on on lr0;;
    ldr_mult_reg mem_bypass lr4 cr5; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on on lr1;;
    ldr_mult_reg mem_bypass lr4 cr6; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on on lr2;;
    ldr_mult_reg mem_bypass lr4 cr7; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on on lr3;;
    str_acc_reg         lr9 cr9;;           # TL tg=1 → DST_TL + ch×1024 + 512

    # -----------------------------------------------------------------------
    # Stream TR  (h=inverted=odd cols,  v=enabled=even rows)
    # -----------------------------------------------------------------------
    reset_acc;;
    ldr_mult_reg mem_bypass lr4 cr0; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand on lr0;;
    ldr_mult_reg mem_bypass lr4 cr1; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand on lr1;;
    ldr_mult_reg mem_bypass lr4 cr2; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand on lr2;;
    ldr_mult_reg mem_bypass lr4 cr3; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand on lr3;;
    str_acc_reg         lr8 cr10;;          # TR tg=0

    reset_acc;;
    ldr_mult_reg mem_bypass lr4 cr4; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand on lr0;;
    ldr_mult_reg mem_bypass lr4 cr5; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand on lr1;;
    ldr_mult_reg mem_bypass lr4 cr6; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand on lr2;;
    ldr_mult_reg mem_bypass lr4 cr7; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand on lr3;;
    str_acc_reg         lr9 cr10;;          # TR tg=1

    # -----------------------------------------------------------------------
    # Stream BL  (h=enabled=even cols,  v=inverted=odd rows)
    # -----------------------------------------------------------------------
    reset_acc;;
    ldr_mult_reg mem_bypass lr4 cr0; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on reserved3 lr0;;
    ldr_mult_reg mem_bypass lr4 cr1; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on reserved3 lr1;;
    ldr_mult_reg mem_bypass lr4 cr2; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on reserved3 lr2;;
    ldr_mult_reg mem_bypass lr4 cr3; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on reserved3 lr3;;
    str_acc_reg         lr8 cr11;;          # BL tg=0

    reset_acc;;
    ldr_mult_reg mem_bypass lr4 cr4; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on reserved3 lr0;;
    ldr_mult_reg mem_bypass lr4 cr5; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on reserved3 lr1;;
    ldr_mult_reg mem_bypass lr4 cr6; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on reserved3 lr2;;
    ldr_mult_reg mem_bypass lr4 cr7; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on reserved3 lr3;;
    str_acc_reg         lr9 cr11;;          # BL tg=1

    # -----------------------------------------------------------------------
    # Stream BR  (h=inverted=odd cols,  v=inverted=odd rows)
    # -----------------------------------------------------------------------
    reset_acc;;
    ldr_mult_reg mem_bypass lr4 cr0; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand reserved3 lr0;;
    ldr_mult_reg mem_bypass lr4 cr1; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand reserved3 lr1;;
    ldr_mult_reg mem_bypass lr4 cr2; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand reserved3 lr2;;
    ldr_mult_reg mem_bypass lr4 cr3; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand reserved3 lr3;;
    str_acc_reg         lr8 cr12;;          # BR tg=0

    reset_acc;;
    ldr_mult_reg mem_bypass lr4 cr4; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand reserved3 lr0;;
    ldr_mult_reg mem_bypass lr4 cr5; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand reserved3 lr1;;
    ldr_mult_reg mem_bypass lr4 cr6; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand reserved3 lr2;;
    ldr_mult_reg mem_bypass lr4 cr7; mult.ev mem_bypass lr0 lr0 lr0; acc.stride 32 on_expand reserved3 lr3;;
    str_acc_reg         lr9 cr12;;          # BR tg=1

    # -----------------------------------------------------------------------
    # Advance pointers; loop
    # -----------------------------------------------------------------------
    incr                lr4 128;;            # src offset: next channel within each stripe
    incr                lr8 1024; incr lr9 1024;;   # dst offsets: skip 2 rows × 512 B
    incr                lr10 1;;
    blt                 lr10 lr11 ch_loop;;  # loop while ch < 144

end:
    bkpt;;
