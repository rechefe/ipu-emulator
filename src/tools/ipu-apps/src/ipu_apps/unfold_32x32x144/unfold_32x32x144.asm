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
#              MULT.RC.VV acc_slot0 r0 0 acc_slot0: r0[i] × r_cyclic[i] = stripe[i] × 1.0 = stripe[i].
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
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.
#
# Register assignments and their meanings are listed in the register-name
# block below -- it is the single source of truth for this kernel.
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
#   acc_slot0=0, acc_slot1=1, acc_slot2=2, acc_slot3=3, src_off=0, src_stride=128, dst_stride=1024,
#   dst_off_tg0=0, dst_off_tg1=512, ch_index=0, ch_limit=144

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set acc_slot0   = "lr0"  %}  {# const 0: r_cyclic slot / mask imm / ACC.STRIDE slot 0 #}
{% set acc_slot1   = "lr1"  %}  {# const 1: ACC.STRIDE r_acc slot 1 -> [32..63] #}
{% set acc_slot2   = "lr2"  %}  {# const 2: ACC.STRIDE r_acc slot 2 -> [64..95] #}
{% set acc_slot3   = "lr3"  %}  {# const 3: ACC.STRIDE r_acc slot 3 -> [96..127] #}
{% set src_off     = "lr4"  %}  {# src byte offset within each stripe (ch*128) #}
{% set src_stride  = "lr5"  %}  {# 128 = src stride per channel #}
{% set dst_stride  = "lr6"  %}  {# 1024 = dst stride per channel (2 rows x 512B) #}
{% set dst_off_tg0 = "lr8"  %}  {# tg=0 dst byte offset = ch*1024 #}
{% set dst_off_tg1 = "lr9"  %}  {# tg=1 dst byte offset = ch*1024 + 512 #}
{% set ch_index    = "lr10" %}  {# channel counter (0..143) #}
{% set ch_limit    = "lr11" %}  {# 144 = channel count #}

{% set STRIPE0_TG0 = "cr0"  %}  {# SRC_BASE + 0*144*128 #}
{% set ONE         = "cr1"  %}  {# hardwired read-only 1 #}
{% set STRIPE2_TG0 = "cr2"  %}  {# SRC_BASE + 2*144*128 #}
{% set STRIPE3_TG0 = "cr3"  %}  {# SRC_BASE + 3*144*128 #}
{% set STRIPE0_TG1 = "cr4"  %}  {# SRC_BASE + 4*144*128 #}
{% set STRIPE1_TG1 = "cr5"  %}  {# SRC_BASE + 5*144*128 #}
{% set STRIPE2_TG1 = "cr6"  %}  {# SRC_BASE + 6*144*128 #}
{% set STRIPE3_TG1 = "cr7"  %}  {# SRC_BASE + 7*144*128 #}
{% set ONES_BASE   = "cr8"  %}  {# 128 bytes of dtype 1.0 (r_cyclic init) #}
{% set DST_TL      = "cr9"  %}  {# stream TL output base #}
{% set DST_TR      = "cr10" %}  {# stream TR output base #}
{% set DST_BL      = "cr11" %}  {# stream BL output base #}
{% set DST_BR      = "cr12" %}  {# stream BR output base #}
{% set STRIPE1_TG0 = "cr13" %}  {# SRC_BASE + 1*144*128 #}
{% set DSTRUCT     = "cr15" %}  {# reserved dstructure register #}

    LDR_CYCLIC_MULT_REG {{ acc_slot0 }} {{ ONES_BASE }} {{ acc_slot0 }};;  # r_cyclic[0..127] = 1.0 (dtype-specific)

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..143)
# ---------------------------------------------------------------------------

ch_loop:

    # -----------------------------------------------------------------------
    # Stream TL  (h=on=even cols,  v=on=even rows)
    # -----------------------------------------------------------------------
    # tg=0: stripes 0..3 with STRIPE0_TG0..STRIPE3_TG0
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE0_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on {{ acc_slot0 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE1_TG0 }};MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on {{ acc_slot1 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE2_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on {{ acc_slot2 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE3_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on {{ acc_slot3 }};;
    STR_ACC_REG         {{ dst_off_tg0 }} {{ DST_TL }};;                   # TL tg=0 → DST_TL + ch×2×512

    # tg=1: stripes 4..7 with STRIPE0_TG1..STRIPE3_TG1
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE0_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on {{ acc_slot0 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE1_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on {{ acc_slot1 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE2_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on {{ acc_slot2 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE3_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on {{ acc_slot3 }};;
    STR_ACC_REG         {{ dst_off_tg1 }} {{ DST_TL }};;                   # TL tg=1 → DST_TL + ch×1024 + 512

    # -----------------------------------------------------------------------
    # Stream TR  (h=on_inv=odd cols,  v=on=even rows)
    # -----------------------------------------------------------------------
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE0_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on {{ acc_slot0 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE1_TG0 }};MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on {{ acc_slot1 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE2_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on {{ acc_slot2 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE3_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on {{ acc_slot3 }};;
    STR_ACC_REG         {{ dst_off_tg0 }} {{ DST_TR }};;                   # TR tg=0

    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE0_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on {{ acc_slot0 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE1_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on {{ acc_slot1 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE2_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on {{ acc_slot2 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE3_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on {{ acc_slot3 }};;
    STR_ACC_REG         {{ dst_off_tg1 }} {{ DST_TR }};;                   # TR tg=1

    # -----------------------------------------------------------------------
    # Stream BL  (h=on=even cols,  v=on_inv=odd rows)
    # -----------------------------------------------------------------------
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE0_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on_inv {{ acc_slot0 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE1_TG0 }};MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on_inv {{ acc_slot1 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE2_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on_inv {{ acc_slot2 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE3_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on_inv {{ acc_slot3 }};;
    STR_ACC_REG         {{ dst_off_tg0 }} {{ DST_BL }};;                   # BL tg=0

    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE0_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on_inv {{ acc_slot0 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE1_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on_inv {{ acc_slot1 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE2_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on_inv {{ acc_slot2 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE3_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on on_inv {{ acc_slot3 }};;
    STR_ACC_REG         {{ dst_off_tg1 }} {{ DST_BL }};;                   # BL tg=1

    # -----------------------------------------------------------------------
    # Stream BR  (h=on_inv=odd cols,  v=on_inv=odd rows)
    # -----------------------------------------------------------------------
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE0_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on_inv {{ acc_slot0 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE1_TG0 }};MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on_inv {{ acc_slot1 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE2_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on_inv {{ acc_slot2 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE3_TG0 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on_inv {{ acc_slot3 }};;
    STR_ACC_REG         {{ dst_off_tg0 }} {{ DST_BR }};;                   # BR tg=0

    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE0_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on_inv {{ acc_slot0 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE1_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on_inv {{ acc_slot1 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE2_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on_inv {{ acc_slot2 }};;
    LDR_MULT_REG r0 {{ src_off }} {{ STRIPE3_TG1 }}; MULT.RC.VV {{ acc_slot0 }} r0 0 {{ acc_slot0 }} {{ DSTRUCT }}; ACC.STRIDE 32 on_inv on_inv {{ acc_slot3 }};;
    STR_ACC_REG         {{ dst_off_tg1 }} {{ DST_BR }};;                   # BR tg=1

    # -----------------------------------------------------------------------
    # Advance pointers; loop
    # -----------------------------------------------------------------------
    ADD                 {{ src_off }} {{ src_off }} {{ src_stride }};;     # src offset: next channel (+128)
    ADD                 {{ dst_off_tg0 }} {{ dst_off_tg0 }} {{ dst_stride }}; ADD {{ dst_off_tg1 }} {{ dst_off_tg1 }} {{ dst_stride }};; # dst offsets: +1024 per channel
    ADD                 {{ ch_index }} {{ ch_index }} {{ ONE }};;
    BLT                 {{ ch_index }} {{ ch_limit }} ch_loop;;            # loop while ch < 144

end:
    BKPT;;
