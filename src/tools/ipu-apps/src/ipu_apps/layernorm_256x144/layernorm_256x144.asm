# LayerNorm 256×144: per-PE normalization across 144 channels, 256 tokens (2 tg × 128)
#
# Wide-vector debug mode (512 B per row = 128 × FP32).
# Same algorithm and VLIW patterns as layernorm_128x16. Key differences:
#   - N_CH=144, N_TG=2 → outer tg loop reuses all scratch buffers
#   - Data layout: DATA_BASE + (ch*N_TG + tg)*512  (channel-major, tg interleaved)
#   - γ/β have 144 values (>128) → reload r0/r1 at ch=128 in step 6
#   - NEG_INV_N = FP32(-1/144), INV_N = FP32(1/144)
#
# VLIW rules (same as 128x16):
#   (a) Startup-offset: init ptr to -stride; ADD fires first → live=0 on first use
#   (b) Loop counter init to 1: BLT reads snapshot (pre-ADD) → runs exactly N times
#   (c) r0/r1 snap: pre-load before loop, hold constant; MULT sees previous cycle's r0/r1
#   (d) r_cyclic live: LDR_CYCLIC + MULT in same cycle → cyclic is immediately visible
#
# CRs:
#   cr0  = DATA_BASE        = 0x00000  (hardwired 0; also the const-zero source)
#   cr1  = 1  (read-only hardwired constant; not used for a base)
#   cr2  = BETA_BASE        = 0x24400
#   cr3  = ONES_BASE        = 0x24800
#   cr4  = NEG_INV_N_BASE   = 0x24A00
#   cr5  = INV_N_BASE       = 0x24C00
#   cr6  = NEG_MEAN_BASE    = 0x24E00
#   cr7  = CENTERED_BASE    = 0x25000
#   cr8  = TEMP_BASE        = 0x37000
#   cr9  = INVSTD_BASE      = 0x37200
#   cr10 = OUTPUT_BASE      = 0x37400
#   cr11 = GAMMA_BASE       = 0x24000  (moved off read-only CR1)
#   cr12 = 144              (N_CH)
#   cr13 = 512              (row stride within one tg)
#   cr14 = 128              (valid_elements; r1 base offset for MULT.VE.CYCLIC)
#   cr15 is reserved
#
# Data stride between consecutive channels within one tg: 1024 B (= N_TG × 512)
# stored in lr7 (overriding cr13 used only for output/scratch stride).
#
# Persistent LRs:
#   lr0  = 0    (cyclic_offset=0, xmem offset=0)
#   lr1  = 0    (mask_shift=0)
#   lr6  = 144  (N_CH loop bound)
#   lr7  = 1024 (data stride per channel = N_TG*512)
#   lr11 = 2    (N_TG loop bound)
#   lr12 = 512  (scratch/output row stride)
# Per-step temporaries:
#   lr2, lr3 (offsets), lr5 (ch counter), lr9 (tg counter), lr10 (tg byte offset)
#   lr9, lr13, lr14 (fixed_idx in step 6)

    SET     lr0  cr0;;
    SET     lr1  cr0;;
    SET     lr6  cr12;;
    SET     lr7  cr13;;
    ADD     lr7  lr7  lr7;;            # lr7 = 1024 (data stride = N_TG × 512)
    SET     lr11 cr0;;
    ADD     lr11 lr11 2;;              # lr11 = 2  (N_TG)
    SET     lr12 cr13;;                # lr12 = 512 (scratch/output stride)

# ─────────────────────────────────────────────────────────────────────────────
# Outer loop: tg = 0, 1
# lr9  = tg counter (1..2, BLT-style)
# lr10 = tg byte offset into data (0 or 512)
# ─────────────────────────────────────────────────────────────────────────────

    SET     lr9  cr0;;
    ADD     lr9  lr9  1;;              # lr9 = 1  (tg counter, BLT reads snap)
    SET     lr10 cr0;;                # lr10 = 0  (tg byte offset)

tg_loop:

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: -μ[i] = Σ_ch  x[ch,i] × (-1/N)    for this tg
#
# Data row for (ch, tg): DATA_BASE + (ch*N_TG + tg)*512 = DATA_BASE + ch*1024 + tg*512
# lr2 starts at lr10 - 1024 = tg*512 - 1024.
# ADD lr2 lr2 lr7 (=1024) fires first → live = ch*1024 + tg*512 on iteration ch.
# ─────────────────────────────────────────────────────────────────────────────

    RESET_ACC;;
    LDR_MULT_REG        r0 lr0 cr4;;   # r0 ← -1/N

    SET     lr2  cr0;;
    SUB     lr2  lr2  lr7;;            # lr2 = -1024
    ADD     lr2  lr2  lr10;;           # lr2 = tg_offset - 1024
    SET     lr5  cr0;;
    ADD     lr5  lr5  1;;
step1_loop:
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0; ADD lr2 lr2 lr7; MULT.EE r0 lr0 0 lr1; ACC;;
    ADD     lr5  lr5  1; BLT lr5 lr6 step1_loop;;

    STR_ACC_REG         lr0 cr6;;      # NEG_MEAN_BASE = -μ

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: centered[ch,i] = x[ch,i] + (-μ[i])
#
# r0 = ONES, r1 = -μ. lr2 = data read ptr. lr3 = centered write ptr (-512).
# Per ch (3 cycles): same as 128x16.
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr3;;
    LDR_MULT_REG        r1 lr0 cr6;;

    SET     lr2  cr0;;
    SUB     lr2  lr2  lr7;;
    ADD     lr2  lr2  lr10;;
    SET     lr3  cr0;;
    SUB     lr3  lr3  lr12;;           # lr3 = -512 (centered stride)
    SET     lr5  cr0;;
    ADD     lr5  lr5  1;;
step2_loop:
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0; ADD lr2 lr2 lr7; MULT.EE r0 lr0 0 lr1; ACC.FIRST;;
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0; MULT.EE r1 lr0 0 lr1; ACC;;
    STR_ACC_REG         lr3 cr7; ADD lr3 lr3 lr12;;
    ADD     lr5  lr5  1; BLT lr5 lr6 step2_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Σ_ch (centered[ch,i])²    using MULT.EE.RR
#
# lr2 = -512 (centered read ptr). Two cycles per ch.
# ─────────────────────────────────────────────────────────────────────────────

    RESET_ACC;;

    SET     lr2  cr0;;
    SUB     lr2  lr2  lr12;;
    SET     lr5  cr0;;
    ADD     lr5  lr5  1;;
step3_loop:
    LDR_MULT_REG        r0 lr2 cr7; ADD lr2 lr2 lr12;;
    MULT.EE.RR          r0 0 lr1; ACC;;
    ADD     lr5  lr5  1; BLT lr5 lr6 step3_loop;;

    STR_ACC_REG         lr0 cr8;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: variance = (1/N) × Σ(x-μ)²;  1/σ = ACTIVATE rsqrt
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr8;;
    LDR_CYCLIC_MULT_REG lr0 cr5 lr0; MULT.EE r0 lr0 0 lr1; ACC.FIRST;;

    ACTIVATE            rsqrt 1;;
    STR_POST_AAQ_REG    lr0 cr9;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: normalized[ch,i] = centered[ch,i] × 1/σ[i]  (overwrite CENTERED)
#
# r0 = 1/σ. lr2 = -512. Three cycles per ch.
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr9;;

    SET     lr2  cr0;;
    SUB     lr2  lr2  lr12;;
    SET     lr5  cr0;;
    ADD     lr5  lr5  1;;
step5_loop:
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0; ADD lr2 lr2 lr12; MULT.EE r0 lr0 0 lr1; ACC.FIRST;;
    STR_ACC_REG         lr2 cr7;;
    ADD     lr5  lr5  1; BLT lr5 lr6 step5_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: output[ch,i] = γ[ch] × normalized[ch,i] + β[ch]
#
# N_CH=144 > 128: γ/β span two 512-byte rows.
#   Row 0: γ[0..127], β[0..127]
#   Row 1: γ[128..143], β[128..143] (rest zero-padded by harness)
#
# Sub-loop A: ch=0..127   → r0=γ row0, r1=β row0, lr13=fixed_idx 0..127
# Sub-loop B: ch=128..143 → reload r0=γ row1, r1=β row1, lr13=0..15
#
# Output: OUTPUT_BASE + (ch*N_TG + tg)*512 = OUTPUT_BASE + ch*1024 + tg*512
# lr3 = output write ptr, init = tg_offset - 1024.
# ADD lr3 lr3 lr7 fires → live = ch*1024+tg_offset on each STR cycle.
# But STR is in the same cycle as ADD lr3 and ADD lr13/lr14...
# To avoid incrementing lr3 in the same cycle as lr13/lr14, use 4 cycles per ch:
#   A: LDR_CYCLIC normalized[ch]; ADD lr2; MULT.VE.CYCLIC lr13; ACC.FIRST
#   B: LDR_CYCLIC ONES; MULT.VE.CYCLIC lr14; ACC
#   C: ADD lr3; STR output[ch]; ADD lr13; ADD lr14
#   D: ADD lr5; BLT
# ─────────────────────────────────────────────────────────────────────────────

    # ---- Sub-loop A: ch=0..127 ----
    LDR_MULT_REG        r0 lr0 cr11;;  # r0 ← γ row 0
    LDR_MULT_REG        r1 lr0 cr2;;   # r1 ← β row 0

    SET     lr2  cr0;;
    SUB     lr2  lr2  lr12;;           # normalized read ptr = -512
    SET     lr3  cr0;;
    SUB     lr3  lr3  lr7;;            # output write ptr = -1024
    ADD     lr3  lr3  lr10;;           # = tg_offset - 1024
    SET     lr5  cr0;;
    ADD     lr5  lr5  1;;
    SET     lr13 cr0;;                # fixed_idx γ = 0
    SET     lr14 cr14;;                # fixed_idx β = 128

    # loop bound for sub-loop A: 128 channels
    # lr6 currently = 144; use a separate bound lr15=128 for sub-loop A
    SET     lr15 cr14;;                # lr15 = 128

step6A_loop:
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0; ADD lr2 lr2 lr12; MULT.VE.CYCLIC lr0 0 lr1 lr13; ACC.FIRST;;
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0; MULT.VE.CYCLIC lr0 0 lr1 lr14; ACC;;
    STR_ACC_REG         lr3 cr10; ADD lr3 lr3 lr7; ADD lr13 lr13 1; ADD lr14 lr14 1;;
    ADD     lr5  lr5  1; BLT lr5 lr15 step6A_loop;;

    # ---- Sub-loop B: ch=128..143 (16 channels) ----
    LDR_MULT_REG        r0 lr12 cr11;; # r0 ← γ row 1 (offset=512)
    LDR_MULT_REG        r1 lr12 cr2;;  # r1 ← β row 1

    # lr2 and lr3 carry over from sub-loop A (already at ch=128 positions)
    SET     lr5  cr0;;
    ADD     lr5  lr5  1;;
    SET     lr13 cr0;;                # fixed_idx γ = 0 (row 1 starts at lane 0)
    SET     lr14 cr14;;                # fixed_idx β = 128

    # bound for sub-loop B: 16 channels
    SET     lr15 cr0;;
    ADD     lr15 lr15 16;;             # lr15 = 16

step6B_loop:
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0; ADD lr2 lr2 lr12; MULT.VE.CYCLIC lr0 0 lr1 lr13; ACC.FIRST;;
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0; MULT.VE.CYCLIC lr0 0 lr1 lr14; ACC;;
    STR_ACC_REG         lr3 cr10; ADD lr3 lr3 lr7; ADD lr13 lr13 1; ADD lr14 lr14 1;;
    ADD     lr5  lr5  1; BLT lr5 lr15 step6B_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Advance tg
# ─────────────────────────────────────────────────────────────────────────────

    ADD     lr10 lr10 lr12;;           # tg_offset += 512
    ADD     lr9  lr9  1; BLT lr9  lr11 tg_loop;;

end:
    BKPT;;
