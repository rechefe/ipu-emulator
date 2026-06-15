# LayerNorm 128×16: per-PE normalization across 16 channels, 128 tokens
#
# Wide-vector debug mode (512 B per row = 128 × FP32).
#
# VLIW execution order: LR → XMEM → MULT → ACC → AAQ → COND
#   r0/r1 snap  : captured before slots fire; MULT sees previous cycle's r0/r1
#   r_cyclic live: XMEM updates r_cyclic before MULT reads it
#   LR fires BEFORE XMEM: any ADD on an offset applies before the address is used
#
# Key patterns used:
#   (a) Startup-offset: init ptr to -stride → ADD fires → live = 0 on first use
#   (b) Loop counter: BLT reads snapshot (pre-ADD) → init counter to 1 so loop
#       runs exactly N_CH times (snapshot reaches N_CH after N_CH increments from 1)
#   (c) r0/r1 as scalar source: pre-load before loop, never change inside loop
#   (d) r_cyclic as row data: LDR_CYCLIC in same cycle as MULT → live visible to MULT
#
# CRs:
#   cr0  = DATA_BASE        = 0x00000
#   cr1  = GAMMA_BASE       = 0x02000
#   cr2  = BETA_BASE        = 0x02200
#   cr3  = ONES_BASE        = 0x02400
#   cr4  = NEG_INV_N_BASE   = 0x02600
#   cr5  = INV_N_BASE       = 0x02800
#   cr6  = NEG_MEAN_BASE    = 0x02A00
#   cr7  = CENTERED_BASE    = 0x02C00
#   cr8  = TEMP_BASE        = 0x04E00
#   cr9  = INVSTD_BASE      = 0x05000
#   cr10 = OUTPUT_BASE      = 0x05200
#   cr11 = 0   (const zero)
#   cr12 = 16  (N_CH)
#   cr13 = 512 (row stride)
#   cr14 = 128 (valid_elements; r1 base offset for MULT.VE.CYCLIC)
#
# Persistent LRs:
#   lr0 = 0   (cyclic_offset=0, single-row xmem offset=0)
#   lr1 = 0   (mask_shift=0)
#   lr6 = 16  (N_CH loop bound)
#   lr7 = 512 (row stride)
#   lr8 = 128 (valid_elements for ACTIVATE)
# Per-step temporaries: lr2, lr3 (offsets), lr5 (counter), lr9, lr10 (fixed_idx)

    SET     lr0  cr11;;
    SET     lr1  cr11;;
    SET     lr6  cr12;;
    SET     lr7  cr13;;
    SET     lr8  cr14;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: -μ[i] = Σ_ch  x[ch,i] × (-1/N)
#
# r0 = -1/N (pre-loaded). lr2 = -512 (startup). lr5 = 1 (counter init).
# Body: LDR_CYCLIC x[ch] live; MULT.EE r0_snap×cyclic_live; ACC.
# ─────────────────────────────────────────────────────────────────────────────

    RESET_ACC;;
    LDR_MULT_REG        r0 lr0 cr4;;   # r0 ← -1/N

    SET     lr2  cr11;;
    SUB     lr2  lr2 lr7;;             # lr2 = -512
    SET     lr5  cr11;;
    ADD     lr5  lr5 1;;               # lr5 = 1  (BLT reads snapshot; loop runs N_CH times)
step1_loop:
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0; ADD lr2 lr2 lr7; MULT.EE r0 lr0 0 lr1; ACC;;
    ADD     lr5  lr5 1; BLT lr5 lr6 step1_loop;;

    STR_ACC_REG         lr0 cr6;;      # NEG_MEAN_BASE = -μ

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: centered[ch,i] = x[ch,i] + (-μ[i])
#
# r0 = ONES, r1 = -μ (pre-loaded). lr2 = -512 (data read). lr3 = -512 (centered write).
# Per ch (3 cycles):
#   A: LDR_CYCLIC x[ch]; ADD lr2; MULT.EE r0_snap(ONES)×cyclic_live → ACC.FIRST
#   B: LDR_CYCLIC ONES;  MULT.EE r1_snap(-μ)×cyclic_live(ONES) → ACC
#   C: STR centered[ch]; ADD lr3
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr3;;   # r0 ← ONES
    LDR_MULT_REG        r1 lr0 cr6;;   # r1 ← -μ

    SET     lr2  cr11;;
    SUB     lr2  lr2 lr7;;
    SET     lr3  cr11;;
    SUB     lr3  lr3 lr7;;
    SET     lr5  cr11;;
    ADD     lr5  lr5 1;;
step2_loop:
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0; ADD lr2 lr2 lr7; MULT.EE r0 lr0 0 lr1; ACC.FIRST;;
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0; MULT.EE r1 lr0 0 lr1; ACC;;
    STR_ACC_REG         lr3 cr7; ADD lr3 lr3 lr7;;
    ADD     lr5  lr5 1; BLT lr5 lr6 step2_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Σ_ch (centered[ch,i])²
#
# MULT.EE.RR uses r0 snapshot. lr2 = -512. Two cycles per ch:
#   A: LDR_MULT_REG r0 ← centered[ch]; ADD lr2 (no MULT/ACC)
#   B: MULT.EE.RR r0 (snap = centered[ch]); ACC
# ─────────────────────────────────────────────────────────────────────────────

    RESET_ACC;;

    SET     lr2  cr11;;
    SUB     lr2  lr2 lr7;;
    SET     lr5  cr11;;
    ADD     lr5  lr5 1;;
step3_loop:
    LDR_MULT_REG        r0 lr2 cr7; ADD lr2 lr2 lr7;;
    MULT.EE.RR          r0 0 lr1; ACC;;
    ADD     lr5  lr5 1; BLT lr5 lr6 step3_loop;;

    STR_ACC_REG         lr0 cr8;;      # TEMP_BASE = Σ(x-μ)²

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: variance = (1/N) × Σ(x-μ)²;  1/σ = ACTIVATE inv_sqrt
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr8;;
    LDR_CYCLIC_MULT_REG lr0 cr5 lr0; MULT.EE r0 lr0 0 lr1; ACC.FIRST;;

    ACTIVATE            lr8 inv_sqrt;;
    STR_POST_AAQ_REG    lr0 cr9;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: normalized[ch,i] = centered[ch,i] × 1/σ[i]  (overwrite CENTERED)
#
# r0 = 1/σ. lr2 = -512. Three cycles per ch:
#   A: LDR_CYCLIC centered[ch]; ADD lr2; MULT.EE r0_snap×cyclic_live → ACC.FIRST
#   B: STR_ACC_REG at lr2 (live = row*512, no ADD this cycle)
#   C: ADD lr5; BLT
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr9;;

    SET     lr2  cr11;;
    SUB     lr2  lr2 lr7;;
    SET     lr5  cr11;;
    ADD     lr5  lr5 1;;
step5_loop:
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0; ADD lr2 lr2 lr7; MULT.EE r0 lr0 0 lr1; ACC.FIRST;;
    STR_ACC_REG         lr2 cr7;;
    ADD     lr5  lr5 1; BLT lr5 lr6 step5_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: output[ch,i] = γ[ch] × normalized[ch,i] + β[ch]
#
# r0 = γ, r1 = β (pre-loaded). lr2 = -512 (normalized read). lr3 = -512 (output write).
# lr9 = 0 (fixed_idx γ), lr10 = 128 (fixed_idx β).
# fixed_idx is live; do NOT increment in the same cycle as MULT.VE.CYCLIC reads it.
#
# Per ch (4 cycles):
#   A: LDR_CYCLIC normalized[ch]; ADD lr2; MULT.VE.CYCLIC lr9; ACC.FIRST
#   B: LDR_CYCLIC ONES; MULT.VE.CYCLIC lr10; ACC
#   C: ADD lr3; STR output[ch]; ADD lr9; ADD lr10
#   D: ADD lr5; BLT
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr1;;
    LDR_MULT_REG        r1 lr0 cr2;;

    SET     lr2  cr11;;
    SUB     lr2  lr2 lr7;;
    SET     lr3  cr11;;
    SUB     lr3  lr3 lr7;;
    SET     lr5  cr11;;
    ADD     lr5  lr5 1;;
    SET     lr9  cr11;;               # fixed_idx γ = 0
    SET     lr10 cr14;;               # fixed_idx β = 128

step6_loop:
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0; ADD lr2 lr2 lr7; MULT.VE.CYCLIC lr0 0 lr1 lr9; ACC.FIRST;;
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0; MULT.VE.CYCLIC lr0 0 lr1 lr10; ACC;;
    STR_ACC_REG         lr3 cr10; ADD lr3 lr3 lr7; ADD lr9 lr9 1; ADD lr10 lr10 1;;
    ADD     lr5  lr5 1; BLT lr5 lr6 step6_loop;;

end:
    BKPT;;
