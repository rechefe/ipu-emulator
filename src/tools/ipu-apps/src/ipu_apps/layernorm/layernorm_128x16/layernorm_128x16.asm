# LayerNorm 128×16: per-PE normalization across 16 channels, 128 tokens
#
# Layer:   not layer-specific (16-channel LayerNorm, not one of the L3/L4/L5
#          transformer-block shapes; see kernel_docs/kernel_layer_map.md)
# Scope:   single-stream
# Layout:  unpacked
# Shape:   128tok x 16chan
# Status:  validated
# Related: ancestor of layernorm_256x144/layernorm_64x192/layernorm_16x240
#          (those document their differences relative to this kernel)
# Tests:   test_layernorm_128x16 (src/tools/ipu-apps/BUILD.bazel)
#
# Wide-vector debug mode (512 B per row = 128 × FP32).
#
# VLIW execution order: LR → XMEM → MULT → ACC → AAQ → COND
#   r0/r1 snap    : captured before slots fire; MULT sees previous cycle's r0/r1
#   r_cyclic snap : ALSO captured before slots fire (issue #157) -- MULT sees the
#                   value r_cyclic held at the START of the cycle, so a MULT
#                   cannot consume a row that LDR_CYCLIC loads in its own bundle
#   LR fires BEFORE XMEM: any ADD on an offset applies before the address is used
#
# `;;` ends one VLIW word = one cycle = one snapshot; `;` separates slots within
# that word. A load and a MULT written in the same bundle therefore always
# execute in the same cycle regardless of textual order -- co-issuing them is
# fine, CONSUMING the same-cycle load is not.
#
# Key patterns used:
#   (a) Startup-offset: init ptr to -stride → ADD fires → live = 0 on first use
#   (b) Loop counter: BLT reads snapshot (pre-ADD) → init counter to 1 so loop
#       runs exactly N_CH times (snapshot reaches N_CH after N_CH increments from 1)
#   (c) r0/r1 as scalar source: pre-load before loop, never change inside loop
#   (d) r_cyclic as row data: the LDR_CYCLIC feeding a MULT runs one bundle
#       EARLIER than that MULT. Each step primes its first row before the loop,
#       and every loop body multiplies the row loaded last cycle while loading
#       the next one. The offset LR (lr2) is read LIVE by the load, so its ADD
#       stays co-issued with the load -- which is exactly what makes the
#       co-issued load fetch the NEXT channel rather than the current one.
#
# CRs:
#   cr0  = DATA_BASE        = 0x00000  (hardwired 0; also the const-zero source)
#   cr1  = 1  (read-only hardwired constant; not used for a base)
#   cr2  = BETA_BASE        = 0x02200
#   cr3  = ONES_BASE        = 0x02400
#   cr4  = NEG_INV_N_BASE   = 0x02600
#   cr5  = INV_N_BASE       = 0x02800
#   cr6  = NEG_MEAN_BASE    = 0x02A00
#   cr7  = CENTERED_BASE    = 0x02C00
#   cr8  = TEMP_BASE        = 0x04E00
#   cr9  = INVSTD_BASE      = 0x05000
#   cr10 = OUTPUT_BASE      = 0x05200
#   cr11 = GAMMA_BASE      = 0x02000  (moved off read-only CR1)
#   cr12 = 16  (N_CH)
#   cr13 = 512 (row stride)
#   cr14 = 128 (valid_elements; r1 base offset for MULT.VE.CYCLIC)
#
# Persistent LRs:
#   lr0 = 0   (cyclic_offset=0, single-row xmem offset=0)
#   lr1 = 0   (mask_shift=0)
#   lr6 = 16  (N_CH loop bound)
#   lr7 = 512 (row stride)
# Per-step temporaries: lr2, lr3 (offsets), lr5 (counter), lr9, lr10 (fixed_idx)

    SET     lr0  cr0;;
    SET     lr1  cr0;;
    SET     lr6  cr12;;
    SET     lr7  cr13;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: -μ[i] = Σ_ch  x[ch,i] × (-1/N)
#
# r0 = -1/N (pre-loaded). lr2 = -512 (startup). lr5 = 1 (counter init).
# Body: MULT r0_snap × x[ch] (loaded last cycle); ACC; LDR_CYCLIC x[ch+1].
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr4;;   # r0 ← -1/N

    SET     lr2  cr0;;
    SUB     lr2  lr2 lr7;;             # lr2 = -512
    SET     lr5  cr0;;
    ADD     lr5  lr5 cr1;;             # lr5 = 1  (BLT reads snapshot; loop runs N_CH times)

    LDR_CYCLIC_MULT_REG lr2 cr0 lr0;
    ADD lr2 lr2 lr7;;  # prime x[ch=0]

    # Peeled first ch (ch=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    MULT.RC.VV lr0 r0 0 lr1 cr15;
    ACC.ADD.FIRST;;
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0;
    ADD lr2 lr2 lr7;
    ADD lr5 lr5 cr1;
    BLT lr5 lr6 step1_loop;;
    B       step1_done;;
step1_loop:
    MULT.RC.VV lr0 r0 0 lr1 cr15;
    ACC.ADD;;
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0;
    ADD lr2 lr2 lr7;
    ADD lr5 lr5 cr1;
    BLT lr5 lr6 step1_loop;;
step1_done:

    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG lr0 cr6;;  # NEG_MEAN_BASE = -μ

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: centered[ch,i] = x[ch,i] + (-μ[i])
#
# r0 = ONES, r1 = -μ (pre-loaded). lr2 = -512 (data read). lr3 = -512 (centered write).
# Per ch (3 cycles), each MULT consuming the row loaded one cycle earlier:
#   A: MULT r0_snap(ONES) × x[ch] → ACC.FIRST;  LDR_CYCLIC ONES (for B)
#   B: MULT r1_snap(-μ) × ONES    → ACC
#   C: STR centered[ch]; ADD lr3;  LDR_CYCLIC x[ch+1] (prefetch for next A)
# x[ch=0] is primed before the loop; the trailing prefetch reads one row past
# the last channel, which is harmless (the value is never multiplied).
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr3;;   # r0 ← ONES
    LDR_MULT_REG        r1 lr0 cr6;;   # r1 ← -μ

    SET     lr2  cr0;;
    SUB     lr2  lr2 lr7;;
    SET     lr3  cr0;;
    SUB     lr3  lr3 lr7;;
    SET     lr5  cr0;;
    ADD     lr5  lr5 cr1;;
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0;
    ADD lr2 lr2 lr7;;  # prime x[ch=0]
step2_loop:
    MULT.RC.VV lr0 r0 0 lr1 cr15;
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0;;
    MULT.RC.VV lr0 r1 0 lr1 cr15;
    ACC.ADD;;
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG lr3 cr7;
    ADD lr3 lr3 lr7;
    LDR_CYCLIC_MULT_REG lr2 cr0 lr0;
    ADD lr2 lr2 lr7;;
    ADD     lr5  lr5 cr1;
    BLT lr5 lr6 step2_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Σ_ch (centered[ch,i])²
#
# MULT.EE.RR uses r0 snapshot. lr2 = -512. Two cycles per ch:
#   A: LDR_MULT_REG r0 ← centered[ch]; ADD lr2 (no MULT/ACC)
#   B: MULT.EE.RR r0 (snap = centered[ch]); ACC
# ─────────────────────────────────────────────────────────────────────────────

    SET     lr2  cr0;;
    SUB     lr2  lr2 lr7;;
    SET     lr5  cr0;;
    ADD     lr5  lr5 cr1;;

    LDR_CYCLIC_MULT_REG lr2 cr7 lr0;
    ADD lr2 lr2 lr7;;  # prime centered[ch=0]

    # Peeled first ch (ch=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    # MULT.EE.RR (square r0) -> MULT.RC.VS (square r_cyclic): centered[ch] was
    # loaded into r_cyclic a cycle earlier, and is squared in place here.
    MULT.RC.VS lr0 0 lr1 cr15;
    ACC.ADD.FIRST;;
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0;
    ADD lr2 lr2 lr7;
    ADD lr5 lr5 cr1;
    BLT lr5 lr6 step3_loop;;
    B       step3_done;;
step3_loop:
    MULT.RC.VS lr0 0 lr1 cr15;
    ACC.ADD;;
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0;
    ADD lr2 lr2 lr7;
    ADD lr5 lr5 cr1;
    BLT lr5 lr6 step3_loop;;
step3_done:

    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG lr0 cr8;;  # TEMP_BASE = Σ(x-μ)²

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: variance = (1/N) × Σ(x-μ)²;  1/σ = ACTIVATE rsqrt
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr8;;
    LDR_CYCLIC_MULT_REG lr0 cr5 lr0;;   # load 1/N a cycle ahead of the MULT
    MULT.RC.VV lr0 r0 0 lr1 cr15;
    ACC.ADD.FIRST;;

    ACTIVATE.QUANTIZE rsqrt cr15;;
    STR_POST_AAQ_REG    lr0 cr9;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: normalized[ch,i] = centered[ch,i] × 1/σ[i]  (overwrite CENTERED)
#
# r0 = 1/σ. lr2 = load ptr (-512 startup), lr3 = store ptr (0, trails by one row).
# This step reads and writes the SAME rows, so the load running a cycle ahead of
# its MULT means the load and the store can no longer share one pointer: lr2
# now points at the row being PREFETCHED while lr3 points at the row whose
# product is being stored. Three cycles per ch:
#   A: MULT r0_snap × centered[ch] (loaded last cycle) → ACC.FIRST
#   B: STR_ACC_REG normalized[ch] at lr3; ADD lr3;
#      LDR_CYCLIC centered[ch+1]; ADD lr2   (prefetch; reads before the next
#      iteration's store, so it still sees the pre-overwrite value)
#   C: ADD lr5; BLT
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr9;;

    SET     lr2  cr0;;
    SUB     lr2  lr2 lr7;;
    SET     lr3  cr0;;
    SET     lr5  cr0;;
    ADD     lr5  lr5 cr1;;
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0;
    ADD lr2 lr2 lr7;;  # prime centered[ch=0]
step5_loop:
    MULT.RC.VV lr0 r0 0 lr1 cr15;
    ACC.ADD.FIRST;;
    # STR_ACC_REG reads lr3 LIVE, so lr3 must NOT be bumped in the store word --
    # it advances in the branch word below, exactly as the original store
    # pointer did.
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG lr3 cr7;
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0;
    ADD lr2 lr2 lr7;;
    ADD     lr3 lr3 lr7;
    ADD lr5 lr5 cr1;
    BLT lr5 lr6 step5_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: output[ch,i] = γ[ch] × normalized[ch,i] + β[ch]
#
# r0 = γ, r1 = β (pre-loaded). lr2 = -512 (normalized read). lr3 = -512 (output write).
# lr9 = 0 (fixed_idx γ), lr10 = 128 (fixed_idx β).
# fixed_idx is live; do NOT increment in the same cycle as MULT.VE.CYCLIC reads it.
#
# Per ch (4 cycles), each MULT consuming the row loaded one cycle earlier:
#   A: MULT.RC.VE lr9 × normalized[ch] → ACC.FIRST;  LDR_CYCLIC ONES (for B)
#   B: MULT.RC.VE lr10 × ONES → ACC
#   C: STR output[ch]; ADD lr3; ADD lr9; ADD lr10;
#      LDR_CYCLIC normalized[ch+1]; ADD lr2   (prefetch for the next A)
#   D: ADD lr5; BLT
# lr9/lr10 (fixed_idx) are still read LIVE by their MULTs and are still bumped
# in cycle C, one cycle before the A that reads them -- unchanged by the load
# shift, since only the r_cyclic loads moved.
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 lr0 cr11;;
    LDR_MULT_REG        r1 lr0 cr2;;

    SET     lr2  cr0;;
    SUB     lr2  lr2 lr7;;
    SET     lr3  cr0;;
    SUB     lr3  lr3 lr7;;
    SET     lr5  cr0;;
    ADD     lr5  lr5 cr1;;
    SET     lr9  cr0;;               # fixed_idx γ = 0
    SET     lr10 cr14;;              # fixed_idx β = 128

    LDR_CYCLIC_MULT_REG lr2 cr7 lr0;
    ADD lr2 lr2 lr7;;  # prime normalized[ch=0]
step6_loop:
    MULT.RC.VE lr0 lr9 0 lr1 cr15;
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG lr0 cr3 lr0;;
    MULT.RC.VE lr0 lr10 0 lr1 cr15;
    ACC.ADD;;
    # Cycle C is already at the 3-LR-per-word limit (lr3, lr9, lr10), so the
    # prefetch load rides here but its lr2 bump moves to cycle D. lr2 is read
    # LIVE, so the load still uses the pre-bump value = the row it should fetch.
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG lr3 cr10;
    ADD lr3 lr3 lr7;
    ADD lr9 lr9 cr1;
    ADD lr10 lr10 cr1;;
    LDR_CYCLIC_MULT_REG lr2 cr7 lr0;
    ADD lr2 lr2 lr7;
    ADD lr5 lr5 cr1;
    BLT lr5 lr6 step6_loop;;

end:
    BKPT;;
