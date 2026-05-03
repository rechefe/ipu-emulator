# LayerNorm 256×144
#
# out[ch, t] = γ[ch] × (x[ch,t] − mean_t) / sqrt(var_t) + β[ch]
# for 256 tokens (2 tg × 128 tokens) and 144 channels, dtype = FP8_E4M3.
#
# aaq uses INT8 truncation (val >> 24) at 5 points where FP32 must re-enter mult.
# Each is marked: // TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
#
# Memory layout (no overlaps, total < 2 MB):
#   DATA_BASE     = 0x00000   x[144 ch × 2 tg × 128 tok] = 36,864 B    ends 0x08FFF
#   GAMMA_BASE    = 0x0A000   γ: row0=γ[0..127], row1=γ[128..143+zeros] ends 0x0A0FF
#   BETA_BASE     = 0x0A100   β: same layout                             ends 0x0A1FF
#   ONES_BASE     = 0x0A200   128 × FP8(1.0) = 0x38                     ends 0x0A27F
#   NEG_ONES_BASE = 0x0A280   128 × FP8(-1.0) = 0xB8                    ends 0x0A2FF
#   MASK_BASE     = 0x0A300   128 masks × 16 B, 8/block = 2048 B        ends 0x0AAFF
#   TEMP_BASE     = 0x0AB00   3 × 128 B scratch                         ends 0x0AB7F
#   OUTPUT_BASE   = 0x0AC00   144 × 2 × 512 B = 147,456 B               ends 0x2FBFF
#
# CR assignments (loaded by harness):
#   cr0  = DATA_BASE     = 0x00000
#   cr1  = GAMMA_BASE    = 0x0A000
#   cr2  = BETA_BASE     = 0x0A100
#   cr3  = MASK_BASE     = 0x0A300
#   cr4  = ONES_BASE     = 0x0A200
#   cr5  = TEMP_BASE     = 0x0AB00
#   cr6  = OUTPUT_BASE   = 0x0AC00
#   cr7  = NEG_ONES_BASE = 0x0A280
#   cr15 = DType (set by harness)
#
# LR constants (set once at init):
#   lr0  = 0     cyclic slot-0 index; mask_shift; const-zero for addressing
#   lr1  = 2     tg loop limit
#   lr2  = 8     tokens-per-batch limit
#   lr3  = 128   γ/β row-1 offset; also sub-loop 4A limit
#   lr4  = 16    sub-loop 4B limit (ch=128..143)
#   lr5  = 256   data stride per channel
#   lr7  = 144   step1/step2 loop limit (N_CH)
#   lr8  = 1024  output stride per channel
#
# LR loop variables:
#   lr9  = tg             (0..1)
#   lr10 = batch          (0..15)
#   lr11 = mask_blk_off   (batch*128, running offset into MASK_BASE)
#   lr12 = tok            (0..7, within batch; also used as mask_offset)
#   lr13 = data_ptr       (offset from DATA_BASE)
#   lr14 = ch_idx         (channel index within γ/β row: 0..127 or 0..15)
#   lr15 = out_ptr        (offset from OUTPUT_BASE)
#
# PIPELINE TIMING (live LR semantics):
#   Slot order: LR fires first, then XMEM/MULT/ACC/COND all read LIVE register values.
#   All LrIdx operands in XMEM/MULT/COND instructions use the POST-incr value.
#   Pattern: init ptr to (first_addr - stride); incr fires → live = first_addr on first use.
#
# OUTPUT LAYOUT (interleaved tg):
#   Row (ch*2 + tg) at OUTPUT_BASE + (ch*2+tg) * 512 bytes.
#   str_acc_reg live lr15: init = out_start - 1024; incr 1024 fires → live = out_start on ch=0.
#     tg=0: init=-1024, ch=k → live offset k*1024       → row 2k   ✓
#     tg=1: init=-512,  ch=k → live offset 512+k*1024   → row 2k+1 ✓

# ===========================================================================
# ONE-TIME INITIALIZATION
# ===========================================================================

    set lr0 0;;
    set lr1 2;;
    set lr2 8;;
    set lr3 128;;
    set lr4 16;;
    set lr5 256;;
    set lr7 144;;
    set lr8 1024;;

# ===========================================================================
# OUTER LOOP: tg = 0, 1
# ===========================================================================

    set lr9  0;;
    set lr11 0;;    # mask_blk_off (reset at each tg start)

tg_loop:
    set lr10 0;;    # batch = 0

batch_loop:
    # Load 128-byte mask block: r_mask = MASK_BASE + batch*128
    ldr_mult_mask_reg lr11 cr3;;

    set lr12 0;;    # tok = 0

token_loop:

    # =========================================================================
    # STEP 1: Σx across channels (masked to lane t)
    #
    # Live-LR pattern: init lr13 = tg_offset - 256.
    # Each cycle: incr lr13 256 fires first → live lr13 = tg_offset + ch*256.
    # Reads x[ch=0] on first iteration. ✓
    # =========================================================================

    reset_acc;;
    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # r_cyclic = ONES

    # lr13 = tg_offset - 256  (live-LR startup: incr fires before XMEM read)
    set lr13 -256;;
    bne lr9 lr0 step1_tg1;;
    b step1_go;;
step1_tg1:
    set lr13 -128;;
step1_go:
    set lr14 0;;

step1_loop:
    ldr_mult_reg mem_bypass lr13 cr0;  incr lr13 256;  mult.ee mem_bypass lr0 lr12 lr0;  acc;  incr lr14 1;  blt lr14 lr7 step1_loop;;

    agg sum value cr0 aaq0;;   # aaq0 = Σx (raw int32 sum)
    agg sum value cr0 aaq1;;   # aaq1 = Σx (same value, used in step 4)

    # =========================================================================
    # STEP 2: Σx² across channels (masked to lane t)
    #
    # 3-cycle body per channel (no LR in cycles A or B):
    #   Cycle A: ldr_cyclic_mult_reg lr13 → r_cyclic = x[ch]  (no incr → live = current lr13)
    #   Cycle B: ldr_mult_reg r0 lr13     → r0       = x[ch]  (no incr → same live lr13)
    #   Cycle C: mult.ee r0 (x[ch] × r_cyclic[x[ch]]) + acc + incr lr13 256 + blt
    #
    # Init lr13 = exact tg_offset (no startup offset since cycle A has no incr).
    # =========================================================================

    reset_acc;;

    set lr13 0;;
    bne lr9 lr0 step2_tg1;;
    b step2_go;;
step2_tg1:
    set lr13 128;;
step2_go:
    set lr14 0;;

step2_loop_A:
    ldr_cyclic_mult_reg lr13 cr0 lr0;;   # Cycle A: r_cyclic = x[ch] (no LR incr)

    ldr_mult_reg r0 lr13 cr0;;           # Cycle B: r0 = x[ch] (same lr13, no LR incr)

    mult.ee r0 lr0 lr12 lr0;  acc;  incr lr14 1;  incr lr13 256;  blt lr14 lr7 step2_loop_A;;   # Cycle C

    agg sum value cr0 aaq2;;   # aaq2 = Σx²

    # =========================================================================
    # STEP 3: variance and inv_std
    #
    # All ldr_mult_reg here use lr0=0 with no incr in same cycle → no offset issue.
    # xmem.store_aaq_result uses lr0=0 and lr3=128, both unchanged here.
    # =========================================================================

    # 3a: mean scalar → TEMP_BASE+0
    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # r_cyclic = ones
    mult.ve.aaq lr0 lr12 lr0 aaq0;  acc.first;;
    # TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
    aaq;;
    xmem.store_aaq_result lr0 cr5;;     # TEMP_BASE+0 = mean

    # 3b: -mean → TEMP_BASE+128
    ldr_cyclic_mult_reg lr0 cr7 lr0;;   # r_cyclic = neg_ones
    ldr_mult_reg mem_bypass lr0 cr5;  mult.ee mem_bypass lr0 lr12 lr0;  acc.first;;
    # TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
    aaq;;
    xmem.store_aaq_result lr3 cr5;;     # TEMP_BASE+128 = -mean

    # 3c: mean × (-mean) → r_acc[t]
    ldr_cyclic_mult_reg lr3 cr5 lr0;;   # r_cyclic = TEMP_BASE+128 (-mean)
    ldr_mult_reg mem_bypass lr0 cr5;  mult.ee mem_bypass lr0 lr12 lr0;  acc.first;;

    # 3d: + E[x²] → var; inv_std
    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # r_cyclic = ones
    mult.ve.aaq lr0 lr12 lr0 aaq2;  acc;;

    agg sum inv_sqrt cr0 aaq3;;         # aaq3 = inv_std (float32 bits)

    # =========================================================================
    # STEP 4: per-channel affine  out[ch,t] = γ[ch]×(x[ch,t]-mean)/std + β[ch]
    #
    # Live-LR pattern:
    #   lr13 init = tg_offset - 256  → first ldr_mult reads x[ch=0] after incr ✓
    #   lr15 init = out_start - 1024 → first str_acc_reg writes to out_start after incr ✓
    #
    # Per channel (11 cycles):
    #  C1:  ldr x[ch]; incr lr13 256; mult.ee masked; acc.add_aaq.first aaq1
    #  C2:  aaq
    #  C3:  xmem.store_aaq_result → TEMP_BASE+0  (x-mean, truncation #3)
    #  C4:  ldr_cyclic TEMP_BASE+0; mult.ve.aaq aaq3 masked; acc.first
    #  C5:  aaq
    #  C6:  xmem.store_aaq_result → TEMP_BASE+0  (normalized, truncation #4)
    #  C7:  ldr_cyclic γ row
    #  C8:  ldr TEMP_BASE+0; mult.ev γ[live lr14] masked; acc.first
    #  C9:  ldr_cyclic β row
    # C10:  ldr ones; mult.ev β[live lr14] masked; acc
    # C11:  incr lr14 1; incr lr15 1024; str_acc_reg (live lr15) cr6; blt lr14 limit
    #
    # mult.ev uses live lr14. incr lr14 1 is ONLY in C11 (no LR in C8/C10) → live lr14
    # in C8/C10 = current ch_idx (unchanged). ✓
    #
    # Sub-loop 4A: ch=0..127, γ/β row 0, limit lr3=128.
    # Sub-loop 4B: ch=128..143, γ/β row 1 (offset lr3=128), limit lr4=16.
    # lr13 and lr15 carry over from 4A into 4B.
    # =========================================================================

    # lr13: tg_offset - 256 (live-LR startup)
    set lr13 -256;;
    bne lr9 lr0 step4_tg1;;
    b step4_go;;
step4_tg1:
    set lr13 -128;;
step4_go:

    # lr15: out_start - 1024 (live-LR startup)
    set lr15 -1024;;
    bne lr9 lr0 step4_out_tg1;;
    b step4_out_go;;
step4_out_tg1:
    set lr15 -512;;
step4_out_go:

    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # r_cyclic = ones (for C1 of first ch)

    # ---- Sub-loop 4A: ch = 0..127 ----
    set lr14 0;;

step4A_loop:
    # C1: x[ch,t] + aaq1
    ldr_mult_reg mem_bypass lr13 cr0;  incr lr13 256;  mult.ee mem_bypass lr0 lr12 lr0;  acc.add_aaq.first aaq1;;
    # C2-C3: aaq → TEMP_BASE+0
    aaq;;
    xmem.store_aaq_result lr0 cr5;;

    # C4: × inv_std
    ldr_cyclic_mult_reg lr0 cr5 lr0;  mult.ve.aaq lr0 lr12 lr0 aaq3;  acc.first;;
    # C5-C6: aaq → TEMP_BASE+0
    aaq;;
    xmem.store_aaq_result lr0 cr5;;

    # C7: load γ row 0
    ldr_cyclic_mult_reg lr0 cr1 lr0;;

    # C8: normalized × γ[ch]  (live lr14 = ch_idx; no LR in C8)
    ldr_mult_reg mem_bypass lr0 cr5;  mult.ev mem_bypass lr14 lr12 lr0;  acc.first;;

    # C9: load β row 0
    ldr_cyclic_mult_reg lr0 cr2 lr0;;

    # C10: ones × β[ch]  (live lr14 = ch_idx; no LR in C10)
    ldr_mult_reg mem_bypass lr0 cr4;  mult.ev mem_bypass lr14 lr12 lr0;  acc;;

    # C11: advance ch_idx and out_ptr, store row, branch
    incr lr14 1;  incr lr15 1024;  str_acc_reg lr15 cr6;  blt lr14 lr3 step4A_loop;;

    # ---- Sub-loop 4B: ch = 128..143 ----
    # lr13 and lr15 carry over; reset ch_idx for row-1 indexing
    set lr14 0;;

step4B_loop:
    # C1
    ldr_mult_reg mem_bypass lr13 cr0;  incr lr13 256;  mult.ee mem_bypass lr0 lr12 lr0;  acc.add_aaq.first aaq1;;
    # C2-C3
    aaq;;
    xmem.store_aaq_result lr0 cr5;;

    # C4
    ldr_cyclic_mult_reg lr0 cr5 lr0;  mult.ve.aaq lr0 lr12 lr0 aaq3;  acc.first;;
    # C5-C6
    aaq;;
    xmem.store_aaq_result lr0 cr5;;

    # C7: load γ row 1
    ldr_cyclic_mult_reg lr3 cr1 lr0;;

    # C8: normalized × γ[128+ch_idx]
    ldr_mult_reg mem_bypass lr0 cr5;  mult.ev mem_bypass lr14 lr12 lr0;  acc.first;;

    # C9: load β row 1
    ldr_cyclic_mult_reg lr3 cr2 lr0;;

    # C10: ones × β[128+ch_idx]
    ldr_mult_reg mem_bypass lr0 cr4;  mult.ev mem_bypass lr14 lr12 lr0;  acc;;

    # C11
    incr lr14 1;  incr lr15 1024;  str_acc_reg lr15 cr6;  blt lr14 lr4 step4B_loop;;

    # =======================================================================
    # END OF TOKEN: advance tok; restore ones for next token's step1
    # =======================================================================
    ldr_cyclic_mult_reg lr0 cr4 lr0;;

    incr lr12 1;;
    blt lr12 lr2 token_loop;;

    # Advance batch
    incr lr10 1;  incr lr11 128;;
    blt lr10 lr4 batch_loop;;

    # Advance tg
    set lr11 0;;
    incr lr9 1;;
    blt lr9 lr1 tg_loop;;

end:
    bkpt;;
