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
#   lr3  = 128   γ/β row-1 offset; also γ/β boundary threshold (4A limit)
#   lr4  = 16    4B limit (ch=128..143)
#   lr5  = 256   data stride per channel
#   lr6  = 143   (unused now, kept for consistency)
#   lr7  = 144   inner loop limit (N_CH)
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
# PIPELINE TIMING:
#   Slot order in emulator: LR fires first, then XMEM/MULT/ACC/COND from snapshot.
#   "snapshot" reads (XMEM offset, MULT operands, COND): value from start of cycle
#     (before LR of same cycle).
#   Startup: set ptr to first_addr; LR incr fires before XMEM reads snapshot → first
#     XMEM read uses the pre-incr value = first_addr. ✓
#
# OUTPUT LAYOUT (interleaved tg):
#   Row (ch*2 + tg) at OUTPUT_BASE + (ch*2+tg) * 512 bytes.
#   str_acc_reg uses snapshot of lr15:
#     tg=0: ch=0→offset 0, ch=1→offset 1024, ..., ch=k→offset k*1024
#     tg=1: ch=0→offset 512, ch=1→offset 1536, ..., ch=k→offset 512+k*1024
#   Teardown reads contiguously at 512-byte stride → correct interleaved layout.

# ===========================================================================
# ONE-TIME INITIALIZATION
# ===========================================================================

    set lr0 0;;
    set lr1 2;;
    set lr2 8;;
    set lr3 128;;
    set lr4 16;;
    set lr5 256;;
    set lr6 143;;
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
    # STEP 1: Σx and mean
    # Accumulate masked lane t of x[ch,t] × 1.0 for ch=0..143.
    # r_cyclic[0..127] = ones throughout.
    # lr13 = tg*128 (exact start for ch=0); snapshot used by XMEM.
    # Counter lr14 (0..143); blt sees snapshot so runs 0..143 + 1 extra on 0
    # read (harmless zero-read at ch=144).
    # =========================================================================

    reset_acc;;
    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # r_cyclic = ONES

    set lr13 0;;
    bne lr9 lr0 step1_tg1;;
    b step1_go;;
step1_tg1:
    set lr13 128;;
step1_go:
    set lr14 0;;

step1_loop:
    ldr_mult_reg mem_bypass lr13 cr0;  incr lr13 256;  mult.ee mem_bypass lr0 lr12 lr0;  acc;  incr lr14 1;  blt lr14 lr7 step1_loop;;

    agg sum value cr0 aaq0;;   # aaq0 = Σx (raw int32 sum)
    agg sum value cr0 aaq1;;   # aaq1 = Σx (same value, used in step 4A)

    # =========================================================================
    # STEP 2: Σx² and E[x²]
    # Two-cycle body per channel:
    #   Cycle A: ldr_cyclic_mult_reg x[ch,tg] (lr13 = exact address, snapshot)
    #   Cycle B: ldr_mult_reg x[ch,tg] (same snapshot addr) + mult.ee (x²) + acc
    #            + incr lr13 256 + incr lr14 1 + blt
    # lr13 reset to exact tg*128 (snapshot used by both cycles).
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
    ldr_cyclic_mult_reg lr13 cr0 lr0;;  # r_cyclic = x[ch,tg] (snapshot lr13)

    ldr_mult_reg mem_bypass lr13 cr0;  incr lr13 256;  mult.ee mem_bypass lr0 lr12 lr0;  acc;  incr lr14 1;  blt lr14 lr7 step2_loop_A;;

    agg sum value cr0 aaq2;;   # aaq2 = Σx² (raw int32 sum)

    # =========================================================================
    # STEP 3: variance = E[x²] - mean²,  inv_std = 1/sqrt(var)
    #
    # 3a: broadcast mean scalar to lane t → TEMP_BASE+0 (truncation #1)
    # 3b: compute -mean = mean × (-1.0) → TEMP_BASE+128 (truncation #2)
    # 3c: compute -mean² = mean × (-mean) → r_acc (masked, acc.first)
    # 3d: add E[x²] via mult.ve.aaq aaq2 → var = E[x²] - mean²
    #     agg sum inv_sqrt → aaq3 = inv_std
    # =========================================================================

    # 3a: mean → TEMP_BASE+0
    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # r_cyclic = ones
    mult.ve.aaq lr0 lr12 lr0 aaq0;  acc.first;;
    # TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
    aaq;;
    xmem.store_aaq_result lr0 cr5;;     # TEMP_BASE+0 = mean (INT8 garbage)

    # 3b: -mean → TEMP_BASE+128
    ldr_cyclic_mult_reg lr0 cr7 lr0;;   # r_cyclic = neg_ones (FP8 -1.0)
    ldr_mult_reg mem_bypass lr0 cr5;  mult.ee mem_bypass lr0 lr12 lr0;  acc.first;;
    # TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
    aaq;;
    xmem.store_aaq_result lr3 cr5;;     # TEMP_BASE+128 = -mean; lr3=128

    # 3c: -mean² = mean × (-mean), masked to lane t
    ldr_cyclic_mult_reg lr3 cr5 lr0;;   # r_cyclic = TEMP_BASE+128 (-mean); lr3=128
    ldr_mult_reg mem_bypass lr0 cr5;  mult.ee mem_bypass lr0 lr12 lr0;  acc.first;;
    # r_acc[t] = -mean²  (garbage from INT8 truncation)

    # 3d: + E[x²] → var; then inv_std
    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # r_cyclic = ones
    mult.ve.aaq lr0 lr12 lr0 aaq2;  acc;;
    # r_acc[t] = E[x²] - mean²  (garbage at mean² term)

    agg sum inv_sqrt cr0 aaq3;;         # aaq3 = inv_std (FP32 bits)

    # =========================================================================
    # STEP 4: Apply affine per channel
    # out[ch,t] = γ[ch] × (x[ch,t]-mean) × inv_std + β[ch]
    #
    # Per channel (11 cycles):
    #  C1:  ldr x[ch]; incr lr13 256; mult.ee masked; acc.add_aaq.first aaq1
    #       → r_acc[t] = x[ch,t] + aaq1  (= x - mean in truncated sense)
    #  C2:  aaq  // TODO(fp8_aaq)
    #  C3:  xmem.store_aaq_result → TEMP_BASE+0  (truncation #3)
    #  C4:  ldr_cyclic TEMP_BASE; mult.ve.aaq aaq3; acc.first
    #       → r_acc[t] = normalized (FP8(inv_std) × FP8(x-mean))
    #  C5:  aaq  // TODO(fp8_aaq)
    #  C6:  xmem.store_aaq_result → TEMP_BASE+0  (truncation #4)
    #  C7:  ldr_cyclic γ row
    #  C8:  ldr TEMP_BASE; mult.ev γ[ch_idx]; acc.first
    #       → r_acc[t] = normalized × γ[ch]
    #  C9:  ldr_cyclic β row
    # C10:  ldr ones; mult.ev β[ch_idx]; acc
    #       → r_acc[t] += β[ch]
    # C11:  incr lr14 1; incr lr15 lr8; str_acc_reg lr15 cr6; blt lr14 limit loop
    #       str_acc_reg uses snapshot lr15 → writes correct row offset
    #       blt uses snapshot lr14 → runs for snapshots 0..limit-1
    #
    # No dummy iteration: lr14=0, lr13=data_start, lr15=output_start (both tg variants).
    # str_acc_reg snapshot for ch=k: lr15 = output_start + k*1024.
    #   tg=0: start=0,     ch=k → offset k*1024 → row 2k   ✓
    #   tg=1: start=512,   ch=k → offset 512+k*1024 → row 2k+1 ✓
    # mult.ev uses snapshot lr14 = ch_idx = k ✓
    #
    # Sub-loop 4A: ch=0..127, γ/β row 0 (cr1/cr2), limit lr3=128.
    # Sub-loop 4B: ch=128..143, γ/β row 1 at offset lr3=128, limit lr4=16.
    # lr13 and lr15 continue from 4A into 4B without reset.
    # =========================================================================

    # Initialize data and output ptrs for step4
    set lr13 0;;
    bne lr9 lr0 step4_tg1;;
    b step4_go;;
step4_tg1:
    set lr13 128;;
step4_go:
    set lr15 0;;
    bne lr9 lr0 step4_out_tg1;;
    b step4_out_go;;
step4_out_tg1:
    set lr15 512;;
step4_out_go:

    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # r_cyclic = ones (for Phase A of first ch)

    # ---- Sub-loop 4A: ch = 0..127 ----
    set lr14 0;;

step4A_loop:
    # C1: x[ch,t] + aaq1 (= x - mean)
    ldr_mult_reg mem_bypass lr13 cr0;  incr lr13 256;  mult.ee mem_bypass lr0 lr12 lr0;  acc.add_aaq.first aaq1;;
    # C2-C3: aaq → store (x-mean) // TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
    aaq;;
    xmem.store_aaq_result lr0 cr5;;

    # C4: × inv_std (scalar from aaq3 low byte)
    ldr_cyclic_mult_reg lr0 cr5 lr0;  mult.ve.aaq lr0 lr12 lr0 aaq3;  acc.first;;
    # C5-C6: aaq → store normalized // TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
    aaq;;
    xmem.store_aaq_result lr0 cr5;;

    # C7: load γ row 0
    ldr_cyclic_mult_reg lr0 cr1 lr0;;

    # C8: normalized × γ[ch_idx]  (snapshot lr14 = ch_idx = k)
    ldr_mult_reg mem_bypass lr0 cr5;  mult.ev mem_bypass lr14 lr12 lr0;  acc.first;;

    # C9: load β row 0
    ldr_cyclic_mult_reg lr0 cr2 lr0;;

    # C10: ones × β[ch_idx]; acc
    ldr_mult_reg mem_bypass lr0 cr4;  mult.ev mem_bypass lr14 lr12 lr0;  acc;;

    # C11: advance ptrs, store row, branch  (lr3=128: runs for snapshots 0..127)
    incr lr14 1;  incr lr15 1024;  str_acc_reg lr15 cr6;  blt lr14 lr3 step4A_loop;;

    # ---- Sub-loop 4B: ch = 128..143 ----
    # lr13 and lr15 continue from 4A; reset ch_idx to 0 for row-1 indexing
    set lr14 0;;

step4B_loop:
    # C1: x[ch,t] + aaq1 (= x - mean)
    ldr_mult_reg mem_bypass lr13 cr0;  incr lr13 256;  mult.ee mem_bypass lr0 lr12 lr0;  acc.add_aaq.first aaq1;;
    # C2-C3: aaq → store (x-mean) // TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
    aaq;;
    xmem.store_aaq_result lr0 cr5;;

    # C4: × inv_std
    ldr_cyclic_mult_reg lr0 cr5 lr0;  mult.ve.aaq lr0 lr12 lr0 aaq3;  acc.first;;
    # C5-C6: aaq → store normalized // TODO(fp8_aaq): replace aaq with aaq fp8_e4m3
    aaq;;
    xmem.store_aaq_result lr0 cr5;;

    # C7: load γ row 1 (GAMMA_BASE + 128)
    ldr_cyclic_mult_reg lr3 cr1 lr0;;   # lr3=128

    # C8: normalized × γ[128+ch_idx]  (snapshot lr14 = row-1 ch_idx = 0..15)
    ldr_mult_reg mem_bypass lr0 cr5;  mult.ev mem_bypass lr14 lr12 lr0;  acc.first;;

    # C9: load β row 1 (BETA_BASE + 128)
    ldr_cyclic_mult_reg lr3 cr2 lr0;;   # lr3=128

    # C10: ones × β[128+ch_idx]; acc
    ldr_mult_reg mem_bypass lr0 cr4;  mult.ev mem_bypass lr14 lr12 lr0;  acc;;

    # C11: advance, store, branch  (lr4=16: runs for snapshots 0..15)
    incr lr14 1;  incr lr15 1024;  str_acc_reg lr15 cr6;  blt lr14 lr4 step4B_loop;;

    # =======================================================================
    # END OF TOKEN: advance tok counter; restore ones in r_cyclic for next token
    # =======================================================================
    ldr_cyclic_mult_reg lr0 cr4 lr0;;   # restore ones in r_cyclic for next token step1

    incr lr12 1;;
    blt lr12 lr2 token_loop;;       # lr2=8; runs for snapshots 0..7 (8 tokens/batch)

    # Advance batch: increment batch counter and mask block offset
    incr lr10 1;  incr lr11 128;;
    blt lr10 lr4 batch_loop;;       # lr4=16; runs for batches 0..15

    # Advance tg: reset mask offset, increment tg counter
    set lr11 0;;
    incr lr9 1;;
    blt lr9 lr1 tg_loop;;           # lr1=2; runs for tg=0,1

end:
    bkpt;;
