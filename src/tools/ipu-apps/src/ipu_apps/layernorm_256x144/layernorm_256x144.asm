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
#   (d) r_cyclic snap: r_cyclic is ALSO read from the start-of-cycle snapshot
#       (issue #157), so a MULT cannot consume a row that LDR_CYCLIC loads in
#       its own bundle. `;;` ends one VLIW word = one cycle = one snapshot, so a
#       load and a MULT written in the same bundle always run in the same cycle
#       regardless of textual order -- co-issuing is fine, consuming the
#       same-cycle load is not. Every step therefore primes its first row before
#       the loop and prefetches the next row from inside the body; the offset LR
#       is read LIVE by the load, so its ADD stays co-issued with the load,
#       which is what makes that load fetch the NEXT channel.
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
# Data stride between consecutive channels within one tg: 1024 B (= N_TG × 512)
# stored in data_stride (overriding ROW_STRIDE used only for output/scratch stride).

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0       = "lr0"  %}  {# const 0: cyclic_offset / xmem offset #}
{% set mask_shift     = "lr1"  %}  {# const 0: mask_shift #}
{% set read_ptr       = "lr2"  %}  {# per-step read pointer (data / centered) #}
{% set write_ptr      = "lr3"  %}  {# per-step write pointer (centered / output) #}
{% set ch_index       = "lr5"  %}  {# channel counter within a step #}
{% set ch_limit       = "lr6"  %}  {# 144 = N_CH loop bound #}
{% set data_stride    = "lr7"  %}  {# 1024 = data stride per channel (N_TG*512) #}
{% set tg_index       = "lr9"  %}  {# token-group counter (BLT-style, 1..2) #}
{% set tg_off         = "lr10" %}  {# token-group byte offset (0 or 512) #}
{% set tg_limit       = "lr11" %}  {# 2 = N_TG #}
{% set row_stride     = "lr12" %}  {# 512 = scratch/output row stride #}
{% set gamma_idx      = "lr13" %}  {# step-6 scalar index into r0 (gamma) #}
{% set beta_idx       = "lr14" %}  {# step-6 scalar index into r1 (beta) #}
{% set sub_bound      = "lr15" %}  {# step-6 sub-loop bound (128, then 16) #}

{% set DATA_BASE      = "cr0"  %}  {# 0x00000; also the const-zero source #}
{% set ONE            = "cr1"  %}  {# hardwired read-only 1 #}
{% set BETA_BASE      = "cr2"  %}  {# beta #}
{% set ONES_BASE      = "cr3"  %}  {# 128 lanes of 1.0 #}
{% set NEG_INV_N_BASE = "cr4"  %}  {# FP32(-1/144) #}
{% set INV_N_BASE     = "cr5"  %}  {# FP32(1/144) #}
{% set NEG_MEAN_BASE  = "cr6"  %}  {# -mu scratch #}
{% set CENTERED_BASE  = "cr7"  %}  {# centered / normalized scratch #}
{% set TEMP_BASE      = "cr8"  %}  {# sum-of-squares scratch #}
{% set INVSTD_BASE    = "cr9"  %}  {# 1/sigma scratch #}
{% set OUTPUT_BASE    = "cr10" %}  {# output #}
{% set GAMMA_BASE     = "cr11" %}  {# gamma #}
{% set N_CH           = "cr12" %}  {# 144 #}
{% set ROW_STRIDE     = "cr13" %}  {# 512 #}
{% set LANES          = "cr14" %}  {# 128 = valid_elements; r1 base offset #}
{% set DSTRUCT        = "cr15" %}  {# reserved dstructure register #}

    SET     {{ rc_slot0 }}  {{ DATA_BASE }};;
    SET     {{ mask_shift }}  {{ DATA_BASE }};;
    SET     {{ ch_limit }}  {{ N_CH }};;
    SET     {{ data_stride }}  {{ ROW_STRIDE }};;
    ADD     {{ data_stride }}  {{ data_stride }}  {{ data_stride }};; # {{ data_stride }} = 1024 (data stride = N_TG × 512)
    SET     {{ tg_limit }} {{ ONE }};;
    ADD     {{ tg_limit }} {{ tg_limit }} {{ ONE }};;             # {{ tg_limit }} = 2  (N_TG); built from CR1(=1)
    SET     {{ row_stride }} {{ ROW_STRIDE }};;                   # {{ row_stride }} = 512 (scratch/output stride)

# ─────────────────────────────────────────────────────────────────────────────
# Outer loop: tg = 0, 1
# tg_index  = tg counter (1..2, BLT-style)
# tg_off = tg byte offset into data (0 or 512)
# ─────────────────────────────────────────────────────────────────────────────

    SET     {{ tg_index }}  {{ DATA_BASE }};;
    ADD     {{ tg_index }}  {{ tg_index }}  {{ ONE }};;           # {{ tg_index }} = 1  (tg counter, BLT reads snap)
    SET     {{ tg_off }} {{ DATA_BASE }};;                        # {{ tg_off }} = 0  (tg byte offset)

tg_loop:

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: -μ[i] = Σ_ch  x[ch,i] × (-1/N)    for this tg
#
# Data row for (ch, tg): DATA_BASE + (ch*N_TG + tg)*512 = DATA_BASE + ch*1024 + tg*512
# read_ptr starts at tg_off - 1024 = tg*512 - 1024.
# ADD read_ptr read_ptr data_stride (=1024) fires first → live = ch*1024 + tg*512 on iteration ch.
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ NEG_INV_N_BASE }};;  # r0 ← -1/N

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ data_stride }};;   # {{ read_ptr }} = -1024
    ADD     {{ read_ptr }}  {{ read_ptr }}  {{ tg_off }};;        # {{ read_ptr }} = tg_offset - 1024
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;

    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ data_stride }};;   # prime x[ch=0]

    # Peeled first ch (ch=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD.FIRST;;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ data_stride }}; ADD {{ ch_index }} {{ ch_index }} {{ ONE }}; BLT {{ ch_index }} {{ ch_limit }} step1_loop;;
    B       step1_done;;
step1_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD;;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ data_stride }}; ADD {{ ch_index }} {{ ch_index }} {{ ONE }}; BLT {{ ch_index }} {{ ch_limit }} step1_loop;;
step1_done:

    STR_ACC_REG         {{ rc_slot0 }} {{ NEG_MEAN_BASE }};;      # NEG_MEAN_BASE = -μ

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: centered[ch,i] = x[ch,i] + (-μ[i])
#
# r0 = ONES, r1 = -μ. read_ptr = data read ptr. write_ptr = centered write ptr (-512).
# Per ch (3 cycles): same as 128x16.
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ ONES_BASE }};;
    LDR_MULT_REG        r1 {{ rc_slot0 }} {{ NEG_MEAN_BASE }};;

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ data_stride }};;
    ADD     {{ read_ptr }}  {{ read_ptr }}  {{ tg_off }};;
    SET     {{ write_ptr }}  {{ DATA_BASE }};;
    SUB     {{ write_ptr }}  {{ write_ptr }}  {{ row_stride }};;  # {{ write_ptr }} = -512 (centered stride)
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ data_stride }};;   # prime x[ch=0]
step2_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD.FIRST; LDR_CYCLIC_MULT_REG {{ rc_slot0 }} {{ ONES_BASE }} {{ rc_slot0 }};;
    MULT.RC.VV {{ rc_slot0 }} r1 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD;;
    STR_ACC_REG         {{ write_ptr }} {{ CENTERED_BASE }}; ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }}; LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ data_stride }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }}; BLT {{ ch_index }} {{ ch_limit }} step2_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Σ_ch (centered[ch,i])²    using MULT.RC.VS
#
# read_ptr = -512 (centered read ptr). Two cycles per ch.
# ─────────────────────────────────────────────────────────────────────────────

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ row_stride }};;
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;

    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;   # prime centered[ch=0]

    # Peeled first ch (ch=0): ACC.ADD.FIRST seeds r_acc.
    # MULT.RC.VS squares r_cyclic in place: centered[ch] was loaded into
    # r_cyclic a cycle earlier, and is squared in place here.
    MULT.RC.VS {{ rc_slot0 }} 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD.FIRST;;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }}; ADD {{ ch_index }} {{ ch_index }} {{ ONE }}; BLT {{ ch_index }} {{ ch_limit }} step3_loop;;
    B       step3_done;;
step3_loop:
    MULT.RC.VS {{ rc_slot0 }} 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD;;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }}; ADD {{ ch_index }} {{ ch_index }} {{ ONE }}; BLT {{ ch_index }} {{ ch_limit }} step3_loop;;
step3_done:

    STR_ACC_REG         {{ rc_slot0 }} {{ TEMP_BASE }};;

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: variance = (1/N) × Σ(x-μ)²;  1/σ = ACTIVATE rsqrt
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ TEMP_BASE }};;
    LDR_CYCLIC_MULT_REG {{ rc_slot0 }} {{ INV_N_BASE }} {{ rc_slot0 }};;   # load 1/N a cycle ahead of the MULT
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD.FIRST;;

    ACTIVATE.QUANTIZE rsqrt {{ DSTRUCT }};;
    STR_POST_AAQ_REG    {{ rc_slot0 }} {{ INVSTD_BASE }};;

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: normalized[ch,i] = centered[ch,i] × 1/σ[i]  (overwrite CENTERED)
#
# r0 = 1/σ. read_ptr = -512 (load ptr), write_ptr = 0 (store ptr, trails by a row).
# This step reads and writes the SAME rows, so once the load runs a cycle ahead
# of its MULT the two can no longer share one pointer: read_ptr points at the
# row being PREFETCHED while write_ptr points at the row being stored.
# STR_ACC_REG reads its offset LIVE, so write_ptr advances in the branch word,
# never in the store word. Three cycles per ch.
# ─────────────────────────────────────────────────────────────────────────────

    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ INVSTD_BASE }};;

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ row_stride }};;
    SET     {{ write_ptr }}  {{ DATA_BASE }};;
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;   # prime centered[ch=0]
step5_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD.FIRST;;
    STR_ACC_REG         {{ write_ptr }} {{ CENTERED_BASE }}; LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    ADD     {{ write_ptr }} {{ write_ptr }} {{ row_stride }}; ADD {{ ch_index }} {{ ch_index }} {{ ONE }}; BLT {{ ch_index }} {{ ch_limit }} step5_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: output[ch,i] = γ[ch] × normalized[ch,i] + β[ch]
#
# N_CH=144 > 128: γ/β span two 512-byte rows.
#   Row 0: γ[0..127], β[0..127]
#   Row 1: γ[128..143], β[128..143] (rest zero-padded by harness)
#
# Sub-loop A: ch=0..127   → r0=γ row0, r1=β row0, gamma_idx=fixed_idx 0..127
# Sub-loop B: ch=128..143 → reload r0=γ row1, r1=β row1, gamma_idx=0..15
#
# Output: OUTPUT_BASE + (ch*N_TG + tg)*512 = OUTPUT_BASE + ch*1024 + tg*512
# write_ptr = output write ptr, init = tg_offset - 1024.
# ADD write_ptr write_ptr data_stride fires → live = ch*1024+tg_offset on each STR cycle.
# But STR is in the same cycle as ADD write_ptr and ADD gamma_idx/beta_idx...
# To avoid incrementing write_ptr in the same cycle as gamma_idx/beta_idx, use 4 cycles per ch:
#   A: MULT.RC.VE gamma_idx × normalized[ch] (loaded last cycle); ACC.ADD.FIRST;
#      LDR_CYCLIC ONES (for B)
#   B: MULT.RC.VE beta_idx × ONES; ACC.ADD
#   C: STR output[ch]; ADD write_ptr; ADD gamma_idx; ADD beta_idx
#   D: ADD ch_index; BLT; LDR_CYCLIC normalized[ch+1]; ADD read_ptr
# Cycle C is at the 3-LR-per-word ceiling (write_ptr, gamma_idx, beta_idx), so
# the prefetch rides in cycle D instead.
# normalized[ch=0] is primed once, before sub-loop A. Sub-loop B carries
# read_ptr over from A and inherits A's final prefetch, so it must NOT re-prime:
# the chunk for its first channel is already in flight.
# ─────────────────────────────────────────────────────────────────────────────

    # ---- Sub-loop A: ch=0..127 ----
    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ GAMMA_BASE }};;      # r0 ← γ row 0
    LDR_MULT_REG        r1 {{ rc_slot0 }} {{ BETA_BASE }};;       # r1 ← β row 0

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ row_stride }};;    # normalized read ptr = -512
    SET     {{ write_ptr }}  {{ DATA_BASE }};;
    SUB     {{ write_ptr }}  {{ write_ptr }}  {{ data_stride }};; # output write ptr = -1024
    ADD     {{ write_ptr }}  {{ write_ptr }}  {{ tg_off }};;      # = tg_offset - 1024
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
    SET     {{ gamma_idx }} {{ DATA_BASE }};;                     # fixed_idx γ = 0
    SET     {{ beta_idx }} {{ LANES }};;                          # fixed_idx β = 128

    # loop bound for sub-loop A: 128 channels
    # ch_limit currently = 144; use a separate bound sub_bound=128 for sub-loop A
    SET     {{ sub_bound }} {{ LANES }};;                         # {{ sub_bound }} = 128

    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;   # prime normalized[ch=0]

step6A_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ gamma_idx }} 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD.FIRST; LDR_CYCLIC_MULT_REG {{ rc_slot0 }} {{ ONES_BASE }} {{ rc_slot0 }};;
    MULT.RC.VE {{ rc_slot0 }} {{ beta_idx }} 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD;;
    STR_ACC_REG         {{ write_ptr }} {{ OUTPUT_BASE }}; ADD {{ write_ptr }} {{ write_ptr }} {{ data_stride }}; ADD {{ gamma_idx }} {{ gamma_idx }} {{ ONE }}; ADD {{ beta_idx }} {{ beta_idx }} {{ ONE }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }}; ADD {{ ch_index }} {{ ch_index }} {{ ONE }}; BLT {{ ch_index }} {{ sub_bound }} step6A_loop;;

    # ---- Sub-loop B: ch=128..143 (16 channels) ----
    LDR_MULT_REG        r0 {{ row_stride }} {{ GAMMA_BASE }};;    # r0 ← γ row 1 (offset=512)
    LDR_MULT_REG        r1 {{ row_stride }} {{ BETA_BASE }};;     # r1 ← β row 1

    # read_ptr and write_ptr carry over from sub-loop A (already at ch=128 positions)
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
    SET     {{ gamma_idx }} {{ DATA_BASE }};;                     # fixed_idx γ = 0 (row 1 starts at lane 0)
    SET     {{ beta_idx }} {{ LANES }};;                          # fixed_idx β = 128

    # bound for sub-loop B: 16 channels (built by doubling CR1: 1→2→4→8→16)
    SET     {{ sub_bound }} {{ ONE }};;
    ADD     {{ sub_bound }} {{ sub_bound }} {{ sub_bound }};;     # 2
    ADD     {{ sub_bound }} {{ sub_bound }} {{ sub_bound }};;     # 4
    ADD     {{ sub_bound }} {{ sub_bound }} {{ sub_bound }};;     # 8
    ADD     {{ sub_bound }} {{ sub_bound }} {{ sub_bound }};;     # 16

    # No priming load here: sub-loop A's last prefetch already fetched ch=128,
    # and read_ptr carries over pointing one row past it.
step6B_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ gamma_idx }} 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD.FIRST; LDR_CYCLIC_MULT_REG {{ rc_slot0 }} {{ ONES_BASE }} {{ rc_slot0 }};;
    MULT.RC.VE {{ rc_slot0 }} {{ beta_idx }} 0 {{ mask_shift }} {{ DSTRUCT }}; ACC.ADD;;
    STR_ACC_REG         {{ write_ptr }} {{ OUTPUT_BASE }}; ADD {{ write_ptr }} {{ write_ptr }} {{ data_stride }}; ADD {{ gamma_idx }} {{ gamma_idx }} {{ ONE }}; ADD {{ beta_idx }} {{ beta_idx }} {{ ONE }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }}; ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }}; ADD {{ ch_index }} {{ ch_index }} {{ ONE }}; BLT {{ ch_index }} {{ sub_bound }} step6B_loop;;

# ─────────────────────────────────────────────────────────────────────────────
# Advance tg
# ─────────────────────────────────────────────────────────────────────────────

    ADD     {{ tg_off }} {{ tg_off }} {{ row_stride }};;          # tg_offset += 512
    ADD     {{ tg_index }}  {{ tg_index }}  {{ ONE }}; BLT {{ tg_index }}  {{ tg_limit }} tg_loop;;

end:
    BKPT;;
