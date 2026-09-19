# LayerNorm 16x240 (Layer 5): per-PE normalization across 240 channels, 16 tokens (1 tg x 16)
#
# Layer:   L5
# Scope:   single-stream
# Layout:  unpacked
# Shape:   16tok (1 tg) x 240chan
# Status:  validated
# Related: L5 port of layernorm_256x144 (L3) via layernorm_64x192 (L4);
#          feeds residual_add_16x240
# Tests:   test_layernorm_16x240_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Wide-vector debug mode (512 B per row = 128 x FP32).
# Same algorithm and VLIW patterns as layernorm_64x192. Key differences:
#   - N_CH=240, N_TOK=16, N_TG=1 -> NO outer tg loop; the whole kernel runs once
#   - One channel per XMEM row: row for ch is DATA_BASE + ch (rows, issue #179),
#     so the per-channel data stride IS the row stride (1). The 256x144
#     ancestor's separate 2-row data_stride is gone along with N_TG=2.
#   - Only lanes 0..15 carry tokens; lanes 16..127 are zero padding that rides
#     through the arithmetic. The reduction is across CHANNELS (the 240 rows),
#     independently per lane, so padding lanes simply normalize zeros and are
#     cropped by the harness teardown. Rows are never shared between channels:
#     a channel owns a whole 512 B row even though only 64 B of it is real.
#   - gamma/beta have 240 values (>128) -> reload r0/r1 at ch=128 in step 6, and
#     sub-loop B covers 112 channels (240-128).
#   - NEG_INV_N = FP32(-1/240), INV_N = FP32(1/240)  (over CHANNELS, not tokens;
#     the kernel name's 16 is the token count and is NOT the normalizer)
#
# VLIW rules (same as the ancestor):
#   (a) Startup-offset: init ptr to -stride; ADD fires first -> live=0 on first use
#   (b) Loop counter init to 1: BLT reads snapshot (pre-ADD) -> runs exactly N times
#   (c) r0/r1 snap: pre-load before loop, hold constant; MULT sees previous cycle's r0/r1
#   (d) r_cyclic snap: r_cyclic is ALSO read from the start-of-cycle snapshot
#       (issue #157), so a MULT cannot consume a row that LDR_CYCLIC loads in
#       its own bundle. ';;' ends one VLIW word = one cycle = one snapshot, so a
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

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0       = "lr0"  %}  {# const 0: cyclic_offset / xmem offset #}
{% set mask_shift     = "lr1"  %}  {# const 0: mask_shift #}
{% set read_ptr       = "lr2"  %}  {# per-step read pointer (data / centered) #}
{% set write_ptr      = "lr3"  %}  {# per-step write pointer (centered / output) #}
{% set ch_index       = "lr5"  %}  {# channel counter within a step #}
{% set ch_limit       = "lr6"  %}  {# 240 = N_CH loop bound #}
{% set row_stride     = "lr12" %}  {# 1 = row stride (data, scratch and output) #}
{% set gamma_idx      = "lr13" %}  {# step-6 scalar index into r0 (gamma) #}
{% set beta_idx       = "lr14" %}  {# step-6 scalar index into r1 (beta) #}
{% set sub_bound      = "lr15" %}  {# step-6 sub-loop bound (128, then 112) #}
{% set tmp_bound      = "lr4"  %}  {# scratch used to build the 112 bound #}

{% set DATA_BASE      = "cr0"  %}  {# row 0; also the const-zero source #}
{% set ONE            = "cr1"  %}  {# hardwired read-only 1 #}
{% set BETA_BASE      = "cr2"  %}  {# beta #}
{% set ONES_BASE      = "cr3"  %}  {# 128 lanes of 1.0 #}
{% set NEG_INV_N_BASE = "cr4"  %}  {# FP32(-1/240) #}
{% set INV_N_BASE     = "cr5"  %}  {# FP32(1/240) #}
{% set NEG_MEAN_BASE  = "cr6"  %}  {# -mu scratch #}
{% set CENTERED_BASE  = "cr7"  %}  {# centered / normalized scratch #}
{% set TEMP_BASE      = "cr8"  %}  {# sum-of-squares scratch #}
{% set INVSTD_BASE    = "cr9"  %}  {# 1/sigma scratch #}
{% set OUTPUT_BASE    = "cr10" %}  {# output #}
{% set GAMMA_BASE     = "cr11" %}  {# gamma #}
{% set N_CH           = "cr12" %}  {# 240 #}
{% set ROW_STRIDE     = "cr13" %}  {# 1 row #}
{% set LANES          = "cr14" %}  {# 128 = sub-loop A bound; r1 base offset #}
{% set DSTRUCT        = "cr15" %}  {# reserved dstructure register #}

    SET     {{ rc_slot0 }}  {{ DATA_BASE }};;
    SET     {{ mask_shift }}  {{ DATA_BASE }};;
    SET     {{ ch_limit }}  {{ N_CH }};;
    SET     {{ row_stride }} {{ ROW_STRIDE }};;                   # {{ row_stride }} = 1 row (data/scratch/output stride; N_TG=1)

# -----------------------------------------------------------------------------
# Step 1: -mu[i] = sum_ch  x[ch,i] * (-1/N_CH)
#
# Data row for ch: DATA_BASE + ch  (N_TG=1, one channel per row).
# read_ptr starts at -1 row; ADD read_ptr read_ptr row_stride fires first
# -> live = ch on iteration ch.
# -----------------------------------------------------------------------------

    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ NEG_INV_N_BASE }};;  # r0 <- -1/N_CH

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ row_stride }};;    # {{ read_ptr }} = -1
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;

    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime x[ch=0]

    # Peeled first ch (ch=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD.FIRST;;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT {{ ch_index }} {{ ch_limit }} step1_loop;;
    B       step1_done;;
step1_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD;;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT {{ ch_index }} {{ ch_limit }} step1_loop;;
step1_done:

    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ rc_slot0 }} {{ NEG_MEAN_BASE }};;  # NEG_MEAN_BASE = -mu

# -----------------------------------------------------------------------------
# Step 2: centered[ch,i] = x[ch,i] + (-mu[i])
#
# r0 = ONES, r1 = -mu. read_ptr = data read ptr. write_ptr = centered write ptr (-1).
# Three cycles per ch.
# -----------------------------------------------------------------------------

    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ ONES_BASE }};;
    LDR_MULT_REG        r1 {{ rc_slot0 }} {{ NEG_MEAN_BASE }};;

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ row_stride }};;
    SET     {{ write_ptr }}  {{ DATA_BASE }};;
    SUB     {{ write_ptr }}  {{ write_ptr }}  {{ row_stride }};;  # {{ write_ptr }} = -1 (centered stride)
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime x[ch=0]
step2_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ rc_slot0 }} {{ ONES_BASE }} {{ rc_slot0 }};;
    MULT.RC.VV {{ rc_slot0 }} r1 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ CENTERED_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ DATA_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};
    BLT {{ ch_index }} {{ ch_limit }} step2_loop;;

# -----------------------------------------------------------------------------
# Step 3: sum_ch (centered[ch,i])^2    using MULT.RC.VS
#
# read_ptr = -1 (centered read ptr). Two cycles per ch.
# -----------------------------------------------------------------------------

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ row_stride }};;
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;

    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime centered[ch=0]

    # Peeled first ch (ch=0): ACC.ADD.FIRST seeds r_acc.
    # MULT.RC.VS squares r_cyclic in place: centered[ch] was loaded into
    # r_cyclic a cycle earlier, and is squared in place here.
    MULT.RC.VS {{ rc_slot0 }} 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD.FIRST;;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT {{ ch_index }} {{ ch_limit }} step3_loop;;
    B       step3_done;;
step3_loop:
    MULT.RC.VS {{ rc_slot0 }} 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD;;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT {{ ch_index }} {{ ch_limit }} step3_loop;;
step3_done:

    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ rc_slot0 }} {{ TEMP_BASE }};;

# -----------------------------------------------------------------------------
# Step 4: variance = (1/N_CH) * sum(x-mu)^2;  1/sigma = ACTIVATE rsqrt
#
# The padding lanes reach rsqrt with a variance of exactly 0. The rsqrt
# activation maps a non-positive input to 0.0, so those lanes carry a harmless
# zero forward instead of an infinity; they are cropped in teardown anyway.
# -----------------------------------------------------------------------------

    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ TEMP_BASE }};;
    LDR_CYCLIC_MULT_REG {{ rc_slot0 }} {{ INV_N_BASE }} {{ rc_slot0 }};;   # load 1/N a cycle ahead of the MULT
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD.FIRST;;

    ACTIVATE.QUANTIZE rsqrt {{ DSTRUCT }};;
    STR_POST_AAQ_REG    {{ rc_slot0 }} {{ INVSTD_BASE }};;

# -----------------------------------------------------------------------------
# Step 5: normalized[ch,i] = centered[ch,i] * 1/sigma[i]  (overwrite CENTERED)
#
# r0 = 1/sigma. read_ptr = -1 (load ptr), write_ptr = 0 (store ptr, trails by a row).
# This step reads and writes the SAME rows, so once the load runs a cycle ahead
# of its MULT the two can no longer share one pointer: read_ptr points at the
# row being PREFETCHED while write_ptr points at the row being stored.
# STR_POST_AAQ_REG reads its offset LIVE, so write_ptr advances in the branch
# word, never in the store word. Three cycles per ch.
# -----------------------------------------------------------------------------

    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ INVSTD_BASE }};;

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ row_stride }};;
    SET     {{ write_ptr }}  {{ DATA_BASE }};;
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime centered[ch=0]
step5_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD.FIRST;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ CENTERED_BASE }};
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;
    ADD     {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT {{ ch_index }} {{ ch_limit }} step5_loop;;

# -----------------------------------------------------------------------------
# Step 6: output[ch,i] = gamma[ch] * normalized[ch,i] + beta[ch]
#
# N_CH=240 > 128: gamma/beta span two rows.
#   Row 0: gamma[0..127],   beta[0..127]
#   Row 1: gamma[128..239], beta[128..239] (rest zero-padded by harness)
#
# Sub-loop A: ch=0..127   -> r0=gamma row0, r1=beta row0, gamma_idx=fixed_idx 0..127
# Sub-loop B: ch=128..239 -> reload r0=gamma row1, r1=beta row1, gamma_idx=0..111
#
# Output row for ch: OUTPUT_BASE + ch (N_TG=1). write_ptr init = -1 row.
# 4 cycles per ch:
#   A: MULT.RC.VE gamma_idx x normalized[ch] (loaded last cycle); ACC.ADD.FIRST;
#      LDR_CYCLIC ONES (for B)
#   B: MULT.RC.VE beta_idx x ONES; ACC.ADD
#   C: STR output[ch]; ADD write_ptr; ADD gamma_idx; ADD beta_idx
#   D: ADD ch_index; BLT; LDR_CYCLIC normalized[ch+1]; ADD read_ptr
# Cycle C is at the 3-LR-per-word ceiling (write_ptr, gamma_idx, beta_idx), so
# the prefetch rides in cycle D instead.
# normalized[ch=0] is primed once, before sub-loop A. Sub-loop B carries
# read_ptr over from A and inherits A's final prefetch, so it must NOT re-prime:
# the chunk for its first channel is already in flight.
# -----------------------------------------------------------------------------

    # ---- Sub-loop A: ch=0..127 ----
    LDR_MULT_REG        r0 {{ rc_slot0 }} {{ GAMMA_BASE }};;      # r0 <- gamma row 0
    LDR_MULT_REG        r1 {{ rc_slot0 }} {{ BETA_BASE }};;       # r1 <- beta row 0

    SET     {{ read_ptr }}  {{ DATA_BASE }};;
    SUB     {{ read_ptr }}  {{ read_ptr }}  {{ row_stride }};;    # normalized read ptr = -1
    SET     {{ write_ptr }}  {{ DATA_BASE }};;
    SUB     {{ write_ptr }}  {{ write_ptr }}  {{ row_stride }};;  # output write ptr = -1
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
    SET     {{ gamma_idx }} {{ DATA_BASE }};;                     # fixed_idx gamma = 0
    SET     {{ beta_idx }} {{ LANES }};;                          # fixed_idx beta = 128

    # loop bound for sub-loop A: 128 channels
    # ch_limit currently = 240; use a separate bound sub_bound=128 for sub-loop A
    SET     {{ sub_bound }} {{ LANES }};;                         # {{ sub_bound }} = 128

    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};;  # prime normalized[ch=0]

step6A_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ gamma_idx }} 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ rc_slot0 }} {{ ONES_BASE }} {{ rc_slot0 }};;
    MULT.RC.VE {{ rc_slot0 }} {{ beta_idx }} 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ OUTPUT_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    ADD {{ gamma_idx }} {{ gamma_idx }} {{ ONE }};
    ADD {{ beta_idx }} {{ beta_idx }} {{ ONE }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT {{ ch_index }} {{ sub_bound }} step6A_loop;;

    # ---- Sub-loop B: ch=128..239 (112 channels) ----
    LDR_MULT_REG        r0 {{ row_stride }} {{ GAMMA_BASE }};;    # r0 <- gamma row 1 (offset = 1 row)
    LDR_MULT_REG        r1 {{ row_stride }} {{ BETA_BASE }};;     # r1 <- beta row 1

    # read_ptr and write_ptr carry over from sub-loop A (already at ch=128 positions)
    SET     {{ ch_index }}  {{ DATA_BASE }};;
    ADD     {{ ch_index }}  {{ ch_index }}  {{ ONE }};;
    SET     {{ gamma_idx }} {{ DATA_BASE }};;                     # fixed_idx gamma = 0 (row 1 starts at lane 0)
    SET     {{ beta_idx }} {{ LANES }};;                          # fixed_idx beta = 128

    # bound for sub-loop B: 112 channels (240-128). CR15 is the reserved
    # dstructure register, so the bound is built from CRs instead of taking one:
    # 16 comes from doubling CR1 four times, then 112 = CR14(128) - 16.
    SET     {{ tmp_bound }} {{ ONE }};;
    ADD     {{ tmp_bound }} {{ tmp_bound }} {{ tmp_bound }};;     # 2
    ADD     {{ tmp_bound }} {{ tmp_bound }} {{ tmp_bound }};;     # 4
    ADD     {{ tmp_bound }} {{ tmp_bound }} {{ tmp_bound }};;     # 8
    ADD     {{ tmp_bound }} {{ tmp_bound }} {{ tmp_bound }};;     # 16
    SET     {{ sub_bound }} {{ LANES }};;                         # 128
    SUB     {{ sub_bound }} {{ sub_bound }} {{ tmp_bound }};;     # 112

    # No priming load here: sub-loop A's last prefetch already fetched ch=128,
    # and read_ptr carries over pointing one row past it.
step6B_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ gamma_idx }} 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ rc_slot0 }} {{ ONES_BASE }} {{ rc_slot0 }};;
    MULT.RC.VE {{ rc_slot0 }} {{ beta_idx }} 0 {{ mask_shift }} {{ DSTRUCT }};
    ACC.ADD;;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ write_ptr }} {{ OUTPUT_BASE }};
    ADD {{ write_ptr }} {{ write_ptr }} {{ row_stride }};
    ADD {{ gamma_idx }} {{ gamma_idx }} {{ ONE }};
    ADD {{ beta_idx }} {{ beta_idx }} {{ ONE }};;
    LDR_CYCLIC_MULT_REG {{ read_ptr }} {{ CENTERED_BASE }} {{ rc_slot0 }};
    ADD {{ read_ptr }} {{ read_ptr }} {{ row_stride }};
    ADD {{ ch_index }} {{ ch_index }} {{ ONE }};
    BLT {{ ch_index }} {{ sub_bound }} step6B_loop;;

end:
    BKPT;;
