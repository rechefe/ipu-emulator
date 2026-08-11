# Multi-stream transformer projection matmul (Layer 3 QKV, P=4 pixel-streams).
#
#   C[p, tg, j, t] = sum_k W[j, k] * D[p, k, tg, t]
#     p in [0,4), j in [0,N_OUT    = 432), k in [0,K        = 144), tg in [0,N_TG=2), t in [0,128)
#
# One shared weight matrix W (output-major, no transpose) is applied
# independently to each of 4 per-stream activation blocks D[p]. Same
# per-layer property as every proj_*_p4 kernel: one set of learned weights,
# four pixel-streams needing the same projection in one invocation instead of
# 4 host round-trips.
#
# Loop nest: STREAM OUTERMOST, then j (output channel), then TWO
# HAND-DUPLICATED token-group blocks (tg=0, tg=1 -- see "WHY HAND-DUPLICATED
# TG" below), each with a RUNTIME chunk loop over the K-dimension contraction.
#   for p in 0..3 { for j in 0..N_OUT-1 {
#       tg=0: for chunk in 0..chunk_count-1 { for k_local in chunk width { ACC.ADD[.FIRST] } } store C[p,0,j,:]
#       tg=1: for chunk in 0..chunk_count-1 { for k_local in chunk width { ACC.ADD[.FIRST] } } store C[p,1,j,:]
#   } }
#
# WHY STREAM OUTERMOST: identical rationale to every other proj_*_p4 kernel
# (proj_qkv_192_p4 et al.) -- the single-stream ancestor (matmul_144x432_x128)
# already has a fully-debugged tg/chunk/k timing pattern including the MULT
# snapshot-contract priming/biasing (issue #157). Looping streams OUTSIDE the
# whole j-loop reuses that proven inner structure UNCHANGED per stream; only a
# per-stream base-row offset is added before each j-loop. W is identical
# across streams and D/C's per-stream blocks are independently addressed
# contiguous regions, so there is no data reuse to exploit by interleaving.
#
# WHY HAND-DUPLICATED TG (not a runtime tg loop): the single-stream ancestor
# (matmul_144x432_x128) hand-duplicates tg=0/tg=1 blocks rather than looping,
# because L3's D layout INTERLEAVES tg within each channel's row-pair (row
# 2k+tg, see the layout note below) -- a real tg loop would need its own
# runtime-selected data-pointer stride/base logic on top of the already
# runtime chunk loop, for only 2 iterations. Two hand-duplicated blocks (one
# extra level versus L4/L5's stream -> j -> chunk nest) keeps the addressing
# exactly as simple as the proven ancestor while still generalizing the K
# dimension (the harder, shape-varying axis) into ONE runtime chunk-loop body.
# This is the one real structural difference from L4/L5 (N_TG=1): L3's N_TG=2
# adds this extra fixed-2-way split inside the j-loop, ABOVE the stream loop
# structure which is unchanged from L4/L5.
#
# RUNTIME CHUNK LOOP (generalizes chunk0/chunk1 hand-unrolling in the
# ancestor): K        = 144 needs ceil(144/128)=2 chunks: full width 128, tail width
# 16. One chunk-loop BODY (label chunk_loop) runs chunk_count times per tg;
# the harness supplies chunk_count (CHUNK_COUNT) and the LAST chunk's index
# (LAST_CHUNK_IDX = chunk_count-1). Every non-last chunk is exactly 128 wide,
# so there are only two possible inner-loop widths: the constant
# FULL_BOUND=126 (width 128) and a harness-supplied TAIL_BOUND (width = K -
# 128*(chunk_count-1), shape-specific). Each chunk iteration compares its
# runtime chunk_idx against LAST_CHUNK_IDX and selects the bound register
# accordingly (bound_sel: BLT chunk_idx LAST_CHUNK_IDX -> full width; else
# tail width) -- this is what makes chunk_count and the tail width purely DATA
# (harness-loaded registers), so the exact same .asm control-flow text will
# handle K=288 (proj_ffn2_144_p4, 3 chunks) without new labels, same as this
# body being reused unchanged from the L4/L5 CHUNK_COUNT=2/3/4 family.
#
# MULT SNAPSHOT CONTRACT (issue #157) -- same rule as every existing matmul
# kernel: MULT.RC.VE reads its r_cyclic DATA from the start-of-cycle
# snapshot, so it cannot consume a chunk LDR_CYCLIC_MULT_REG loads in its own
# bundle. Chunk 0 of each (j, tg) primes k=0's row one bundle ahead (startup
# bias -1 -> -2); every later chunk's first row is already in flight from the
# previous chunk's trailing prefetch (data pointer never resets within a tg
# block), so only chunk 0 primes/biases -- identical to the ancestor and to
# every 3-/4-chunk FFN kernel in the L4/L5 family.
#
# All .asm operands are ROW numbers (issue #179), one row = 128 lanes = 512 B
# wide-FP32. Layouts (channel-major activations, one channel per WHOLE row):
#   D[p]: interleaved channel-major [K rows, N_TG, 128 tokens] within the
#         stream's block -- row (k, tg) at
#         DATA_BASE_ROW + p*DATA_STREAM_STRIDE_ROWS + k*N_TG + tg
#         (DATA_STREAM_STRIDE_ROWS = K*N_TG rows/stream), mirroring the
#         single-stream ancestor's D row (k, tg) at DATA_BASE + k*256 + tg*128
#         exactly (256 B = 2 rows, 128 B = half a row -> in row units: k*2+tg).
#   W:    output-major [N_OUT, K], shared across streams, W_STRIDE_ROWS rows
#         per output channel (= ceil(K/128)), chunk c of channel j at
#         WEIGHTS_BASE_ROW + j*W_STRIDE_ROWS + c
#   C[p]: grouped channel-major [N_TG, N_OUT, 128 tokens] per stream -- row
#         (j, tg) at OUTPUT_BASE_ROW + p*OUT_STREAM_STRIDE_ROWS +
#         tg*N_OUT + j (OUT_STREAM_STRIDE_ROWS = N_TG*N_OUT rows/stream),
#         mirroring the ancestor's grouped-by-tg output block layout.
#         All roles use identity except FFN1 (silu applied at the store).
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing (pure source-level rename, no runtime unrolling), so the
# emitted binary is byte-identical to the raw form. NOTE: Jinja runs before
# comment stripping, so '#' comments must not contain Jinja delimiters.

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set data_ptr        = "lr0"  %}  {# data row pointer for the active tg, walks by tg_stride per k-step #}
{% set rc_slot0         = "lr1"  %}  {# const 0: r_cyclic write-index 0 #}
{% set k_idx            = "lr2"  %}  {# fixed_idx (MULT.RC.VE rc_idx), reset per chunk, biased on chunk 0 #}
{% set bound            = "lr3"  %}  {# selected inner-loop bound for the current chunk #}
{% set weight_row_off   = "lr4"  %}  {# j's weight-row base = j*W_STRIDE_ROWS, PERSISTENT across the whole j_loop body -- reset only at stream_loop top (once per stream, alongside j_idx), advanced ONCE per j at the bottom of j_loop. Must NOT be reset inside tg=0/tg=1 blocks. #}
{% set chunk_w_ptr      = "lr5"  %}  {# weight_row_off + chunk_idx: this chunk's W row #}
{% set chunk_idx        = "lr6"  %}  {# runtime chunk counter 0..chunk_count-1 #}
{% set out_ptr          = "lr7"  %}  {# output row pointer for stream p, tg=0 block, += 1 row per j #}
{% set out_ptr_tg1      = "lr8"  %}  {# output row pointer for stream p, tg=1 block, += 1 row per j #}
{% set j_idx            = "lr9"  %}  {# output-channel counter j #}
{% set stream_data_base = "lr10" %}  {# stream p's D base row, += DATA_STREAM_STRIDE_ROWS per stream #}
{% set stream_idx       = "lr11" %}  {# stream counter p 0..3 #}
{% set tg0_data_start   = "lr12" %}  {# stream_data_base - N_TG: tg=0 startup skew (k=0 primed below) #}
{% set tg1_data_start   = "lr13" %}  {# stream_data_base + 1 - N_TG: tg=1 startup skew #}
{% set stream_out_base  = "lr14" %}  {# stream p's C base row (tg=0 sub-block), += OUT_STREAM_STRIDE_ROWS per stream #}
{% set tg_stride        = "lr15" %}  {# N_TG=2: per-k-step data_ptr stride within a tg block (set once, outside hot loops) #}

{% set ZERO             = "cr0"  %}  {# const 0 #}
{% set ONE              = "cr1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE        = "cr2"  %}  {# D base row for stream 0 #}
{% set WEIGHTS_BASE     = "cr3"  %}  {# W base row (shared, all streams) #}
{% set OUTPUT_BASE      = "cr4"  %}  {# C base row for stream 0, tg=0 sub-block #}
{% set NEG_ONE          = "cr5"  %}  {# -1: fixed_idx startup #}
{% set FULL_BOUND       = "cr6"  %}  {# 126: inner bound for any width-128 chunk #}
{% set TAIL_BOUND       = "cr7"  %}  {# tail chunk's inner bound (width-specific) #}
{% set W_STRIDE         = "cr8"  %}  {# W_STRIDE_ROWS: weight rows per output channel #}
{% set N_OUT_CR         = "cr9"  %}  {# N_OUT: j-loop limit AND tg1 output sub-block offset #}
{% set CHUNK_COUNT      = "cr10" %}  {# ceil(K/128): chunk-loop limit #}
{% set LAST_CHUNK_IDX   = "cr11" %}  {# CHUNK_COUNT - 1 #}
{% set DATA_STREAM_STR  = "cr12" %}  {# K*N_TG: D row stride between streams #}
{% set OUT_STREAM_STR   = "cr13" %}  {# N_TG*N_OUT: C row stride between streams #}
{% set STREAM_COUNT     = "cr14" %}  {# 4: number of pixel-streams #}
{% set DSTRUCT          = "cr15" %}  {# reserved dstructure register #}

    SET {{ stream_idx }} {{ ZERO }};;                               # p = 0
    SET {{ stream_data_base }} {{ DATA_BASE }};;                     # stream 0 D base
    SET {{ stream_out_base }} {{ OUTPUT_BASE }};;                    # stream 0 C base (tg=0 sub-block)
    SET {{ tg_stride }} {{ ONE }};;                                  # tg_stride = 1
    ADD {{ tg_stride }} {{ tg_stride }} {{ ONE }};;                  # tg_stride = 2 (N_TG); set ONCE, not per-bundle

stream_loop:
    SET {{ j_idx }} {{ ZERO }};;                                     # j = 0
    SET {{ weight_row_off }} {{ ZERO }};;                            # j=0's weight base = 0 (reset once per stream)
    ADD {{ out_ptr }} {{ stream_out_base }} {{ ZERO }};;             # out_ptr (tg=0) = stream C base
    ADD {{ out_ptr_tg1 }} {{ stream_out_base }} {{ N_OUT_CR }};;     # out_ptr (tg=1) = stream C base + N_OUT

j_loop:
    # weight_row_off already holds THIS j's weight base (persistent -- see
    # register-name block; NOT reset here, unlike the single-stream ancestor
    # which re-derives its w_ptr from a separately-tracked running pointer of
    # its own). Advanced once per j at the bottom of this loop.

    # ---- tg=0: chunk 0 (prime + peel first k-iter, ACC.ADD.FIRST) ----------
    SUB {{ tg0_data_start }} {{ stream_data_base }} {{ ONE }};;      # tg0 data_ptr start = stream_data_base - 1 (row 2k+0 stride)
    SUB {{ tg0_data_start }} {{ tg0_data_start }} {{ ONE }};;        # -= 1 more: -2 total (N_TG=2 rows/channel bias)
    ADD {{ data_ptr }} {{ tg0_data_start }} {{ ZERO }};;             # data_ptr = tg0 startup
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ ZERO }};;          # chunk0 W row = weight_row_off
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;           # r0 = W[j, chunk0]
    SET {{ k_idx }} {{ NEG_ONE }};;                                  # fixed_idx startup: -1
    SUB {{ k_idx }} {{ k_idx }} {{ ONE }};;                          # biased to -2 (load runs a bundle ahead)
    SET {{ chunk_idx }} {{ ZERO }};;                                 # chunk_idx = 0

    # Prime k=0's row for tg0 chunk0 (snapshot contract, see header). data_ptr
    # already holds the ABSOLUTE row, so loads below use base=ZERO (cr0=0).
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }}; ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }}; ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ FULL_BOUND }} tg0_k_chunk0;;
    B tg0_after_chunk0;;

tg0_k_chunk0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }}; ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ FULL_BOUND }} tg0_k_chunk0;;

tg0_after_chunk0:
    INC {{ chunk_idx }} 1;;                                          # chunk_idx = 1
    BLT {{ chunk_idx }} {{ CHUNK_COUNT }} tg0_chunk_loop;;
    B tg0_store;;

    # ---- tg=0 chunk_loop: chunks 1 .. chunk_count-1 (runtime loop) ---------
    # No re-prime/bias here: each chunk's first row is already in flight from
    # the previous chunk's trailing prefetch (data_ptr walks continuously and
    # is never reset), exactly as in the L4/L5 proj_*_p4 family.
tg0_chunk_loop:
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ chunk_idx }};;     # this chunk's W row = weight_row_off + chunk_idx
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }}; SET {{ k_idx }} {{ NEG_ONE }};;   # r0 = W[j, this chunk]; fixed_idx reset: -1 (not biased)

    BLT {{ chunk_idx }} {{ LAST_CHUNK_IDX }} tg0_use_full_bound;;
    SET {{ bound }} {{ TAIL_BOUND }};;
    B tg0_chunk_body;;
tg0_use_full_bound:
    SET {{ bound }} {{ FULL_BOUND }};;

tg0_chunk_body:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }}; ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ bound }} tg0_chunk_body;;

    INC {{ chunk_idx }} 1;;
    BLT {{ chunk_idx }} {{ CHUNK_COUNT }} tg0_chunk_loop;;

tg0_store:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }}; STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;   # C[p,tg=0,j,:] = R_ACC

    # ---- tg=1: chunk 0 (prime + peel first k-iter, ACC.ADD.FIRST) ----------
    # weight_row_off is UNCHANGED here -- tg=1 re-reads the SAME j's weights
    # as tg=0 just did (W has no tg axis), so it must NOT be reset.
    ADD {{ tg1_data_start }} {{ stream_data_base }} {{ ZERO }};;     # tg1 data_ptr start = stream_data_base + 1 - N_TG
    SUB {{ tg1_data_start }} {{ tg1_data_start }} {{ ONE }};;        # -1
    ADD {{ data_ptr }} {{ tg1_data_start }} {{ ZERO }};;             # data_ptr = tg1 startup
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ ZERO }};;          # chunk0 W row = weight_row_off
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;           # r0 = W[j, chunk0]
    SET {{ k_idx }} {{ NEG_ONE }};;
    SUB {{ k_idx }} {{ k_idx }} {{ ONE }};;                          # biased to -2
    SET {{ chunk_idx }} {{ ZERO }};;

    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }}; ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;

    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }}; ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ FULL_BOUND }} tg1_k_chunk0;;
    B tg1_after_chunk0;;

tg1_k_chunk0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }}; ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ FULL_BOUND }} tg1_k_chunk0;;

tg1_after_chunk0:
    INC {{ chunk_idx }} 1;;
    BLT {{ chunk_idx }} {{ CHUNK_COUNT }} tg1_chunk_loop;;
    B tg1_store;;

tg1_chunk_loop:
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ chunk_idx }};;
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }}; SET {{ k_idx }} {{ NEG_ONE }};;

    BLT {{ chunk_idx }} {{ LAST_CHUNK_IDX }} tg1_use_full_bound;;
    SET {{ bound }} {{ TAIL_BOUND }};;
    B tg1_chunk_body;;
tg1_use_full_bound:
    SET {{ bound }} {{ FULL_BOUND }};;

tg1_chunk_body:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }}; ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ bound }} tg1_chunk_body;;

    INC {{ chunk_idx }} 1;;
    BLT {{ chunk_idx }} {{ CHUNK_COUNT }} tg1_chunk_loop;;

tg1_store:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }}; STR_POST_AAQ_REG {{ out_ptr_tg1 }} {{ ZERO }};;   # C[p,tg=1,j,:] = R_ACC

    ADD {{ out_ptr }} {{ out_ptr }} {{ ONE }};;                      # advance tg=0 output ptr, packed (1 row/j)
    ADD {{ out_ptr_tg1 }} {{ out_ptr_tg1 }} {{ ONE }};;              # advance tg=1 output ptr, packed (1 row/j)

    ADD {{ weight_row_off }} {{ weight_row_off }} {{ W_STRIDE }};;   # advance to next j's weight base (post both tg blocks)
    INC {{ j_idx }} 1;;
    BLT {{ j_idx }} {{ N_OUT_CR }} j_loop;;

    # ----- next stream: stream D/C bases += stride, p++ -----
    ADD {{ stream_data_base }} {{ stream_data_base }} {{ DATA_STREAM_STR }};;
    ADD {{ stream_out_base }} {{ stream_out_base }} {{ OUT_STREAM_STR }};;
    INC {{ stream_idx }} 1;;
    BLT {{ stream_idx }} {{ STREAM_COUNT }} stream_loop;;

end:
    BKPT;;
