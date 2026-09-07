# Multi-stream transformer projection matmul (Layer 4 OutProj, P=4 pixel-streams).
#
# Layer:   L4
# Scope:   all-stream/P4
# Layout:  unpacked
# Shape:   192ch, P4 (4-stream), K=192->N_OUT=192
# Status:  validated
# Related: proj_qkv_192_p4 / proj_ffn1_192_p4 / proj_ffn2_192_p4 are the
#          other L4 P4 projections; shape suffix 144/192/240 = L3/L4/L5
#          (see kernel_docs/kernel_layer_map.md)
# Tests:   test_proj_outproj_192_p4_wide (src/tools/ipu-apps/BUILD.bazel)
#
#   C[p, j, t] = sum_k W[j, k] * D[p, k, t]
#     p in [0,4), j in [0,N_OUT=192), k in [0,K=192), t in [0,N_TOK=64)
#
# One shared weight matrix W (output-major, no transpose) is applied
# independently to each of 4 per-stream activation blocks D[p]. This is the
# real per-transformer-layer situation: one set of learned weights, four
# pixel-streams needing the same projection in one invocation instead of 4
# host round-trips.
#
# Loop nest: STREAM OUTERMOST, then j (output channel), then a RUNTIME chunk
# loop over the K-dimension contraction, then the k-index inner loop.
#   for p in 0..3 { for j in 0..N_OUT-1 { for chunk in 0..chunk_count-1 {
#       for k_local in chunk width { ACC.ADD[.FIRST] }
#   } store C[p,j,:] } }
#
# WHY STREAM OUTERMOST (not innermost, not interleaved with j/chunk):
#   The single-stream kernels this is built from (matmul_576x192_x128 et al.)
#   already have a fully-debugged j/chunk/k timing pattern, including the
#   MULT snapshot-contract priming/biasing (issue #157, see below). Looping
#   streams OUTSIDE that whole j-loop means the entire proven inner structure
#   is reused UNCHANGED per stream -- only a per-stream base-row offset is
#   added before each j-loop, exactly like attn_v_256x36's head loop
#   (head_p_off += 512 rows per head). Interleaving the stream loop with j or
#   chunk would force re-deriving the snapshot priming/biasing in a new
#   context for no benefit, since W is identical across streams and D/C's
#   per-stream blocks are independently addressed contiguous regions -- there
#   is no data reuse to exploit by interleaving.
#
# RUNTIME CHUNK LOOP (generalizes chunk0/chunk1/chunk2 hand-unrolling):
#   K=192 needs ceil(192/128)=2 chunks: full width 128, tail width 64.
#   One chunk-loop BODY (label chunk_loop) runs chunk_count times; the harness
#   supplies chunk_count (CHUNK_COUNT) and the LAST chunk's index
#   (LAST_CHUNK_IDX = chunk_count-1). Every non-last chunk is exactly 128
#   wide (K is only ever partial on the FINAL chunk, by construction of
#   ceil-division), so there are only two possible inner-loop widths: the
#   constant FULL_BOUND=126 (width 128) and a harness-supplied TAIL_BOUND
#   (width = K - 128*(chunk_count-1), shape-specific). Each chunk iteration
#   compares its runtime chunk_idx against LAST_CHUNK_IDX and selects the
#   bound register accordingly (bound_sel: BLT chunk_idx LAST_CHUNK_IDX ->
#   full width; else tail width) -- this is what makes chunk_count and the
#   tail width purely DATA (harness-loaded registers), so the exact same
#   .asm control-flow text handles K=192/240/384/480 without new labels.
#   The weight-chunk row pointer (chunk_w_ptr = weight_row_off + chunk_idx)
#   and the data pointer (which walks continuously across ALL chunks, never
#   reset -- see the matmul_576x192_x128 header) are both chunk-count-generic
#   already; only the label structure needed generalizing.
#
# MULT SNAPSHOT CONTRACT (issue #157) -- same rule as every existing matmul
# kernel: MULT.RC.VE reads its r_cyclic DATA from the start-of-cycle
# snapshot, so it cannot consume a chunk LDR_CYCLIC_MULT_REG loads in its own
# bundle. Chunk 0 of each j primes k=0's row one bundle ahead (startup bias
# -1 -> -2); every later chunk's first row is already in flight from the
# previous chunk's trailing prefetch (data pointer never resets), so only
# chunk 0 primes/biases -- identical to the 3-chunk FFN kernels.
#
# All .asm operands are ROW numbers (issue #179), one row = 128 lanes = 512 B
# wide-FP32. Layouts (channel-major activations, one channel per WHOLE row):
#   D[p]: K rows at DATA_BASE_ROW + p*DATA_STREAM_STRIDE_ROWS + k
#         (DATA_STREAM_STRIDE_ROWS = K rows/stream)
#   W:    output-major [N_OUT, K], shared across streams, W_STRIDE_ROWS rows
#         per output channel (= ceil(K/128)), chunk c of channel j at
#         WEIGHTS_BASE_ROW + j*W_STRIDE_ROWS + c
#   C[p]: N_OUT rows at OUTPUT_BASE_ROW + p*OUT_STREAM_STRIDE_ROWS + j
#         (OUT_STREAM_STRIDE_ROWS = N_OUT rows/stream), FFN1 store applies
#         ACTIVATE.QUANTIZE silu; all other roles use identity.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing (pure source-level rename, no runtime unrolling), so the
# emitted binary is byte-identical to the raw form. NOTE: Jinja runs before
# comment stripping, so '#' comments must not contain Jinja delimiters.

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set data_ptr        = "lr0"  %}  {# data row pointer, walks continuously across chunks (+1 row/k-step) #}
{% set rc_slot0         = "lr1"  %}  {# const 0: r_cyclic write-index 0 #}
{% set k_idx            = "lr2"  %}  {# fixed_idx (MULT.RC.VE rc_idx), reset per chunk, biased on chunk 0 #}
{% set bound            = "lr3"  %}  {# selected inner-loop bound for the current chunk #}
{% set weight_row_off   = "lr4"  %}  {# j's weight-row base = j*W_STRIDE_ROWS, += W_STRIDE_ROWS per j #}
{% set chunk_w_ptr      = "lr5"  %}  {# weight_row_off + chunk_idx: this chunk's W row #}
{% set chunk_idx        = "lr6"  %}  {# runtime chunk counter 0..chunk_count-1 #}
{% set out_ptr          = "lr7"  %}  {# output row pointer for stream p, += 1 row per j #}
{% set j_idx             = "lr8"  %}  {# output-channel counter j #}
{% set stream_data_base = "lr9"  %}  {# stream p's D base row, += DATA_STREAM_STRIDE_ROWS per stream #}
{% set stream_out_base  = "lr10" %}  {# stream p's C base row, += OUT_STREAM_STRIDE_ROWS per stream #}
{% set stream_idx       = "lr11" %}  {# stream counter p 0..3 #}

{% set ZERO             = "cr0"  %}  {# const 0 #}
{% set ONE              = "cr1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE        = "cr2"  %}  {# D base row for stream 0 #}
{% set WEIGHTS_BASE     = "cr3"  %}  {# W base row (shared, all streams) #}
{% set OUTPUT_BASE      = "cr4"  %}  {# C base row for stream 0 #}
{% set NEG_ONE          = "cr5"  %}  {# -1: data-ptr startup bias AND per-chunk fixed_idx startup (same value) #}
{% set FULL_BOUND       = "cr6"  %}  {# 126: inner bound for any width-128 chunk #}
{% set TAIL_BOUND       = "cr7"  %}  {# tail chunk's inner bound (width-specific) #}
{% set W_STRIDE         = "cr8"  %}  {# W_STRIDE_ROWS: weight rows per output channel #}
{% set N_OUT_CR         = "cr9"  %}  {# N_OUT: j-loop limit #}
{% set CHUNK_COUNT      = "cr10" %}  {# ceil(K/128): chunk-loop limit #}
{% set LAST_CHUNK_IDX   = "cr11" %}  {# CHUNK_COUNT - 1 #}
{% set DATA_STREAM_STR  = "cr12" %}  {# K: D row stride between streams #}
{% set OUT_STREAM_STR   = "cr13" %}  {# N_OUT: C row stride between streams #}
{% set STREAM_COUNT     = "cr14" %}  {# 4: number of pixel-streams #}
{% set DSTRUCT          = "cr15" %}  {# reserved dstructure register #}

    SET {{ stream_idx }} {{ ZERO }};;                              # p = 0
    SET {{ stream_data_base }} {{ DATA_BASE }};;                    # stream 0 D base
    SET {{ stream_out_base }} {{ OUTPUT_BASE }};;                   # stream 0 C base

stream_loop:
    SET {{ j_idx }} {{ ZERO }};;                                    # j = 0
    SET {{ weight_row_off }} {{ ZERO }};;                           # weight_row_off = 0 (W shared, same each stream)
    ADD {{ out_ptr }} {{ stream_out_base }} {{ ZERO }};;            # out_ptr = stream C base

j_loop:
    # ---- chunk 0: prime + peel first k-iter (ACC.ADD.FIRST) ----
    ADD {{ data_ptr }} {{ stream_data_base }} {{ NEG_ONE }};;       # data ptr = stream D base - 1 row
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ ZERO }};;         # chunk0 W row = weight_row_off
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;          # r0 = W[j, chunk0]
    SET {{ k_idx }} {{ NEG_ONE }};;                                 # fixed_idx startup: -1
    SUB {{ k_idx }} {{ k_idx }} {{ ONE }};;                         # biased to -2 (load runs a bundle ahead)
    SET {{ chunk_idx }} {{ ZERO }};;                                # chunk_idx = 0

    # Prime k=0's row for chunk0 (snapshot contract, see header). data_ptr
    # already holds the ABSOLUTE row (stream_data_base includes DATA_BASE),
    # so every data load below uses base=ZERO (cr0=0).
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ FULL_BOUND }} k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ FULL_BOUND }} k_chunk0;;

after_chunk0:
    INC {{ chunk_idx }} 1;;                                         # chunk_idx = 1
    BLT {{ chunk_idx }} {{ CHUNK_COUNT }} chunk_loop;;
    B store_out;;

    # ---- chunk_loop: chunks 1 .. chunk_count-1 (runtime loop, ONE body) ----
    # No re-prime/bias here: each chunk's first row is already in flight from
    # the previous chunk's trailing prefetch (data_ptr walks continuously and
    # is never reset), exactly as in the 3- and 4-chunk single-stream kernels.
chunk_loop:
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ chunk_idx }};;    # this chunk's W row = weight_row_off + chunk_idx
    LDR_MULT_REG r0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};
    SET {{ k_idx }} {{ NEG_ONE }};;  # r0 = W[j, this chunk]; fixed_idx reset: -1 (not biased -- prefetch already in flight)

    # bound_sel: is this the LAST chunk? last -> TAIL_BOUND, else FULL_BOUND.
    BLT {{ chunk_idx }} {{ LAST_CHUNK_IDX }} use_full_bound;;
    SET {{ bound }} {{ TAIL_BOUND }};;
    B chunk_body;;
use_full_bound:
    SET {{ bound }} {{ FULL_BOUND }};;

chunk_body:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT {{ k_idx }} {{ bound }} chunk_body;;

    INC {{ chunk_idx }} 1;;
    BLT {{ chunk_idx }} {{ CHUNK_COUNT }} chunk_loop;;

store_out:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ ZERO }};;  # C[p,j,:] = R_ACC
    ADD {{ out_ptr }} {{ out_ptr }} {{ ONE }};;                     # advance output ptr, packed (1 row/j)

    ADD {{ weight_row_off }} {{ weight_row_off }} {{ W_STRIDE }};
    INC {{ j_idx }} 1;;
    BLT {{ j_idx }} {{ N_OUT_CR }} j_loop;;

    # ----- next stream: stream D/C bases += stride, p++ -----
    ADD {{ stream_data_base }} {{ stream_data_base }} {{ DATA_STREAM_STR }};;
    ADD {{ stream_out_base }} {{ stream_out_base }} {{ OUT_STREAM_STR }};;
    INC {{ stream_idx }} 1;;
    BLT {{ stream_idx }} {{ STREAM_COUNT }} stream_loop;;

end:
    BKPT;;
