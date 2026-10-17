{#- proj_p4.asm -- multi-stream transformer projection, one token group (P=4).

    y[p, 0, j, t] = act(sum_k a[j, k] * b[p, 0, k, t])
      p in [0, 4), j in [0, n_out), k in [0, k), t in [0, n_tok)

    The template of the L4 (192) and L5 (240) proj_*_p4 entries in
    registry.toml: QKV, OutProj, FFN1 and FFN2 of one MobileViT transformer
    layer. One set of learned weights a (output-major [n_out, k], stored
    verbatim, no transpose) is applied to each of the 4 pixel-streams' data
    blocks b[p] (channel-major [k, n_tok]) in one invocation, instead of 4
    host round-trips through the single-stream matmul_*_x128 kernels it is
    built from. act is the activation parameter, applied by ACTIVATE.QUANTIZE
    at the store: none (identity; QKV, OutProj, FFN2) or silu (FFN1). The
    activation is part of the routing query: a plain projection query must
    never reach a kernel that fuses silu.

    Design notes

    - All operands are ROW numbers (issue #179); one row is 128 lanes. The
      harness places and packs the regions (kernel.toml):
        b: one channel per row, row (p, k) at b + p*data_stream_str + k
           (data_stream_str = k rows per stream), n_tok lanes valid
           (pack channel_major);
        a: w_stride = ceil(k / 128) rows per output channel, chunk c of
           channel j at a + j*w_stride + c, the last chunk zero-padded
           (pack output_major_weights);
        y: row (p, j) at y + p*out_stream_str + j (out_stream_str = n_out
           rows per stream), the first n_tok lanes valid
           (pack projection_output).

    - Loop nest: STREAM OUTERMOST, then j (output channel), then a RUNTIME
      chunk loop over the K contraction, then the k-index inner loop.
        for p in 0..3 { for j in 0..n_out-1 { for chunk in 0..chunk_count-1 {
            for k_local in chunk width { ACC.ADD[.FIRST] }
        } store y[p, 0, j, :] } }

    - Why stream outermost (not innermost, not interleaved with j/chunk):
      the single-stream kernels this is built from (matmul_576x192_x128 et
      al.) already have a fully debugged j/chunk/k timing pattern, including
      the MULT snapshot-contract priming/biasing (issue #157, below). Looping
      streams OUTSIDE the whole j-loop reuses that proven inner structure
      unchanged per stream: only a per-stream base-row offset is added before
      each j-loop, exactly like attn_v_256x36's head loop. Interleaving the
      stream loop with j or chunk would force re-deriving the snapshot
      priming/biasing in a new context for no benefit: a is the same for
      every stream and b/y's per-stream blocks are independently addressed
      contiguous regions, so there is no data reuse to exploit.

    - Runtime chunk loop (generalizes chunk0/chunk1/chunk2 hand-unrolling):
      K needs chunk_count = ceil(k / 128) chunks. One chunk-loop body (label
      chunk_loop) runs chunk_count times; [derived] supplies chunk_count and
      the last chunk's index (last_chunk_idx = chunk_count - 1). Every
      non-last chunk is exactly 128 wide (K is only ever partial on the FINAL
      chunk, by construction of ceil-division), so there are only two
      inner-loop widths: the constant full_bound = 126 (width 128) and
      tail_bound = width - 2 for the tail width k - 128*(chunk_count - 1).
      Each chunk iteration compares its runtime chunk_idx with
      last_chunk_idx and selects the bound register (bound_sel: BLT.S
      chunk_idx last_chunk_idx -> full width, else tail width). That makes
      chunk_count and the tail width pure DATA (CRs), so the same
      control-flow text handles K = 192, 240, 384 and 480 without new
      labels. The weight-chunk row pointer (chunk_w_ptr = weight_row_off +
      chunk_idx) and the data pointer (which walks continuously across ALL
      chunks and is never reset; see the matmul_576x192_x128 header) are
      chunk-count generic already; only the label structure needed
      generalizing. Chunk 0 is always a full 128-wide chunk, hence k >= 128.

    - MULT snapshot contract (issue #157), the rule of every matmul kernel:
      MULT.RC.VE reads its R_CYCLIC data from the start-of-cycle snapshot,
      so it cannot consume a row that LD.MULT.CYCLIC loads in its own
      bundle. Chunk 0 of each j primes k=0's row one bundle ahead (startup
      bias -1 -> -2); every later chunk's first row is already in flight from
      the previous chunk's trailing prefetch (the data pointer never
      resets), so only chunk 0 primes and biases.

    - The registers are named below ({% set %}), a pure source-level rename:
      the binary is the same as with the raw register names. The CR names are
      the [abi] bindings ({{ cr.<name> }}); CR15 holds the harness's default
      dstructure (128 valid elements).
-#}

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set data_ptr        = "LR0"  %}  {# data row pointer, walks continuously across chunks (+1 row/k-step) #}
{% set rc_slot0         = "LR1"  %}  {# const 0: r_cyclic write-index 0 #}
{% set k_idx            = "LR2"  %}  {# fixed_idx (MULT.RC.VE rc_idx), reset per chunk, biased on chunk 0 #}
{% set bound            = "LR3"  %}  {# selected inner-loop bound for the current chunk #}
{% set weight_row_off   = "LR4"  %}  {# j's weight-row base = j*W_STRIDE_ROWS, += W_STRIDE_ROWS per j #}
{% set chunk_w_ptr      = "LR5"  %}  {# weight_row_off + chunk_idx: this chunk's W row #}
{% set chunk_idx        = "LR6"  %}  {# runtime chunk counter 0..chunk_count-1 #}
{% set out_ptr          = "LR7"  %}  {# output row pointer for stream p, += 1 row per j #}
{% set j_idx             = "LR8"  %}  {# output-channel counter j #}
{% set stream_data_base = "LR9"  %}  {# stream p's D base row, += DATA_STREAM_STRIDE_ROWS per stream #}
{% set stream_out_base  = "LR10" %}  {# stream p's C base row, += OUT_STREAM_STRIDE_ROWS per stream #}
{% set stream_idx       = "LR11" %}  {# stream counter p 0..3 #}

{% set ZERO             = "CR0"  %}  {# const 0 #}
{% set ONE              = "CR1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE        = cr.b  %}  {# D base row for stream 0 #}
{% set WEIGHTS_BASE     = cr.a  %}  {# W base row (shared, all streams) #}
{% set OUTPUT_BASE      = cr.y  %}  {# C base row for stream 0 #}
{% set NEG_ONE          = cr.neg_one  %}  {# -1: data-ptr startup bias AND per-chunk fixed_idx startup (same value) #}
{% set FULL_BOUND       = cr.full_bound  %}  {# 126: inner bound for any width-128 chunk #}
{% set TAIL_BOUND       = cr.tail_bound  %}  {# tail chunk's inner bound (width-specific) #}
{% set W_STRIDE         = cr.w_stride  %}  {# W_STRIDE_ROWS: weight rows per output channel #}
{% set N_OUT_CR         = cr.n_out  %}  {# N_OUT: j-loop limit #}
{% set CHUNK_COUNT      = cr.chunk_count %}  {# ceil(K/128): chunk-loop limit #}
{% set LAST_CHUNK_IDX   = cr.last_chunk_idx %}  {# CHUNK_COUNT - 1 #}
{% set DATA_STREAM_STR  = cr.data_stream_str %}  {# K: D row stride between streams #}
{% set OUT_STREAM_STR   = cr.out_stream_str %}  {# N_OUT: C row stride between streams #}
{% set STREAM_COUNT     = cr.stream_count %}  {# 4: number of pixel-streams #}
{% set DSTRUCT          = "CR15" %}  {# reserved dstructure register #}

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
    LD.MULT R0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;          # R0 = W[j, chunk0]
    SET {{ k_idx }} {{ NEG_ONE }};;                                 # fixed_idx startup: -1
    SUB {{ k_idx }} {{ k_idx }} {{ ONE }};;                         # biased to -2 (load runs a bundle ahead)
    SET {{ chunk_idx }} {{ ZERO }};;                                # chunk_idx = 0

    # Prime k=0's row for chunk0 (snapshot contract, see header). data_ptr
    # already holds the ABSOLUTE row (stream_data_base includes DATA_BASE),
    # so every data load below uses base=ZERO (CR0=0).
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ FULL_BOUND }} k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ FULL_BOUND }} k_chunk0;;

after_chunk0:
    INC {{ chunk_idx }} 1;;                                         # chunk_idx = 1
    BLT.S {{ chunk_idx }} {{ CHUNK_COUNT }} chunk_loop;;
    B store_out;;

    # ---- chunk_loop: chunks 1 .. chunk_count-1 (runtime loop, ONE body) ----
    # No re-prime/bias here: each chunk's first row is already in flight from
    # the previous chunk's trailing prefetch (data_ptr walks continuously and
    # is never reset), exactly as in the 3- and 4-chunk single-stream kernels.
chunk_loop:
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ chunk_idx }};;    # this chunk's W row = weight_row_off + chunk_idx
    LD.MULT R0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};
    SET {{ k_idx }} {{ NEG_ONE }};;  # R0 = W[j, this chunk]; fixed_idx reset: -1 (not biased -- prefetch already in flight)

    # bound_sel: is this the LAST chunk? last -> TAIL_BOUND, else FULL_BOUND.
    BLT.S {{ chunk_idx }} {{ LAST_CHUNK_IDX }} use_full_bound;;
    SET {{ bound }} {{ TAIL_BOUND }};;
    B chunk_body;;
use_full_bound:
    SET {{ bound }} {{ FULL_BOUND }};;

chunk_body:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ ONE }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ bound }} chunk_body;;

    INC {{ chunk_idx }} 1;;
    BLT.S {{ chunk_idx }} {{ CHUNK_COUNT }} chunk_loop;;

store_out:
    ACTIVATE.QUANTIZE {{ "identity" if activation == "none" else activation }} {{ DSTRUCT }};
    ST.AAQ {{ out_ptr }} {{ ZERO }};;  # y[p,0,j,:] = act(R_ACC)
    ADD {{ out_ptr }} {{ out_ptr }} {{ ONE }};;                     # advance output ptr, packed (1 row/j)

    ADD {{ weight_row_off }} {{ weight_row_off }} {{ W_STRIDE }};
    INC {{ j_idx }} 1;;
    BLT.S {{ j_idx }} {{ N_OUT_CR }} j_loop;;

    # ----- next stream: stream D/C bases += stride, p++ -----
    ADD {{ stream_data_base }} {{ stream_data_base }} {{ DATA_STREAM_STR }};;
    ADD {{ stream_out_base }} {{ stream_out_base }} {{ OUT_STREAM_STR }};;
    INC {{ stream_idx }} 1;;
    BLT.S {{ stream_idx }} {{ STREAM_COUNT }} stream_loop;;

end:
    HALT;;
