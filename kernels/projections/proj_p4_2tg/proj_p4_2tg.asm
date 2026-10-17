{#- proj_p4_2tg.asm -- multi-stream transformer projection, two token groups (P=4), identity.

    y[p, tg, j, t] = sum_k a[j, k] * b[p, tg, k, t]
      p in [0, 4), tg in [0, 2), j in [0, n_out), k in [0, k), t in [0, n_tok)

    The template of the L3 (144) QKV, OutProj and FFN2 entries in registry.toml. L3
    has 256 tokens per stream, two token groups of 128 (n_tg = 2). One set of
    learned weights a (output-major [n_out, k], stored verbatim, no
    transpose) is applied to each of the 4 pixel-streams' data blocks b[p]
    (channel-major [2, k, n_tok]) in one invocation, instead of 4 host
    round-trips through the single-stream matmul_144x*_x128 kernels it is
    built from. The store applies ACTIVATE.QUANTIZE identity
    (activation = none); proj_p4_2tg_silu is the same body with silu.

    Design notes

    - All operands are ROW numbers (issue #179); one row is 128 lanes. The
      harness places and packs the regions (kernel.toml):
        b: interleaved channel-major, row (p, k, tg) at
           b + p*data_stream_str + k*n_tg + tg (data_stream_str = k*n_tg
           rows per stream; pack channel_major), mirroring the single-stream
           ancestor's row (k, tg);
        a: w_stride = ceil(k / 128) rows per output channel, chunk c of
           channel j at a + j*w_stride + c, the last chunk zero-padded
           (pack output_major_weights);
        y: grouped channel-major, row (p, tg, j) at
           y + p*out_stream_str + tg*n_out + j (out_stream_str = n_tg*n_out
           rows per stream; pack projection_output), mirroring the
           ancestor's grouped-by-tg output block.

    - Loop nest: STREAM OUTERMOST, then j (output channel), then TWO
      HAND-DUPLICATED token-group blocks (tg=0, tg=1), each with a RUNTIME
      chunk loop over the K contraction.
        for p in 0..3 { for j in 0..n_out-1 {
            tg=0: for chunk in 0..chunk_count-1 { for k_local in chunk width { ACC.ADD[.FIRST] } } store y[p,0,j,:]
            tg=1: for chunk in 0..chunk_count-1 { for k_local in chunk width { ACC.ADD[.FIRST] } } store y[p,1,j,:]
        } }

    - Why stream outermost: as in proj_p4, the single-stream ancestor
      (matmul_144x432_x128 et al.) already has a fully debugged tg/chunk/k
      timing pattern, including the MULT snapshot-contract priming/biasing
      (issue #157). Looping streams OUTSIDE the whole j-loop reuses that
      proven inner structure unchanged per stream; only a per-stream
      base-row offset is added before each j-loop. a is the same for every
      stream and b/y's per-stream blocks are independently addressed
      contiguous regions, so there is no data reuse to exploit by
      interleaving.

    - Why hand-duplicated tg (not a runtime tg loop): the ancestor
      hand-duplicates the tg=0/tg=1 blocks rather than looping, because b
      INTERLEAVES tg within each channel's row pair (row 2k+tg). A real tg
      loop would need its own runtime-selected data-pointer stride and base
      logic on top of the runtime chunk loop, for only 2 iterations. Two
      hand-duplicated blocks keep the addressing as simple as the proven
      ancestor while still generalizing the K dimension (the harder,
      shape-varying axis) into ONE runtime chunk-loop body. This is the one
      structural difference from proj_p4 (n_tg = 1), and why the two-group
      bodies have their own LR allocation: merging them with proj_p4 would
      change the binaries.

    - Runtime chunk loop: K needs chunk_count = ceil(k / 128) chunks (2 for
      k = 144, 3 for k = 288). One chunk-loop body runs chunk_count times per
      tg; [derived] supplies chunk_count and last_chunk_idx = chunk_count - 1.
      Every non-last chunk is exactly 128 wide, so there are only two
      inner-loop widths: the constant full_bound = 126 (width 128) and
      tail_bound = width - 2 for the tail width k - 128*(chunk_count - 1).
      Each chunk iteration compares its runtime chunk_idx with
      last_chunk_idx and selects the bound register (bound_sel), which makes
      chunk_count and the tail width pure DATA (CRs). Chunk 0 is always a
      full 128-wide chunk, hence k >= 128.

    - MULT snapshot contract (issue #157): MULT.RC.VE reads its R_CYCLIC
      data from the start-of-cycle snapshot, so it cannot consume a row that
      LD.MULT.CYCLIC loads in its own bundle. Chunk 0 of each (j, tg) primes
      k=0's row one bundle ahead (startup bias -1 -> -2); every later chunk's
      first row is already in flight from the previous chunk's trailing
      prefetch (the data pointer never resets within a tg block), so only
      chunk 0 primes and biases.

    - The registers are named below ({% set %}), a pure source-level rename:
      the binary is the same as with the raw register names. The CR names are
      the [abi] bindings ({{ cr.<name> }}); CR15 holds the harness's default
      dstructure (128 valid elements).
-#}

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set data_ptr        = "LR0"  %}  {# data row pointer for the active tg, walks by tg_stride per k-step #}
{% set rc_slot0         = lr.rc_slot0 %}  {# const 0: r_cyclic write-index 0 #}
{% set k_idx            = "LR2"  %}  {# fixed_idx (MULT.RC.VE rc_idx), reset per chunk, biased on chunk 0 #}
{% set bound            = "LR3"  %}  {# selected inner-loop bound for the current chunk #}
{% set weight_row_off   = "LR4"  %}  {# j's weight-row base = j*W_STRIDE_ROWS, PERSISTENT across the whole j_loop body -- reset only at stream_loop top (once per stream, alongside j_idx), advanced ONCE per j at the bottom of j_loop. Must NOT be reset inside tg=0/tg=1 blocks. #}
{% set chunk_w_ptr      = "LR5"  %}  {# weight_row_off + chunk_idx: this chunk's W row #}
{% set chunk_idx        = "LR6"  %}  {# runtime chunk counter 0..chunk_count-1 #}
{% set out_ptr          = "LR7"  %}  {# output row pointer for stream p, tg=0 block, += 1 row per j #}
{% set out_ptr_tg1      = "LR8"  %}  {# output row pointer for stream p, tg=1 block, += 1 row per j #}
{% set j_idx            = "LR9"  %}  {# output-channel counter j #}
{% set stream_data_base = "LR10" %}  {# stream p's D base row, += DATA_STREAM_STRIDE_ROWS per stream #}
{% set stream_idx       = "LR11" %}  {# stream counter p 0..3 #}
{% set tg0_data_start   = "LR12" %}  {# stream_data_base - N_TG: tg=0 startup skew (k=0 primed below) #}
{% set tg1_data_start   = "LR13" %}  {# stream_data_base + 1 - N_TG: tg=1 startup skew #}
{% set stream_out_base  = "LR14" %}  {# stream p's C base row (tg=0 sub-block), += OUT_STREAM_STRIDE_ROWS per stream #}
{% set tg_stride        = "LR15" %}  {# N_TG=2: per-k-step data_ptr stride within a tg block (set once, outside hot loops) #}

{% set ZERO             = "CR0"  %}  {# const 0 #}
{% set ONE              = "CR1"  %}  {# hardwired read-only 1 #}
{% set DATA_BASE        = cr.b  %}  {# D base row for stream 0 #}
{% set WEIGHTS_BASE     = cr.a  %}  {# W base row (shared, all streams) #}
{% set OUTPUT_BASE      = cr.y  %}  {# C base row for stream 0, tg=0 sub-block #}
{% set NEG_ONE          = cr.neg_one  %}  {# -1: fixed_idx startup #}
{% set FULL_BOUND       = cr.full_bound  %}  {# 126: inner bound for any width-128 chunk #}
{% set TAIL_BOUND       = cr.tail_bound  %}  {# tail chunk's inner bound (width-specific) #}
{% set W_STRIDE         = cr.w_stride  %}  {# W_STRIDE_ROWS: weight rows per output channel #}
{% set N_OUT_CR         = cr.n_out  %}  {# N_OUT: j-loop limit AND tg1 output sub-block offset #}
{% set CHUNK_COUNT      = cr.chunk_count %}  {# ceil(K/128): chunk-loop limit #}
{% set LAST_CHUNK_IDX   = cr.last_chunk_idx %}  {# CHUNK_COUNT - 1 #}
{% set DATA_STREAM_STR  = cr.data_stream_str %}  {# K*N_TG: D row stride between streams #}
{% set OUT_STREAM_STR   = cr.out_stream_str %}  {# N_TG*N_OUT: C row stride between streams #}
{% set STREAM_COUNT     = cr.stream_count %}  {# 4: number of pixel-streams #}
{% set DSTRUCT          = "CR15" %}  {# reserved dstructure register #}

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
    LD.MULT R0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;           # R0 = W[j, chunk0]
    SET {{ k_idx }} {{ NEG_ONE }};;                                  # fixed_idx startup: -1
    SUB {{ k_idx }} {{ k_idx }} {{ ONE }};;                          # biased to -2 (load runs a bundle ahead)
    SET {{ chunk_idx }} {{ ZERO }};;                                 # chunk_idx = 0

    # Prime k=0's row for tg0 chunk0 (snapshot contract, see header). data_ptr
    # already holds the ABSOLUTE row, so loads below use base=ZERO (CR0=0).
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ FULL_BOUND }} tg0_k_chunk0;;
    B tg0_after_chunk0;;

tg0_k_chunk0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ FULL_BOUND }} tg0_k_chunk0;;

tg0_after_chunk0:
    INC {{ chunk_idx }} 1;;                                          # chunk_idx = 1
    BLT.S {{ chunk_idx }} {{ CHUNK_COUNT }} tg0_chunk_loop;;
    B tg0_store;;

    # ---- tg=0 chunk_loop: chunks 1 .. chunk_count-1 (runtime loop) ---------
    # No re-prime/bias here: each chunk's first row is already in flight from
    # the previous chunk's trailing prefetch (data_ptr walks continuously and
    # is never reset), exactly as in the L4/L5 proj_*_p4 family.
tg0_chunk_loop:
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ chunk_idx }};;     # this chunk's W row = weight_row_off + chunk_idx
    LD.MULT R0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};
    SET {{ k_idx }} {{ NEG_ONE }};;  # R0 = W[j, this chunk]; fixed_idx reset: -1 (not biased)

    BLT.S {{ chunk_idx }} {{ LAST_CHUNK_IDX }} tg0_use_full_bound;;
    SET {{ bound }} {{ TAIL_BOUND }};;
    B tg0_chunk_body;;
tg0_use_full_bound:
    SET {{ bound }} {{ FULL_BOUND }};;

tg0_chunk_body:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ bound }} tg0_chunk_body;;

    INC {{ chunk_idx }} 1;;
    BLT.S {{ chunk_idx }} {{ CHUNK_COUNT }} tg0_chunk_loop;;

tg0_store:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ out_ptr }} {{ ZERO }};;  # C[p,tg=0,j,:] = R_ACC

    # ---- tg=1: chunk 0 (prime + peel first k-iter, ACC.ADD.FIRST) ----------
    # weight_row_off is UNCHANGED here -- tg=1 re-reads the SAME j's weights
    # as tg=0 just did (W has no tg axis), so it must NOT be reset.
    ADD {{ tg1_data_start }} {{ stream_data_base }} {{ ZERO }};;     # tg1 data_ptr start = stream_data_base + 1 - N_TG
    SUB {{ tg1_data_start }} {{ tg1_data_start }} {{ ONE }};;        # -1
    ADD {{ data_ptr }} {{ tg1_data_start }} {{ ZERO }};;             # data_ptr = tg1 startup
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ ZERO }};;          # chunk0 W row = weight_row_off
    LD.MULT R0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};;           # R0 = W[j, chunk0]
    SET {{ k_idx }} {{ NEG_ONE }};;
    SUB {{ k_idx }} {{ k_idx }} {{ ONE }};;                          # biased to -2
    SET {{ chunk_idx }} {{ ZERO }};;

    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};;

    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ FULL_BOUND }} tg1_k_chunk0;;
    B tg1_after_chunk0;;

tg1_k_chunk0:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ FULL_BOUND }} tg1_k_chunk0;;

tg1_after_chunk0:
    INC {{ chunk_idx }} 1;;
    BLT.S {{ chunk_idx }} {{ CHUNK_COUNT }} tg1_chunk_loop;;
    B tg1_store;;

tg1_chunk_loop:
    ADD {{ chunk_w_ptr }} {{ weight_row_off }} {{ chunk_idx }};;
    LD.MULT R0 {{ chunk_w_ptr }} {{ WEIGHTS_BASE }};
    SET {{ k_idx }} {{ NEG_ONE }};;

    BLT.S {{ chunk_idx }} {{ LAST_CHUNK_IDX }} tg1_use_full_bound;;
    SET {{ bound }} {{ TAIL_BOUND }};;
    B tg1_chunk_body;;
tg1_use_full_bound:
    SET {{ bound }} {{ FULL_BOUND }};;

tg1_chunk_body:
    MULT.RC.VE {{ rc_slot0 }} {{ k_idx }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ data_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ data_ptr }} {{ data_ptr }} {{ tg_stride }};
    ADD {{ k_idx }} {{ k_idx }} {{ ONE }};
    BLT.S {{ k_idx }} {{ bound }} tg1_chunk_body;;

    INC {{ chunk_idx }} 1;;
    BLT.S {{ chunk_idx }} {{ CHUNK_COUNT }} tg1_chunk_loop;;

tg1_store:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ out_ptr_tg1 }} {{ ZERO }};;  # C[p,tg=1,j,:] = R_ACC

    ADD {{ out_ptr }} {{ out_ptr }} {{ ONE }};;                      # advance tg=0 output ptr, packed (1 row/j)
    ADD {{ out_ptr_tg1 }} {{ out_ptr_tg1 }} {{ ONE }};;              # advance tg=1 output ptr, packed (1 row/j)

    ADD {{ weight_row_off }} {{ weight_row_off }} {{ W_STRIDE }};;   # advance to next j's weight base (post both tg blocks)
    INC {{ j_idx }} 1;;
    BLT.S {{ j_idx }} {{ N_OUT_CR }} j_loop;;

    # ----- next stream: stream D/C bases += stride, p++ -----
    ADD {{ stream_data_base }} {{ stream_data_base }} {{ DATA_STREAM_STR }};;
    ADD {{ stream_out_base }} {{ stream_out_base }} {{ OUT_STREAM_STR }};;
    INC {{ stream_idx }} 1;;
    BLT.S {{ stream_idx }} {{ STREAM_COUNT }} stream_loop;;

end:
    HALT;;
