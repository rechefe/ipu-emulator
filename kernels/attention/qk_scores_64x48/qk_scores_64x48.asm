{#- qk_scores_64x48.asm -- QK^T scores, query-major (Layer 4), per (stream, head) block (FP32)

    Layer:   L4
    Scope:   single-stream
    Layout:  unpacked
    Shape:   64tok x 48chan (head_dim), P=4 streams x 4 heads (16 blocks)
    Related: L4 variant of qk_scores_256x36 (L3); qk_scores_16x60 is the L5 variant;
             feeds attn_v_64x48 (query-major chain); attn_scores_km_64x48 is
             the key-major sibling for the same layer
    Tests:   //kernels/attention/qk_scores_64x48; the family's test_datapath.py holds it
             bit for bit to the ACC datapath mirror

      S[b, i, s] = sum_c Q[b, i, c] * K[b, s, c]   contraction over head_dim c = 0..47

    Layer 4: d = 192, N = 64 tokens/stream, P = 4 streams, h = 4 heads,
    head_dim D = 48, so there are P*h = 16 independent (stream, head) blocks and
    each is a 64-query x 64-key x 48-channel score problem.

    This is the L4 variant of qk_scores_256x36. The mapping is carried over
    unchanged -- broadcast template, no AGG, no MULT.RC.VV score form:
      K channel-major: K[b, s, c] on row (b*48 + c), keys 0..63 in the first 64
        lanes of the row (N = 64 <= LANES = 128, so a channel column is ONE row
        and there is exactly ONE key group -- the 2-group split of the L3 kernel
        collapses away).
      Q staged query-major by the harness (gather of the strided channels):
        QROW[b, i] = Q[b, i, 0..47] contiguous on row (b*64 + i). This lets R0
        hold one query's 48 channels with a single LD.MULT, exactly the
        matmul broadcast template (scalar from R0 indexed by c).

    Broadcast template (mirrors qk_scores_256x36, width 48):
      per query i:  R0 = QROW[b, i]           (48 scalars Q[b, i, 0..47])
      per channel:  R_CYCLIC = K[b, 0..63, c]     (64 keys' channel-c column)
                    MULT.RC.VE: scalar Q[i,c] (= R0[c]) x vector R_CYCLIC
                      -> MULT_RES[s] = Q[i,c] * K[s,c]
                    ACC.ADD.FIRST (c=0) / ACC.ADD      -> R_ACC[s] += Q[i,c] * K[s,c]
      after 48 channels R_ACC[s] = S[b, i, s] for the 64 keys of the block.
      ACTIVATE.QUANTIZE identity + ST.AAQ
                    -> one whole row per query (the leading 64 FP32 lanes are the
                       scores; the harness unpacks the first n_tok lanes -- rows are
                       never shared between queries).

    head_dim = 48 needs no padding: it is the contraction LOOP COUNT (bound
    LR6 = D-2 = 46), not a lane count. Lanes here are keys.

    MULT CYCLE-START CONTRACT (issue #157): MULT.RC.VE reads its R_CYCLIC DATA from
    the cycle-start value while keeping the LR index live, so it cannot
    consume a K column loaded by LD.MULT.CYCLIC in its own bundle -- it
    would see the previous column. `;;` ends one instruction word = one cycle = one
    cycle start, so a load and a MULT in the same bundle always run in the same
    cycle regardless of textual order; co-issuing is fine, consuming the
    same-cycle load is not.
      Each query therefore primes c=0's column, then every loop body multiplies
      the column loaded last cycle while prefetching the next one.
      k_ptr (the load offset) is read LIVE by the load, so its ADD stays
      co-issued with the load it feeds -- and because the load now runs one
      bundle ahead, k_ptr naturally addresses the NEXT channel's column.
      chan_index (the R0 scalar selector) is read LIVE by MULT and must name the
      channel just loaded, so its ADD moves into a c_loop_pre block that falls
      through into the loop body. That changes what BLT.S sees: with the ADD in the
      same word, BLT.S read the pre-ADD cycle start; with the ADD one word earlier it
      reads the already-incremented value and would exit a channel early. The
      loop bound is therefore raised by one, in-kernel, into chan_last
      (= chan_bound + 1 = D-1 = 47), leaving the harness's LR6 untouched.
      Each query re-SETs k_ptr and chan_index, so no pipeline state crosses a
      query or block boundary.

    Loop nest:  for b in 0..15 { for i in 0..63 {
                    prime c=0; for c in 0..47: MULT.RC.VE + ACC
                    ACTIVATE.QUANTIZE + ST.AAQ
                }}

    Addressing is split offset(LR) + base(CR) and all .asm operands are ROW
    numbers (issue #179). q_ptr and out_ptr advance monotonically across all
    16 blocks (block b's queries are rows b*64 .. b*64+63 in both regions), so
    only k_block_off needs a per-block step (+48 rows).

    Registers are referred to below by the symbolic names defined in the
    register-name block. The assembler's Jinja2 preprocessor substitutes them
    before parsing, so the emitted binary is byte-identical to the raw form.
    NOTE: Jinja runs before comment stripping, so '#' comments must not
    contain Jinja delimiters -- the preprocessor would try to execute them.

    The register-name block below names every register. The ones the harness
    sets are kernel.toml's [abi] bindings (cr.<name>, lr.<name>); CR0 and CR1
    are the hardwired 0 and 1, and the other LRs are the program's own.

    ---------------------------------------------------------------------------
    Register names (Jinja2 preprocessor; pure source-level substitution)
    ---------------------------------------------------------------------------
-#}
{% set rc_slot0    = lr.rc_slot0  %}  {# const 0: R_CYCLIC write-index / mask_shift #}
{% set k_stride    = lr.k_stride  %}  {# 1 = rows per K channel column #}
{% set out_stride  = lr.out_stride  %}  {# 1 = rows per query (one key group) #}
{% set k_ptr       = "LR4"  %}  {# row offset into K, walks head channels c #}
{% set chan_index  = "LR5"  %}  {# head-channel index c -> selects Q[i,c] in R0 #}
{% set chan_bound  = lr.chan_bound  %}  {# 46 = contraction bound (first=0, width=48) #}
{% set chan_last   = "LR11" %}  {# 47 = chan_bound + 1; BLT.S bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = lr.out_ptr  %}  {# row offset into S, += out_stride per query #}
{% set q_ptr       = lr.q_ptr  %}  {# row offset into staged QROW, += qrow_stride #}
{% set q_index     = lr.q_index  %}  {# query counter i within the block #}
{% set q_limit     = lr.n_tok %}  {# 64 = N queries per block #}
{% set qrow_stride = lr.q_stride %}  {# 1 = QROW rows per query #}
{% set k_block_off = lr.k_block_off %}  {# K row offset of the current block, += 48 #}
{% set blk_index   = lr.block_index %}  {# block counter b (stream, head) #}
{% set k_start     = "LR15" %}  {# k_block_off + K_START; per-query k_ptr seed #}

{% set ZERO        = "CR0"  %}  {# hardwired read-only 0 #}
{% set K_BASE      = cr.b  %}  {# channel-major K base #}
{% set ONE         = "CR1"  %}  {# hardwired read-only 1 #}
{% set S_BASE      = cr.y  %}  {# query-major score output base #}
{% set K_START     = cr.k_start  %}  {# -1 row startup skew (first live = block base) #}
{% set CHAN_START  = cr.chan_start  %}  {# -1 -> first live c = 0 #}
{% set CHAN_BOUND  = cr.chan_bound_unread  %}  {# 46 = D - 2 #}
{% set QROW_BASE   = cr.a  %}  {# staged query-major Q base #}
{% set BLK_LIMIT   = cr.n_block %}  {# 16 = P * N_HEAD blocks #}
{% set K_BLOCK     = cr.k_block %}  {# 48 = K rows per (stream, head) block #}
{% set DSTRUCT     = cr.dstruct %}  {# reserved dstructure register #}

    # chan_last = chan_bound + 1 (= D-1 = 47). See the cycle-start value note in the
    # header: chan_index's ADD now sits one word ahead of the BLT.S, so the branch
    # compares an already-incremented index and needs the higher bound.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

blk_loop:
    SET {{ q_index }} {{ ZERO }};;                        # query counter = 0
    ADD {{ k_start }} {{ k_block_off }} {{ K_START }};;   # per-query k_ptr seed = block base - 1 row

q_loop:
    LD.MULT R0 {{ q_ptr }} {{ QROW_BASE }};;         # R0 = QROW[b, i] = Q[b, i, 0..47] (rest pad)

    ADD {{ k_ptr }} {{ k_start }} {{ ZERO }};;            # K-data startup: block base - 1 row (LR copy)
    SET {{ chan_index }} {{ CHAN_START }};;               # channel fixed_idx startup: -1

    # Prime c=0's K column: MULT reads R_CYCLIC from the cycle-start value, so the column
    # it consumes must be loaded a cycle earlier (see header).
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.ADD.FIRST seeds R_ACC.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_pre;;
    B after_c;;

c_loop_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;     # name the channel just loaded

c_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_pre;;

after_c:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ out_ptr }} {{ S_BASE }};;  # store row -> S[b, i, keys 0..63]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;    # advance output row

    ADD {{ q_ptr }} {{ q_ptr }} {{ qrow_stride }};
    ADD {{ q_index }} {{ q_index }} {{ ONE }};;  # next query
    BLT.S {{ q_index }} {{ q_limit }} q_loop;;

    ADD {{ k_block_off }} {{ k_block_off }} {{ K_BLOCK }};;   # next block: K base += 48 rows
    ADD {{ blk_index }} {{ blk_index }} {{ ONE }};;
    BLT.S {{ blk_index }} {{ BLK_LIMIT }} blk_loop;;

end:
    HALT;;
