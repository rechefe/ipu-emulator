{#- attn_scores_km_64x48.asm -- K Q^T scores, key-major (Layer 4), one selected head, all streams (FP32)

    Layer:   L4
    Scope:   single-stream
    Layout:  unpacked
    Shape:   64tok x 48chan (head_dim), all 4 streams, one selected head
    Related: L4 variant of attn_scores_km_256x36 (L3); attn_scores_km_16x60 is the
             L5 port; feeds attn_v_bcast_48 (key-major chain); qk_scores_64x48
             is the query-major sibling for the same layer
    Tests:   //kernels/attention/attn_scores_km_64x48; the family's test_datapath.py holds it
             bit for bit to the ACC datapath mirror

    Per stream p in [0,4), for the head the harness selected:
      S[p, i, s] = sum_c Q[p, i, c] * K[p, s, c]     i,s in [0,64), c in [0,48)
        Lanes = QUERIES (i); outer loop = key (s); contraction = head channel (c).

    Layer 4: d = 192, N = 64 tokens/stream, P = 4 streams, h = 4 heads,
    head_dim D = 48. head_dim = 48 needs NO padding: it is the contraction LOOP
    COUNT (bound chan_bound = D-2 = 46), not a lane count.

    This is the L4 variant of attn_scores_km_256x36. The mapping is carried over
    unchanged -- broadcast template, no AGG, no MULT.RC.VV score form:
      Q channel-major: Q[p, i, c] on row (p*48 + c), queries 0..63 in the first 64
        lanes -> loads straight into R_CYCLIC (the vector operand).
      K key-major scratch (built by the harness): K[p, s, 0:48] on row (p*64 + s)
        -> one LD.MULT pulls a key's whole head-channel vector into R0 (the
        scalar operand, indexed by the contraction channel c).
      S key-major: key (p, s) owns row (p*64 + s); its leading 64 lanes are the
        query scores S[p, 0:64, s]. The harness unpacks the first n_tok lanes -- rows are
        NEVER shared between keys.

    N = 64 <= LANES = 128 is the one structural difference from L3: a query axis
    fits in a SINGLE row, so the L3 g=0/g=1 query-group split collapses to one
    group (N_TG = 1) and each key needs exactly one MULT/ACC chain and one store.

    Compute (matmul template: scalar from R0 indexed by c x vector in R_CYCLIC):
      MULT.RC.VE rc_idx=0, src=chan_index (= c -> R0[c])
          -> MULT_RES[i] = Q[i,c] * K[s,c]
      c==0: ACC.ADD.FIRST  else: ACC.ADD   -> R_ACC[i] += Q[i,c] * K[s,c]
      After 48 channels: R_ACC[i] = S[p, i, s] for the 64 queries of the stream.
      ACTIVATE.QUANTIZE identity + ST.AAQ -> one whole row (128 FP32 lanes).

    MULT CYCLE-START CONTRACT (issue #157): MULT.RC.VE reads its R_CYCLIC DATA from
    the cycle-start value while keeping the LR index live, so it cannot
    consume a Q column loaded by LD.MULT.CYCLIC in its own bundle -- it would
    see the previous column. `;;` ends one instruction word = one cycle = one cycle start,
    so a load and a MULT in the same bundle always run in the same cycle
    regardless of textual order; co-issuing is fine, consuming the same-cycle load
    is not.
      Each key therefore primes c=0's column, then every loop body multiplies the
      column loaded last cycle while prefetching the next one.
      q_ptr (the load offset) is read LIVE by the load, so its ADD stays co-issued
      with the load it feeds -- and because the load now runs one bundle ahead,
      q_ptr naturally addresses the NEXT channel's column.
      chan_index (the R0 scalar selector) is read LIVE by MULT and must name the
      channel just loaded, so its ADD moves into a c_loop_pre block that falls
      through into the loop body. That changes what BLT.S sees: with the ADD in the
      same word, BLT.S read the pre-ADD cycle start; with the ADD one word earlier it
      reads the already-incremented value and would drop the last channel. The
      loop bound is therefore raised by one, in-kernel, into chan_last
      (= chan_bound + 1 = D-1 = 47), leaving the harness's LR6 untouched.
      Each key re-SETs q_ptr and chan_index, so no pipeline state crosses a key or
      stream boundary.

    Loop nest:  for p in 0..3 { for s in 0..63 {
                    load K[p,s] -> R0; prime c=0
                    for c in 0..47: MULT.RC.VE + ACC
                    ACTIVATE.QUANTIZE + ST.AAQ
                }}

    Counts: 4 streams x 64 keys x 48 channels = 12288 MULT+ACC bundles + 256
            stores.

    Addressing is split offset(LR) + base(CR) and all .asm operands are ROW
    numbers (issue #179). key_ptr and out_ptr advance monotonically across all 4
    streams (key (p,s) is row p*64 + s in both regions), so only q_stream_off
    needs a per-stream step (+48 rows).

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
{% set rc_slot0     = lr.rc_slot0  %}  {# const 0: R_CYCLIC write/read index #}
{% set chan_stride  = lr.q_chan_stride  %}  {# 1 = rows between consecutive Q channels #}
{% set out_stride   = lr.out_stride  %}  {# 1 = rows per stored key row #}
{% set q_ptr        = "LR4"  %}  {# row offset into Q, walks head channels c #}
{% set chan_index   = "LR5"  %}  {# head-channel index c -> selects K[s,c] in R0 #}
{% set chan_bound   = lr.chan_bound  %}  {# 46 = c-loop bound (first=0, width=48) #}
{% set chan_last    = "LR11" %}  {# 47 = chan_bound + 1; BLT.S bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr      = lr.out_ptr  %}  {# row offset into S, += out_stride per store #}
{% set key_ptr      = lr.key_ptr  %}  {# row offset into key-major K, += key_stride #}
{% set key_index    = lr.key_index  %}  {# key counter s within the stream #}
{% set key_limit    = lr.n_tok %}  {# 64 = N keys per stream #}
{% set key_stride   = lr.key_stride %}  {# 1 = rows per key in the K scratch #}
{% set q_stream_off = lr.q_stream_off %}  {# Q row offset of the current stream, += 48 #}
{% set stream_index = lr.stream_index %}  {# stream counter p #}
{% set q_start      = "LR15" %}  {# q_stream_off + Q_START; per-key q_ptr seed #}

{% set ZERO         = "CR0"  %}  {# hardwired read-only 0 #}
{% set ONE          = "CR1"  %}  {# hardwired read-only 1 #}
{% set S_BASE       = cr.y  %}  {# key-major score output base #}
{% set Q_BASE       = cr.b  %}  {# channel-major Q base #}
{% set Q_START      = cr.q_start  %}  {# -1 row startup skew (first live = stream base) #}
{% set CHAN_START   = cr.chan_start  %}  {# -1 -> first live c = 0 #}
{% set CHAN_BOUND   = cr.chan_bound_unread  %}  {# 46 = D - 2 #}
{% set K_BASE_KM    = cr.a  %}  {# key-major K scratch base #}
{% set STREAM_LIMIT = cr.p %}  {# 4 = P streams #}
{% set Q_STREAM     = cr.q_stream_rows %}  {# 48 = Q rows per stream #}
{% set DSTRUCT      = "CR15" %}  {# reserved dstructure register #}

    # chan_last = chan_bound + 1 (= D-1 = 47). See the cycle-start value note in the
    # header: chan_index's ADD now sits one word ahead of the BLT.S, so the branch
    # compares an already-incremented index and needs the higher bound.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

p_loop:
    SET {{ key_index }} {{ ZERO }};;                        # key counter = 0
    ADD {{ q_start }} {{ q_stream_off }} {{ Q_START }};;    # per-key q_ptr seed = stream base - 1 row

s_loop:
    LD.MULT R0 {{ key_ptr }} {{ K_BASE_KM }};;         # R0 = K[p, s, 0:48] (rest pad)

    ADD {{ q_ptr }} {{ q_start }} {{ ZERO }};;              # Q-data startup: stream base - 1 row (LR copy)
    SET {{ chan_index }} {{ CHAN_START }};;                 # fixed_idx c startup: -1

    # Prime c=0's Q column: MULT reads R_CYCLIC from the cycle-start value, so the column
    # it consumes must be loaded a cycle earlier (see header).
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.ADD.FIRST seeds R_ACC.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_pre;;
    B after_c;;

c_loop_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;       # name the channel just loaded

c_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_pre;;

after_c:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ out_ptr }} {{ S_BASE }};;  # store S[p, 0:64, s] (key-major row)
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;      # advance output row

    ADD {{ key_ptr }} {{ key_ptr }} {{ key_stride }};
    ADD {{ key_index }} {{ key_index }} {{ ONE }};;  # next key
    BLT.S {{ key_index }} {{ key_limit }} s_loop;;

    ADD {{ q_stream_off }} {{ q_stream_off }} {{ Q_STREAM }};;  # next stream: Q base += 48 rows
    ADD {{ stream_index }} {{ stream_index }} {{ ONE }};;
    BLT.S {{ stream_index }} {{ STREAM_LIMIT }} p_loop;;

end:
    HALT;;
