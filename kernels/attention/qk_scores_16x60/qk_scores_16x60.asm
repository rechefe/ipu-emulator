{#- qk_scores_16x60.asm -- QK^T scores, query-major (Layer 5), one attention head (FP32)

    Layer:   L5
    Scope:   single-stream
    Layout:  unpacked
    Shape:   16tok x 60chan (head_dim), one head
    Related: L5 variant of qk_scores_256x36 (L3) / qk_scores_64x48 (L4); feeds
             attn_v_16x60 (query-major chain); attn_scores_km_16x60 is the
             key-major sibling for the same layer
    Tests:   //kernels/attention/qk_scores_16x60; the family's test_datapath.py holds it
             bit for bit to the ACC datapath mirror

      S[i, s] = sum_c Q[i, c] * K[s, c]   contraction over head_dim c = 0..59

    One head: N = 16 tokens (queries = keys), head_dim D = 60.
    This is the L5 variant of qk_scores_256x36. The MAPPING is unchanged; only the
    loop counts move:
      * D 36 -> 60. head_dim is a contraction LOOP COUNT here (chan_bound = D-2),
        not a lane count, so 60 needs no padding.
      * N 256 -> 16. 256 keys spanned two 128-lane key groups; 16 keys fit in ONE
        group, so the entire g=1 half of the L3 kernel is gone.

    ONE CHANNEL PER ROW: a K channel column is 16 live FP32 elements inside a
    WHOLE row (128 FP32 lanes), never packed with a neighbouring channel.

      Q, K logically channel-major. K is loaded channel-major as given:
        K[s, c] at K_BASE + c*K_STRIDE rows (a full channel column, 16 keys,
        loaded as ONE chunk into R_CYCLIC).
      Q is query-major in a (the case supplies it so; gen.py transposes):
        QROW[i] = Q[i, 0..59] contiguous at QROW_BASE + i*QROW_STRIDE.
        This lets R0 hold one query's 60 channels with a single LD.MULT,
        exactly the matmul broadcast template (scalar from R0 indexed by c).

    Broadcast template (mirrors matmul k-loop1, width 60):
      per query i:  R0 = QROW[i]              (60 scalars Q[i, 0..59])
      per channel:  R_CYCLIC = K[0..15, c]        (16 keys' channel-c column)
                    MULT.RC.VE: scalar Q[i,c] (= R0[c]) x vector R_CYCLIC
                      -> MULT_RES[s] = Q[i,c] * K[s,c]
                    ACC.ADD.FIRST (c=0) / ACC.ADD  -> R_ACC[s] += Q[i,c] * K[s,c]
      after 60 channels R_ACC[s] = S[i, s] for the 16 keys.
      ACTIVATE.EXACT identity (activate and store)
                    -> S[i] (one whole row, first 16 lanes live, query-major)

    No AGG. The store goes through the standard quantize boundary, same as every
    other kernel; `identity` is the right activation here -- softmax applies its
    own nonlinearity downstream. CR15.valid_elements = 16 narrows the AAQ window
    to the live keys.

    MULT CYCLE-START CONTRACT (issue #157): MULT.RC.VE reads its R_CYCLIC DATA from
    the cycle-start value while keeping the LR index live, so it cannot
    consume a K column loaded by LD.MULT.CYCLIC in its own instruction word -- it
    would see the previous column. `;;` ends one instruction word = one
    cycle, so a load and a MULT in the same instruction word always run in
    the same cycle regardless of textual order; co-issuing is fine, consuming the
    same-cycle load is not.
      The key group therefore primes c=0's column, then every loop body
      multiplies the column loaded last cycle while prefetching the next one.
      k_ptr (the load offset) is read forwarded by the load, so its ADD stays
      co-issued with the load it feeds -- and because the load now runs one
      instruction word ahead, k_ptr naturally addresses the NEXT channel's column.
      chan_index (the R0 scalar selector) is read forwarded by MULT and must name the
      channel just loaded, so its ADD moves into a c_loop_pre block that falls
      through into the loop body. That changes what BLT.S sees: with the ADD in the
      same word, BLT.S read the pre-ADD cycle-start value; with the ADD one word earlier it
      reads the already-incremented value and would exit a channel early. The
      loop bound is therefore raised by one, in-kernel, into chan_last
      (= chan_bound + 1 = D-1 = 59), leaving the harness's LR6 untouched.

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
{% set k_stride    = lr.k_stride  %}  {# rows per K channel column #}
{% set out_stride  = lr.out_stride  %}  {# rows per query in S (1 key group x 1 row) #}
{% set k_ptr       = "LR4"  %}  {# row offset into K, walks head channels c #}
{% set chan_index  = "LR5"  %}  {# head-channel index c -> selects Q[i,c] in R0 #}
{% set chan_bound  = lr.chan_bound  %}  {# 58 = contraction bound (first=0, width=60) #}
{% set chan_last   = "LR11" %}  {# 59 = chan_bound + 1; BLT.S bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = lr.out_ptr  %}  {# row offset into S, += out_stride per query #}
{% set q_ptr       = lr.q_ptr  %}  {# row offset into staged QROW, += qrow_stride #}
{% set q_index     = lr.q_index  %}  {# query counter i #}
{% set q_limit     = lr.n_tok %}  {# 16 = N queries #}
{% set qrow_stride = lr.q_stride %}  {# 1 = QROW stride in rows per query #}

{% set K_BASE      = "CR0"  %}  {# channel-major K base #}
{% set ONE         = "CR1"  %}  {# hardwired read-only 1 #}
{% set S_BASE      = cr.y  %}  {# S base (single key group) #}
{% set K_START     = cr.k_start  %}  {# -1 row startup skew (first live = row 0) #}
{% set CHAN_START  = cr.chan_start  %}  {# -1 -> first live c = 0 #}
{% set QROW_BASE   = cr.a  %}  {# staged query-major Q base #}
{% set DSTRUCT     = cr.dstruct ~ ".L" %}  {# DS half: valid_elements = 16 #}

    # chan_last = chan_bound + 1 (= D-1 = 59). See the cycle-start value note in the
    # header: chan_index's ADD now sits one word ahead of the BLT.S, so the branch
    # compares an already-incremented index and needs the higher bound.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

q_loop:
    LD.MULT R0 {{ q_ptr }} {{ QROW_BASE }};;                                               # R0 = QROW[i] = Q[i, 0..59] (rest pad)

    # -- the single key group (keys 0..15) -----------------------------------
    SET {{ k_ptr }} {{ K_START }};;                                                             # K-data startup: -1 row
    SET {{ chan_index }} {{ CHAN_START }};;                                                     # channel fixed_idx startup: -1

    # Prime c=0's K column: MULT reads R_CYCLIC from the cycle-start value, so the column
    # it consumes must be loaded a cycle earlier (see header).
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.ADD.FIRST seeds R_ACC.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_pre;;
    B after_c;;

c_loop_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;                                            # name the channel just loaded

c_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_pre;;

after_c:
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ S_BASE }} {{ out_ptr }} ;;  # store one row -> S[i, keys 0..15]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;                                          # advance output ptr (+1 row)

    ADD {{ q_ptr }} {{ q_ptr }} {{ qrow_stride }};
    ADD {{ q_index }} {{ q_index }} {{ ONE }};;  # next query: Q ptr += 1 row, i++
    BLT.S {{ q_index }} {{ q_limit }} q_loop;;

end:
    HALT;;
