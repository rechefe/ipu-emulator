{#- qk_scores_256x36.asm -- QK^T scores, query-major, one attention head (FP32)

    Layer:   L3
    Scope:   single-stream
    Layout:  unpacked
    Shape:   256tok (2 groups x 128) x 36chan (head_dim), one head
    Related: qk_scores_64x48 (L4) / qk_scores_16x60 (L5) are ports of this
             kernel; feeds attn_v_256x36 (query-major chain); attn_scores_km_256x36
             is the key-major sibling for the same layer
    Tests:   //kernels/attention/qk_scores_256x36; the family's test_datapath.py
             holds it bit for bit to the ACC datapath mirror

      S[i, s] = sum_c Q[i, c] * K[s, c]   contraction over head_dim c = 0..35

    One head: N = 256 tokens (queries = keys), head_dim D = 36.
      All XMEM operands are ROW numbers; one row is 128 FP32 lanes. The harness
      places and packs the regions (kernel.toml: b = K, a = Q, y = S):
      K channel-major: channel c is two rows, keys g*128..g*128+127 in row
        K_BASE + 2c + g (loaded as two 128-key chunks).
      Q query-major in a (the case supplies it so; gen.py transposes): QROW[i] = Q[i, 0..35] in
        the first 36 lanes of row QROW_BASE + i.
        This lets R0 hold one query's 36 channels with a single LD.MULT,
        exactly the matmul broadcast template (scalar from R0 indexed by c).

    Broadcast template (mirrors matmul_144x144_x128 k-loop1, width 36):
      per query i:  R0 = QROW[i]              (36 scalars Q[i, 0..35])
      per channel:  R_CYCLIC = K[g*128:+128, c]   (128 keys' channel-c column)
                    MULT.RC.VE: scalar Q[i,c] (= R0[c]) × vector R_CYCLIC
                      → MULT_RES[s] = Q[i,c] * K[s,c]
                    ACC.ADD.FIRST (c=0) / ACC.ADD      → R_ACC[s] += Q[i,c] * K[s,c]
      after 36 channels R_ACC[s] = S[i, s] for the 128 keys of group g.
      ACTIVATE.EXACT identity (activate and store)
                    → S[i, g], one whole row (128 FP32 lanes, query-major),
                      row S_BASE + 2i + g

    No AGG. The store goes through ACTIVATE.EXACT like every kernel's;
    `identity` is the right activation here -- softmax applies its own
    nonlinearity downstream.

    MULT CYCLE-START CONTRACT (issue #157): MULT.RC.VE reads its R_CYCLIC DATA from
    the cycle-start value while keeping the LR index live, so it cannot
    consume a K column loaded by LD.MULT.CYCLIC in its own instruction word -- it
    would see the previous column. `;;` ends one instruction word = one
    cycle, so a load and a MULT in the same instruction word always run in
    the same cycle regardless of textual order; co-issuing is fine, consuming the
    same-cycle load is not.
      Each key group therefore primes c=0's column, then every loop body
      multiplies the column loaded last cycle while prefetching the next one.
      k_ptr (the load offset) is read forwarded by the load, so its ADD stays
      co-issued with the load it feeds -- and because the load now runs one
      instruction word ahead, k_ptr naturally addresses the NEXT channel's column.
      chan_index (the R0 scalar selector) is read forwarded by MULT and must name the
      channel just loaded, so its ADD moves into a c_loop_*_pre block that falls
      through into the loop body. That changes what BLT.S sees: with the ADD in the
      same word, BLT.S read the pre-ADD cycle-start value; with the ADD one word earlier it
      reads the already-incremented value and would exit a channel early. The
      loop bound is therefore raised by one, in-kernel, into chan_last
      (= chan_bound + 1 = D-1 = 35), leaving the harness's LR6 untouched.
      The two key groups are independent contractions (each re-SETs k_ptr and
      chan_index), so no pipeline state crosses the g=0/g=1 boundary.

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
{% set k_stride    = lr.k_stride  %}  {# 2 = rows per K channel column #}
{% set out_stride  = lr.out_stride  %}  {# 2 = rows per query (2 key groups) #}
{% set k_ptr       = "LR4"  %}  {# row offset into K, walks head channels c #}
{% set chan_index  = "LR5"  %}  {# head-channel index c -> selects Q[i,c] in R0 #}
{% set chan_bound  = lr.chan_bound  %}  {# 34 = contraction bound (first=0, width=36) #}
{% set chan_last   = "LR11" %}  {# 35 = chan_bound + 1; BLT.S bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = lr.out_ptr  %}  {# row offset into S, += out_stride per query #}
{% set q_ptr       = lr.q_ptr  %}  {# row offset into staged QROW, += qrow_stride #}
{% set q_index     = lr.q_index  %}  {# query counter i #}
{% set q_limit     = lr.n_tok %}  {# 256 = N queries #}
{% set qrow_stride = lr.q_stride %}  {# 1 = QROW rows per query #}

{% set K_BASE      = "CR0"  %}  {# channel-major K base #}
{% set ONE         = "CR1"  %}  {# hardwired read-only 1 #}
{% set S_BASE_G0   = cr.y  %}  {# S base, key group 0 #}
{% set S_BASE_G1   = cr.y_g1  %}  {# S base + 1 row, key group 1 #}
{% set K_START_G0  = cr.k_start_g0  %}  {# -2 rows startup skew (first live = row 0) #}
{% set K_START_G1  = cr.k_start_g1  %}  {# -1 row startup skew (first live = row 1) #}
{% set CHAN_START  = cr.chan_start  %}  {# -1 -> first live c = 0 #}
{% set QROW_BASE   = cr.a  %}  {# staged query-major Q base #}
{% set DSTRUCT     = "CR15.L" %}  {# the DS half #}

    # chan_last = chan_bound + 1 (= D-1 = 35). See the cycle-start value note in the
    # header: chan_index's ADD now sits one word ahead of the BLT.S, so the branch
    # compares an already-incremented index and needs the higher bound.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

q_loop:
    LD.MULT R0 {{ q_ptr }} {{ QROW_BASE }};;                                               # R0 = QROW[i] = Q[i, 0..35] (rest pad)

    # -- key group 0 ---------------------------------------------------------
    SET {{ k_ptr }} {{ K_START_G0 }};;                                                          # g=0 K-data startup: -2 rows
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
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_g0_pre;;
    B after_c_g0;;

c_loop_g0_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;                                            # name the channel just loaded

c_loop_g0:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_g0_pre;;

after_c_g0:
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ S_BASE_G0 }} {{ out_ptr }} ;;  # store a row → S[i, keys 0..127]

    # -- key group 1 ---------------------------------------------------------
    SET {{ k_ptr }} {{ K_START_G1 }};;                                                          # g=1 K-data startup: -1 row
    SET {{ chan_index }} {{ CHAN_START }};;                                                     # channel fixed_idx startup: -1

    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_g1_pre;;
    B after_c_g1;;

c_loop_g1_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;                                            # name the channel just loaded

c_loop_g1:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_g1_pre;;

after_c_g1:
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ S_BASE_G1 }} {{ out_ptr }} ;;  # store a row → S[i, keys 128..255]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;                                          # advance output ptr (+2 rows)

    ADD {{ q_ptr }} {{ q_ptr }} {{ qrow_stride }};
    ADD {{ q_index }} {{ q_index }} {{ ONE }};;  # next query: Q ptr += 1 row, i++
    BLT.S {{ q_index }} {{ q_limit }} q_loop;;

end:
    HALT;;
