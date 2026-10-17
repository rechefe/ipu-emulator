{#- attn_scores_km_256x36.asm -- K Q^T scores, key-major, one head (D=36, N=256, FP32)

    Layer:   L3
    Scope:   single-stream
    Layout:  unpacked
    Shape:   256tok (2 groups x 128) x 36chan (head_dim), one head
    Related: attn_scores_km_64x48 (L4) / attn_scores_km_16x60 (L5) are ports of
             this kernel; feeds attn_v_bcast_36 (key-major chain); qk_scores_256x36
             is the query-major sibling for the same layer
    Tests:   //kernels/attention/attn_scores_km_256x36; the family's
             test_datapath.py holds it bit for bit to the ACC datapath mirror

    S[i, s] = sum_c Q[i, c] * K[s, c]      i,s in [0,256), c in [0,36)
      Lanes = queries (i); outer loop = key (s); contraction = head channel (c).

    Memory layout. All XMEM operands are ROW numbers; one row is 128 FP32
    lanes. The harness places and packs the selected head's regions (kernel.toml:
    b = Q, a = K, y = S; the case's `head` picks the head):
      Q: channel-major: channel c is two rows, queries g*128..g*128+127 in row
         Q_BASE + 2c + g → loads straight into R_CYCLIC (vector operand).
      K: key-major: K[s, 0:35] in the first 36 lanes of row K_BASE_KM + s.
         Loaded once per key s into R0 (scalar operand, indexed by c).
      S: key-major: (s, g) in row S_BASE + 2s + g, queries g*128.. in its lanes.
         One running one-row-stride pointer covers g=0, g=1, next key.

    Compute (matmul template: scalar from R0 indexed by c × vector in R_CYCLIC):
      MULT.RC.VE rc_idx=0, src=chan_index (=c → R0[c]) → MULT_RES[i] = Q[i,c]·K[s,c]
      c==0: ACC.ADD.FIRST  else: ACC.ADD   → R_ACC[i] += Q[i,c]·K[s,c]
      After 36 channels: R_ACC[i] = S[i,s] for the 128 queries of this group.
      ACTIVATE.QUANTIZE identity + ST.AAQ
        → one whole key-major score row (128 FP32 lanes).  No AGG.

    Counts/head: 256 keys × 2 groups × 36 channels = 18432 MULT+ACC instruction words
                 + 512 stores.

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
{% set rc_slot0    = lr.rc_slot0  %}  {# const 0: R_CYCLIC write/read index #}
{% set chan_stride = lr.q_chan_stride  %}  {# 2 = rows between consecutive Q channels #}
{% set out_stride  = lr.out_stride  %}  {# 1 = rows per stored (key, group) row #}
{% set q_ptr       = "LR4"  %}  {# row offset into Q, walks head channels c #}
{% set chan_index  = "LR5"  %}  {# head-channel index c -> selects K[s,c] in R0 #}
{% set chan_bound  = lr.chan_bound  %}  {# 34 = c-loop bound (first=0, width=36) #}
{% set chan_last   = "LR11" %}  {# 35 = chan_bound + 1; BLT.S bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = lr.out_ptr  %}  {# row offset into S, += 1 per store #}
{% set key_ptr     = lr.key_ptr  %}  {# row offset into key-major K, += key_stride #}
{% set key_index   = lr.key_index  %}  {# key counter s #}
{% set key_limit   = lr.n_tok %}  {# 256 = N keys #}
{% set key_stride  = lr.key_stride %}  {# 1 = rows per key in K #}

{% set Q_BASE      = "CR0"  %}  {# channel-major Q base #}
{% set ONE         = "CR1"  %}  {# hardwired read-only 1 #}
{% set S_BASE      = cr.y  %}  {# key-major score output base #}
{% set Q_START_G0  = cr.q_start_g0  %}  {# -2 rows startup skew (first live = row 0) #}
{% set Q_START_G1  = cr.q_start_g1  %}  {# -1 row startup skew (first live = row 1) #}
{% set CHAN_START  = cr.chan_start  %}  {# -1 -> first live c = 0 #}
{% set K_BASE_KM   = cr.a  %}  {# key-major K base (a) #}
{% set DSTRUCT     = "CR15" %}  {# reserved dstructure register #}

    # chan_last = chan_bound + 1 (= D-1 = 35). MULT reads R_CYCLIC from the
    # cycle-start value (issue #157), so each Q channel column is loaded a
    # instruction word before the MULT consuming it and chan_index's ADD moves into a
    # c_loop_*_pre block (MULT reads it LIVE and must name the channel just
    # loaded). BLT.S reads the CYCLE-START, so with that ADD a word earlier it sees an
    # already-incremented index and needs the higher bound, or the last channel
    # is dropped. Computed in-kernel so the harness's LR6 stays untouched.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

s_loop:
    ADD {{ key_ptr }} {{ key_ptr }} {{ key_stride }};;  # key row offset += 1 (first live = 0)
    LD.MULT R0 {{ key_ptr }} {{ K_BASE_KM }};;     # R0[0..127] = K[s, 0:35] + zeros

    # -- query group 0 (queries 0..127) -------------------------------------
    SET {{ q_ptr }} {{ Q_START_G0 }};;                  # channel-column startup: -2 rows
    SET {{ chan_index }} {{ CHAN_START }};;             # fixed_idx c startup: -1

    # Prime c=0's Q column (the MULT below consumes it from the cycle-start value).
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.ADD.FIRST seeds R_ACC.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_g0_pre;;
    B after_c_g0;;

c_loop_g0_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;   # name the channel just loaded

c_loop_g0:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_g0_pre;;

after_c_g0:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ out_ptr }} {{ S_BASE }};;  # store S[0:128, s] (key-major row, g=0)
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;  # output ptr += 1 row

    # -- query group 1 (queries 128..255) -----------------------------------
    SET {{ q_ptr }} {{ Q_START_G1 }};;                  # g=1 channel-column startup: -1 row → first live = row 1
    SET {{ chan_index }} {{ CHAN_START }};;             # fixed_idx c startup: -1

    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_g1_pre;;
    B after_c_g1;;

c_loop_g1_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;   # name the channel just loaded

c_loop_g1:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_g1_pre;;

after_c_g1:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ out_ptr }} {{ S_BASE }};;  # store S[128:256, s] (key-major row, g=1)
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;  # output ptr += 1 row

    ADD {{ key_index }} {{ key_index }} {{ ONE }};
    BLT.S {{ key_index }} {{ key_limit }} s_loop;;  # next key

end:
    HALT;;
