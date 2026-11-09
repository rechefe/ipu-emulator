{#- attn_scores_km_16x60.asm -- K Q^T scores, key-major, one head (Layer 5: D=60, N=16, FP32)

    Layer:   L5
    Scope:   single-stream
    Layout:  unpacked
    Shape:   16tok x 60chan (head_dim), one head
    Related: L5 variant of attn_scores_km_256x36 (L3) / attn_scores_km_64x48 (L4);
             feeds attn_v_bcast_60 (key-major chain); qk_scores_16x60 is the
             query-major sibling for the same layer
    Tests:   //kernels/attention/attn_scores_km_16x60; the family's test_datapath.py holds it
             bit for bit to the ACC datapath mirror

    S[i, s] = sum_c Q[i, c] * K[s, c]      i,s in [0,16), c in [0,60)
      Lanes = queries (i); outer loop = key (s); contraction = head channel (c).

    L5 variant of attn_scores_km_256x36. The MAPPING is unchanged -- a scalar taken
    from R0 (key s's head-channel vector) times a vector in R_CYCLIC (Q's channel
    column), accumulated over channels with ACC.ADD. Only the counts move:
      * D 36 -> 60. head_dim is the contraction LOOP COUNT (bound D-2),
        not a lane count, so 60 needs no padding.
      * N_TOK 256 -> 16. At L3 the 256 queries spanned TWO 128-lane query groups
        (g=0, g=1), so each key stored two rows. At L5 all 16 queries fit in ONE
        group, so the g=1 half is gone and each key produces exactly ONE row.

    ONE CHANNEL PER ROW: a Q channel column, a key-major K row and an output key
    column are each a WHOLE row (128 FP32 lanes) carrying 16 live FP32 lanes. Rows are never
    shared; CR15.L's valid_elements = 16 tells ACTIVATE.EXACT how many lanes to
    emit, and the harness unpacks those lanes of y.

    Memory layout (row addresses; XMEM operands are ROW numbers, issue #179):
      Q: channel-major     Q[i, c] at QBASE + c*Q_CHAN_ROWS  (one row per channel,
         16 live query lanes) -> loads straight into R_CYCLIC (vector operand).
      K: key-major K[s, 0:59] at KBASE_KM + s*K_STRIDE_ROWS (60 channels
         zero-padded to a whole row). Loaded once per key s into R0, and the
         channel index c selects the scalar K[s, c].
      S: key-major rows    S[:, s] at SBASE + s*OUT_ROWS, lane i = query i.

    Compute (matmul template: scalar from R0 indexed by c × vector in R_CYCLIC):
      MULT.RC.VE rc_idx=0, src=chan_index (=c → R0[c]) → MULT_RES[i] = Q[i,c]·K[s,c]
      c==0: ACC.ADD.FIRST  else: ACC.ADD  → R_ACC[i] += Q[i,c]·K[s,c]
      After 60 channels: R_ACC[i] = S[i,s] for the 16 queries.
      ACTIVATE.EXACT identity (activate and store) → one key-major score row.
      No AGG.

    Counts/head: 16 keys × 60 channels = 960 MULT+ACC instruction words + 16 stores.

    Registers are referred to below by the symbolic names defined in the
    register-name block. The assembler's Jinja2 preprocessor substitutes them
    before parsing, so the emitted binary is byte-identical to the raw form.
    NOTE: Jinja runs before comment stripping, so '#' comments must not
    contain Jinja delimiters -- the preprocessor would try to execute them.

    The register-name block below names every register. The CRs the harness
    sets are kernel.toml's [abi] bindings (cr.<name>); CR0 and CR1 are the
    hardwired 0 and 1, and every LR is the program's own: it sets each one
    before it reads it (a constant that is never written is read from a CR).

    ---------------------------------------------------------------------------
    Register names (Jinja2 preprocessor; pure source-level substitution)
    ---------------------------------------------------------------------------
-#}
{% set rc_slot0    = "LR0"  %}  {# const 0: R_CYCLIC write/read index; set in the key loop #}
{% set chan_stride = "CR1"  %}  {# 1 = rows between consecutive Q channel columns: the hardwired 1 #}
{% set out_stride  = "CR1"  %}  {# 1 = rows per stored key row: the hardwired 1 #}
{% set q_ptr       = "LR4"  %}  {# row offset into Q, walks head channels c #}
{% set chan_index  = "LR5"  %}  {# head-channel index c -> selects K[s,c] in R0 #}
{% set chan_last   = cr.chan_last  %}  {# 59 = D-1; BLT.S bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = "LR7"  %}  {# row offset into S, += out_stride per store #}
{% set key_ptr     = "LR8"  %}  {# row offset into key-major K, += key_stride #}
{% set key_index   = "LR9"  %}  {# key counter s #}
{% set key_limit   = cr.n_tok %}  {# 16 = N keys #}
{% set key_stride  = "CR1" %}  {# 1 = rows per key row in K: the hardwired 1 #}

{% set Q_BASE      = cr.b  %}  {# channel-major Q base (b) #}
{% set ZERO        = "CR0"  %}  {# hardwired read-only 0 #}
{% set ONE         = "CR1"  %}  {# hardwired read-only 1 #}
{% set S_BASE      = cr.y  %}  {# key-major score output base #}
{% set Q_START     = cr.q_start  %}  {# -Q_CHAN_ROWS startup skew (first live = 0) #}
{% set CHAN_START  = cr.chan_start  %}  {# -1 -> first live c = 0 #}
{% set K_BASE_KM   = cr.a  %}  {# key-major K base (a) #}
{% set DSTRUCT     = cr.dstruct ~ ".L" %}  {# the DS half #}

    # chan_last = D-1 = 59. MULT reads R_CYCLIC from the
    # cycle-start value (issue #157), so each Q channel column is loaded a
    # instruction word before the MULT consuming it and chan_index's ADD moves into the
    # c_loop_pre block (the scalar operand is read forwarded from R0 and must name the
    # channel just loaded). BLT.S reads the CYCLE-START, so with that ADD a word
    # earlier it sees an already-incremented index and needs the higher bound, or
    # the last channel is dropped.
    #
    # The program sets every LR it reads: the three below here, LR0 in the key loop.
    SET {{ out_ptr }} {{ ZERO }};
    SET {{ key_ptr }} {{ CHAN_START }};                 # -K_STRIDE_ROWS: the loop's ADD makes it 0
    SET {{ key_index }} {{ ZERO }};;

s_loop:
    ADD {{ key_ptr }} {{ key_ptr }} {{ key_stride }};;  # key row offset += 1 (first live = 0)
    LD.MULT R0 {{ key_ptr }} {{ K_BASE_KM }};;     # R0[0..127] = K[s, 0:59] + zeros

    # -- the single query group (queries 0..15) -----------------------------
    SET {{ q_ptr }} {{ Q_START }};                      # channel-column startup: -Q_CHAN_ROWS
    SET {{ rc_slot0 }} {{ ZERO }};;                     # R_CYCLIC slot 0 (set again every key: harmless)
    SET {{ chan_index }} {{ CHAN_START }};;             # fixed_idx c startup: -1

    # Prime c=0's Q column (the MULT below consumes it from the cycle-start value).
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.ADD.FIRST seeds R_ACC.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_pre;;
    B after_c;;

c_loop_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;   # name the channel just loaded

c_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT.S {{ chan_index }} {{ chan_last }} c_loop_pre;;

after_c:
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ S_BASE }} {{ out_ptr }} ;;  # store S[0:16, s] (key-major row)
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};  # output row ptr += OUT_ROWS
    ADD {{ key_index }} {{ key_index }} {{ ONE }};;    # a word ahead of its BLT.S (BLT.S reads the cycle start)

    BLT.S {{ key_index }} {{ key_limit }} s_loop;;  # next key

end:
    HALT;;
