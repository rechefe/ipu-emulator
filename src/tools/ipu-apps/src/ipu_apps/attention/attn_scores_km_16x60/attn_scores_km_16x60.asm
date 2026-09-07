# kQᵀ → key-major attention scores, one head (Layer 5: D=60, N=16)
#
# Layer:   L5
# Scope:   single-stream
# Layout:  unpacked
# Shape:   16tok x 60chan (head_dim), one head
# Status:  validated
# Related: L5 port of attn_scores_km_256x36 (L3) / attn_scores_km_64x48 (L4);
#          feeds attn_v_bcast_60 (key-major chain); qk_scores_16x60 is the
#          query-major sibling for the same layer
# Tests:   test_attn_scores_km_16x60_wide (src/tools/ipu-apps/BUILD.bazel)
#
# S[i, s] = sum_c Q[i, c] * K[s, c]      i,s in [0,16), c in [0,60)
#   Lanes = queries (i); outer loop = key (s); contraction = head channel (c).
#
# L5 port of attn_scores_km_256x36. The MAPPING is unchanged -- a scalar taken
# from R0 (key s's head-channel vector) times a vector in R_CYCLIC (Q's channel
# column), accumulated over channels with ACC.ADD. Only the counts move:
#   * D 36 -> 60. head_dim is the contraction LOOP COUNT (chan_bound = D-2),
#     not a lane count, so 60 needs no padding.
#   * N_TOK 256 -> 16. At L3 the 256 queries spanned TWO 128-lane query groups
#     (g=0, g=1), so each key stored two rows. At L5 all 16 queries fit in ONE
#     group, so the g=1 half is gone and each key produces exactly ONE row.
#
# ONE CHANNEL PER ROW: a Q channel column, a key-major K row and an output key
# column are each a WHOLE 512-B row carrying 16 live FP32 lanes. Rows are never
# shared; cr15.valid_elements = 16 tells ACTIVATE.QUANTIZE how many lanes to
# emit and the consumer crops to those lanes.
#
# Memory layout (row addresses; XMEM operands are ROW numbers, issue #179):
#   Q: channel-major     Q[i, c] at QBASE + c*Q_CHAN_ROWS  (one row per channel,
#      16 live query lanes) -> loads straight into R_CYCLIC (vector operand).
#   K: key-major scratch K[s, 0:59] at KBASE_KM + s*K_STRIDE_ROWS (60 channels
#      zero-padded to a whole row). Loaded once per key s into R0, and the
#      channel index c selects the scalar K[s, c].
#   S: key-major rows    S[:, s] at SBASE + s*OUT_ROWS, lane i = query i.
#
# Compute (matmul template: scalar from R0 indexed by c × vector in R_CYCLIC):
#   MULT.RC.VE rc_idx=0, src=chan_index (=c → R0[c]) → mult_res[i] = Q[i,c]·K[s,c]
#   c==0: ACC.ADD.FIRST  else: ACC.ADD  → R_ACC[i] += Q[i,c]·K[s,c]
#   After 60 channels: R_ACC[i] = S[i,s] for the 16 queries.
#   ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG → one key-major score row.
#   No AGG.
#
# Counts/head: 16 keys × 60 channels = 960 MULT+ACC bundles + 16 stores.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.
#
# Register assignments and their meanings are listed in the register-name
# block below -- it is the single source of truth for this kernel.

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0    = "lr0"  %}  {# const 0: r_cyclic write/read index #}
{% set chan_stride = "lr2"  %}  {# rows between consecutive Q channel columns #}
{% set out_stride  = "lr3"  %}  {# rows per stored key row #}
{% set q_ptr       = "lr4"  %}  {# row offset into Q, walks head channels c #}
{% set chan_index  = "lr5"  %}  {# head-channel index c -> selects K[s,c] in r0 #}
{% set chan_bound  = "lr6"  %}  {# 58 = c-loop bound (first=0, width=60) #}
{% set chan_last   = "lr11" %}  {# 59 = chan_bound + 1; BLT bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = "lr7"  %}  {# row offset into S, += out_stride per store #}
{% set key_ptr     = "lr8"  %}  {# row offset into key-major K, += key_stride #}
{% set key_index   = "lr9"  %}  {# key counter s #}
{% set key_limit   = "lr10" %}  {# 16 = N keys #}
{% set key_stride  = "lr12" %}  {# rows per key row in the K scratch #}

{% set Q_BASE      = "cr0"  %}  {# channel-major Q base #}
{% set ONE         = "cr1"  %}  {# hardwired read-only 1 #}
{% set S_BASE      = "cr2"  %}  {# key-major score output base #}
{% set Q_START     = "cr5"  %}  {# -Q_CHAN_ROWS startup skew (first live = 0) #}
{% set CHAN_START  = "cr7"  %}  {# -1 -> first live c = 0 #}
{% set K_BASE_KM   = "cr9"  %}  {# key-major K scratch base #}
{% set DSTRUCT     = "cr15" %}  {# reserved dstructure register #}

    # chan_last = chan_bound + 1 (= D-1 = 59). MULT reads r_cyclic from the
    # start-of-cycle snapshot (issue #157), so each Q channel column is loaded a
    # bundle before the MULT consuming it and chan_index's ADD moves into the
    # c_loop_pre block (the scalar operand is read LIVE from R0 and must name the
    # channel just loaded). BLT reads the SNAPSHOT, so with that ADD a word
    # earlier it sees an already-incremented index and needs the higher bound, or
    # the last channel is dropped. Computed in-kernel so the harness's lr6 stays
    # untouched.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

s_loop:
    ADD {{ key_ptr }} {{ key_ptr }} {{ key_stride }};;  # key row offset += 1 (first live = 0)
    LDR_MULT_REG r0 {{ key_ptr }} {{ K_BASE_KM }};;     # r0[0..127] = K[s, 0:59] + zeros

    # -- the single query group (queries 0..15) -----------------------------
    SET {{ q_ptr }} {{ Q_START }};;                     # channel-column startup: -Q_CHAN_ROWS
    SET {{ chan_index }} {{ CHAN_START }};;             # fixed_idx c startup: -1

    # Prime c=0's Q column (the MULT below consumes it from the snapshot).
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_pre;;
    B after_c;;

c_loop_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;   # name the channel just loaded

c_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ q_ptr }} {{ Q_BASE }} {{ rc_slot0 }};
    ADD {{ q_ptr }} {{ q_ptr }} {{ chan_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_pre;;

after_c:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ S_BASE }};;  # store S[0:16, s] (key-major row)
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;  # output row ptr += OUT_ROWS

    ADD {{ key_index }} {{ key_index }} {{ ONE }};
    BLT {{ key_index }} {{ key_limit }} s_loop;;  # next key

end:
    BKPT;;
