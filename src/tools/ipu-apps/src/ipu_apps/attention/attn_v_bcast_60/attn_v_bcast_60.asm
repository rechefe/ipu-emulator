# attn@V (Layer 5), key-major scores -> channel-major output (broadcast / no AGG)
#
# Layer:   L5
# Scope:   single-stream
# Layout:  unpacked
# Shape:   16tok x 60chan (head_dim), 4 heads
# Status:  validated
# Related: L5 port of attn_v_bcast_36 (L3); attn_v_bcast_48 is the L4 port;
#          consumes attn_scores_km_16x60's key-major output; attn_v_16x60
#          is the query-major (AGG) sibling for the same layer
# Tests:   test_attn_v_bcast_60_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Per attention head h in [0,4):
#   O[i,t] = sum_s P[i,s] * V[s,t]
#     i = query in [0,16)          (the SIMD lanes; ONE group at L5)
#     t = head channel in [0,60)   (global value channel = h*60 + t)
#     s = key (contraction) in [0,16)
#
# L5 port of attn_v_bcast_36. The MAPPING is unchanged -- LANES = QUERIES, the
# standard matmul-broadcast template with V in the weight/scalar role and P
# key-major in the streamed-data role:
#   mult_res[i] = P[i,s] * V[s,t]  ;  ACC.ADD[.FIRST] accumulates over s
#   After 16 keys, R_ACC[i] = O[i,t] for the 16 queries.
# No AGG, no collision: every cycle is a full-width scalar*vector MULT + ACC.
# Only the counts move from L3:
#   * D 36 -> 60 head channels (t-loop).
#   * N_TOK 256 -> 16. At L3 the 256 queries needed TWO 128-lane groups (a
#     g-loop) and a channel's 256 key values needed BOTH R0 (s=0..127) and R1
#     (s=128..255). At L5 all 16 queries fit in ONE group and a channel's 16
#     key values fit entirely in R0, so the g-loop AND the R1 load are gone.
#
# ONE CHANNEL PER ROW: every P key row, every V channel row and every stored O
# row is a WHOLE 512-B row with 16 live FP32 lanes. cr15.valid_elements = 16
# tells ACTIVATE.QUANTIZE how many lanes to emit; the rest stay zero.
#
# Layouts (row addresses; XMEM operands are ROW numbers, issue #179).  V and O
# match the query-major sibling attn_v_16x60 EXACTLY, so the two chains'
# outputs are directly comparable:
#   P key-major (head-major): P[i,s] at PBASE + h*P_HEAD_STRIDE + s*KEY_STRIDE
#     (key s's 16 query scores = one row, query i in lane i).
#   V channel-major: V[s, chan] at VBASE + chan*KEY_STRIDE, chan = h*60 + t
#     (the channel's 16 key values = one row, all of them loaded into R0).
#   O channel-major: O[i,t] at OBASE + chan*OUT_STRIDE, lane i.
#
# VLIW timing:
#   LR slot runs first; XMEM + MULT then read LR values LIVE (post-ADD); BLT reads
#   the SNAPSHOT (pre-ADD).  Single-cycle inner body:
#     LDR_CYCLIC P[h,s_live]; ADD data ptr; MULT.RC.VE src=s_live; ACC.ADD; BLT
#   Startup offsets (one-cycle pipeline align):
#     data ptr  p_ptr = first_row - KEY_STRIDE   (ADD -> live = first_row)
#     key index key_index = -1                   (ADD +1 -> live s = 0)
#     BLT bound KEY_BOUND = 14  (first_index=0, width=16 -> 0+16-2=14)
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.
#
# Register assignments and their meanings are listed in the register-name
# block below -- it is the single source of truth for this kernel.
#
# Loop bounds (BLT reads the pre-ADD snapshot; branch taken while snapshot < bound):
#   KEY_BOUND   = 14          (key-loop bound: width 16, peeled + startup)
#   CHAN_BOUND  = 59          (t-loop bound: 60 channels per head)
#   HEAD_BOUND  = 3           (head-loop bound: 4 heads)

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0      = "lr0"  %}  {# const 0: r_cyclic index / mask_shift #}
{% set key_stride    = "lr1"  %}  {# P key stride and V channel stride (rows) #}
{% set out_stride    = "lr2"  %}  {# output-row stride (rows) #}
{% set p_ptr         = "lr4"  %}  {# row offset into P (key-major) #}
{% set key_index     = "lr5"  %}  {# contraction key index s -> selects V[s,t] in r0 #}
{% set head_index    = "lr6"  %}  {# head counter h #}
{% set chan_index    = "lr7"  %}  {# head-channel counter t #}
{% set key_last      = "lr8"  %}  {# 15 = KEY_BOUND + 1; BLT bound now that key_index's ADD sits a word ahead of the branch #}
{% set v_ptr         = "lr10" %}  {# chan row offset -> r0 = V[0:15, chan] #}
{% set p_head_base   = "lr12" %}  {# h*P_HEAD_STRIDE, P head base (rows) #}
{% set out_ptr       = "lr14" %}  {# output row offset, += out_stride per store #}

{% set ZERO          = "cr0"  %}  {# hardwired 0; const-0 and r_cyclic XMEM base #}
{% set ONE           = "cr1"  %}  {# hardwired read-only 1 #}
{% set P_BASE        = "cr2"  %}  {# key-major scores base #}
{% set V_BASE        = "cr3"  %}  {# channel-major value base #}
{% set OUT_BASE      = "cr4"  %}  {# channel-major output base #}
{% set KEY_START     = "cr6"  %}  {# -1 -> first live s = 0 #}
{% set P_HEAD_STRIDE = "cr7"  %}  {# P head stride (rows) #}
{% set KEY_BOUND     = "cr8"  %}  {# 14 = key-loop bound (width 16) #}
{% set CHAN_BOUND    = "cr9"  %}  {# 59 = t-loop bound (60 channels/head) #}
{% set HEAD_BOUND    = "cr10" %}  {# 3 = head-loop bound (4 heads) #}
{% set DSTRUCT       = "cr15" %}  {# reserved dstructure register #}

    # key_last = KEY_BOUND + 1 (= 15). MULT reads r_cyclic from the
    # start-of-cycle snapshot (issue #157), so each P key row is loaded a bundle
    # before the MULT consuming it and key_index's ADD moves into s_loop_pre
    # (the scalar operand is read LIVE from R0 and must name the key just
    # loaded). BLT reads the SNAPSHOT, so with that ADD a word earlier it sees an
    # already-incremented index and needs the higher bound, or the last key is
    # dropped.
    SET     {{ key_last }} {{ KEY_BOUND }};;
    ADD     {{ key_last }} {{ key_last }} {{ ONE }};;

    SET     {{ v_ptr }} {{ ZERO }};;                          # R0 source offset = chan*KEY_STRIDE = 0
    SET     {{ p_head_base }} {{ ZERO }};;                    # P head base = 0
    SET     {{ out_ptr }} {{ ZERO }};;                        # output offset = 0

    SET     {{ head_index }}  {{ ZERO }};;                    # head counter (0..3)
h_loop:

    SET     {{ chan_index }}  {{ ZERO }};;                    # channel (t) counter (0..59)
t_loop:

    # -- load V[:, chan] into R0 (all 16 keys fit in one row; no R1) ----------
    LDR_MULT_REG r0 {{ v_ptr }} {{ V_BASE }};;                # R0 = V[0:15, chan]  (VBASE + chan*KEY_STRIDE)

    # -- key loop: s = 0..15 over the single 16-query group -------------------
    # P[h,s] row = PBASE + h*P_HEAD_STRIDE + s*KEY_STRIDE.  p_ptr = start - KEY_STRIDE.
    SET     {{ p_ptr }}  {{ ZERO }};;                         # P row offset = 0
    ADD     {{ p_ptr }}  {{ p_ptr }}  {{ p_head_base }};;     # + h*P_HEAD_STRIDE -> P[h, s=0]
    SUB     {{ p_ptr }}  {{ p_ptr }}  {{ key_stride }};;      # - KEY_STRIDE startup (ADD -> live s=0)
    SET     {{ key_index }}  {{ KEY_START }};;                # key index startup = -1 (ADD +1 -> s=0)

    # Prime s=0's P row (the MULT below consumes it from the snapshot).
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    ADD {{ key_index }} {{ key_index }} {{ ONE }};;

    # Peeled first key (s=0): ACC.ADD.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT {{ key_index }} {{ key_last }} s_loop_pre;;
    B s_done;;
s_loop_pre:
    ADD     {{ key_index }} {{ key_index }} {{ ONE }};;       # name the key just loaded
s_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT {{ key_index }} {{ key_last }} s_loop_pre;;
s_done:

    # -- store channel-major row O[0:16, chan] --------------------------------
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_BASE }};;  # one whole FP32 row -> OBASE + chan*OUT_STRIDE
    ADD     {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;    # advance output-row offset

    ADD     {{ v_ptr }} {{ v_ptr }} {{ key_stride }};;        # next channel: R0 source += KEY_STRIDE
    ADD     {{ chan_index }}  {{ chan_index }}  {{ ONE }};
    BLT {{ chan_index }} {{ CHAN_BOUND }} t_loop;;

    ADD     {{ p_head_base }} {{ p_head_base }} {{ P_HEAD_STRIDE }};; # next head: P head base += stride
    ADD     {{ head_index }}  {{ head_index }}  {{ ONE }};
    BLT {{ head_index }} {{ HEAD_BOUND }} h_loop;;

end:
    BKPT;;
