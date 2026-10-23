{#- attn_v_bcast_60.asm -- attn@V (Layer 5), key-major scores -> channel-major output (broadcast / no AGG)

    Layer:   L5
    Scope:   single-stream
    Layout:  unpacked
    Shape:   16tok x 60chan (head_dim), 4 heads
    Related: L5 variant of attn_v_bcast_36 (L3); attn_v_bcast_48 is the L4 variant;
             consumes attn_scores_km_16x60's key-major output; attn_v_16x60
             is the query-major (AGG) sibling for the same layer
    Tests:   //kernels/attention/attn_v_bcast_60; the family's test_datapath.py
             holds it bit for bit to the ACC datapath mirror

    Per attention head h in [0,4):
      O[i,t] = sum_s P[i,s] * V[s,t]
        i = query in [0,16)          (the SIMD lanes; ONE group at L5)
        t = head channel in [0,60)   (global value channel = h*60 + t)
        s = key (contraction) in [0,16)

    L5 variant of attn_v_bcast_36. The MAPPING is unchanged -- LANES = QUERIES, the
    standard matmul-broadcast template with V in the weight/scalar role and P
    key-major in the streamed-data role:
      MULT_RES[i] = P[i,s] * V[s,t]  ;  ACC.ADD[.FIRST] accumulates over s
      After 16 keys, R_ACC[i] = O[i,t] for the 16 queries.
    No AGG, no collision: every cycle is a full-width scalar*vector MULT + ACC.
    Only the counts move from L3:
      * D 36 -> 60 head channels (t-loop).
      * N_TOK 256 -> 16. At L3 the 256 queries needed TWO 128-lane groups (a
        g-loop) and a channel's 256 key values needed BOTH R0 (s=0..127) and R1
        (s=128..255). At L5 all 16 queries fit in ONE group and a channel's 16
        key values fit entirely in R0, so the g-loop AND the R1 load are gone.

    ONE CHANNEL PER ROW: every P key row, every V channel row and every stored O
    row is a WHOLE row with 16 live FP32 lanes. CR15.valid_elements = 16
    tells ACTIVATE.EXACT how many lanes to emit; the rest stay zero.

    Layouts (row addresses; XMEM operands are ROW numbers, issue #179).  V and O
    match the query-major sibling attn_v_16x60 EXACTLY, so the two chains'
    outputs are directly comparable:
      P key-major (head-major): P[i,s] at PBASE + h*P_HEAD_STRIDE + s*KEY_STRIDE
        (key s's 16 query scores = one row, query i in lane i).
      V channel-major: V[s, chan] at VBASE + chan*KEY_STRIDE, chan = h*60 + t
        (the channel's 16 key values = one row, all of them loaded into R0).
      O channel-major: O[i,t] at OBASE + chan*OUT_STRIDE, lane i.

    VLIW timing:
      LR slot runs first; XMEM + MULT then read LR values forwarded (post-ADD); BLT.S reads
      the cycle-start value (pre-ADD).  Single-cycle inner body:
        LD.MULT.CYCLIC P[h,s_live]; ADD data ptr; MULT.RC.VE src=s_live; ACC.ADD; BLT.S
      Startup offsets (one-cycle pipeline align):
        data ptr  p_ptr = first_row - KEY_STRIDE   (ADD -> live = first_row)
        key index key_index = -1                   (ADD +1 -> live s = 0)
        BLT.S bound KEY_BOUND = 14  (first_index=0, width=16 -> 0+16-2=14)

    Registers are referred to below by the symbolic names defined in the
    register-name block. The assembler's Jinja2 preprocessor substitutes them
    before parsing, so the emitted binary is byte-identical to the raw form.
    NOTE: Jinja runs before comment stripping, so '#' comments must not
    contain Jinja delimiters -- the preprocessor would try to execute them.

    The register-name block below names every register. The ones the harness
    sets are kernel.toml's [abi] bindings (cr.<name>, lr.<name>); CR0 and CR1
    are the hardwired 0 and 1, and the other LRs are the program's own.

    Loop bounds (BLT.S reads the pre-ADD cycle-start value; branch taken while cycle start < bound):
      KEY_BOUND   = 14          (key-loop bound: width 16, peeled + startup)
      CHAN_BOUND  = 59          (t-loop bound: 60 channels per head)
      HEAD_BOUND  = 3           (head-loop bound: 4 heads)

    ---------------------------------------------------------------------------
    Register names (Jinja2 preprocessor; pure source-level substitution)
    ---------------------------------------------------------------------------
-#}
{% set rc_slot0      = lr.rc_slot0  %}  {# const 0: R_CYCLIC index / mask_shift #}
{% set key_stride    = lr.pv_stride  %}  {# P key stride and V channel stride (rows) #}
{% set out_stride    = lr.out_row_stride  %}  {# output-row stride (rows) #}
{% set p_ptr         = "LR4"  %}  {# row offset into P (key-major) #}
{% set key_index     = "LR5"  %}  {# contraction key index s -> selects V[s,t] in R0 #}
{% set head_index    = "LR6"  %}  {# head counter h #}
{% set chan_index    = "LR7"  %}  {# head-channel counter t #}
{% set key_last      = "LR8"  %}  {# 15 = KEY_BOUND + 1; BLT.S bound now that key_index's ADD sits a word ahead of the branch #}
{% set v_ptr         = "LR10" %}  {# chan row offset -> R0 = V[0:15, chan] #}
{% set p_head_base   = "LR12" %}  {# h*P_HEAD_STRIDE, P head base (rows) #}
{% set out_ptr       = "LR14" %}  {# output row offset, += out_stride per store #}

{% set ZERO          = "CR0"  %}  {# hardwired 0; const-0 and R_CYCLIC XMEM base #}
{% set ONE           = "CR1"  %}  {# hardwired read-only 1 #}
{% set P_BASE        = cr.b  %}  {# key-major scores base #}
{% set V_BASE        = cr.a  %}  {# channel-major value base #}
{% set OUT_BASE      = cr.y  %}  {# channel-major output base #}
{% set KEY_START     = cr.key_start  %}  {# -1 -> first live s = 0 #}
{% set P_HEAD_STRIDE = cr.p_head_stride  %}  {# P head stride (rows) #}
{% set KEY_BOUND     = cr.key_bound  %}  {# 14 = key-loop bound (width 16) #}
{% set CHAN_BOUND    = cr.chan_bound  %}  {# 59 = t-loop bound (60 channels/head) #}
{% set HEAD_BOUND    = cr.head_bound %}  {# 3 = head-loop bound (4 heads) #}
{% set DSTRUCT       = cr.dstruct ~ ".L" %}  {# the DS half #}

    # key_last = KEY_BOUND + 1 (= 15). MULT reads R_CYCLIC from the
    # cycle-start value (issue #157), so each P key row is loaded an instruction word
    # before the MULT consuming it and key_index's ADD moves into s_loop_pre
    # (the scalar operand is read forwarded from R0 and must name the key just
    # loaded). BLT.S reads the cycle-start value, so with that ADD a word earlier it sees an
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
    LD.MULT R0 {{ v_ptr }} {{ V_BASE }};;                # R0 = V[0:15, chan]  (VBASE + chan*KEY_STRIDE)

    # -- key loop: s = 0..15 over the single 16-query group -------------------
    # P[h,s] row = PBASE + h*P_HEAD_STRIDE + s*KEY_STRIDE.  p_ptr = start - KEY_STRIDE.
    SET     {{ p_ptr }}  {{ ZERO }};;                         # P row offset = 0
    ADD     {{ p_ptr }}  {{ p_ptr }}  {{ p_head_base }};;     # + h*P_HEAD_STRIDE -> P[h, s=0]
    SUB     {{ p_ptr }}  {{ p_ptr }}  {{ key_stride }};;      # - KEY_STRIDE startup (ADD -> live s=0)
    SET     {{ key_index }}  {{ KEY_START }};;                # key index startup = -1 (ADD +1 -> s=0)

    # Prime s=0's P row (the MULT below consumes it from the cycle-start value).
    LD.MULT.CYCLIC {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    ADD {{ key_index }} {{ key_index }} {{ ONE }};;

    # Peeled first key (s=0): ACC.ADD.FIRST seeds R_ACC.
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT.S {{ key_index }} {{ key_last }} s_loop_pre;;
    B s_done;;
s_loop_pre:
    ADD     {{ key_index }} {{ key_index }} {{ ONE }};;       # name the key just loaded
s_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT.S {{ key_index }} {{ key_last }} s_loop_pre;;
s_done:

    # -- store channel-major row O[0:16, chan] --------------------------------
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ OUT_BASE }} {{ out_ptr }} ;;  # one whole FP32 row -> OBASE + chan*OUT_STRIDE
    ADD     {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;    # advance output-row offset

    ADD     {{ v_ptr }} {{ v_ptr }} {{ key_stride }};;        # next channel: R0 source += KEY_STRIDE
    ADD     {{ chan_index }}  {{ chan_index }}  {{ ONE }};
    BLT.S {{ chan_index }} {{ CHAN_BOUND }} t_loop;;

    ADD     {{ p_head_base }} {{ p_head_base }} {{ P_HEAD_STRIDE }};; # next head: P head base += stride
    ADD     {{ head_index }}  {{ head_index }}  {{ ONE }};
    BLT.S {{ head_index }} {{ HEAD_BOUND }} h_loop;;

end:
    HALT;;
