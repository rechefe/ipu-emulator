{#- attn_v_bcast_36.asm -- attn@V, key-major scores -> channel-major output (broadcast, no AGG, FP32)

    Layer:   L3
    Scope:   4 heads, one stream
    Layout:  unpacked
    Shape:   256tok (2 groups x 128) x 36chan (head_dim), 4 heads
    Related: attn_v_bcast_48 (L4) / attn_v_bcast_60 (L5) are ports of this
             kernel; consumes attn_scores_km_256x36's key-major output;
             attn_v_256x36 is the query-major (AGG) sibling for the same layer
    Tests:   //kernels/attention/attn_v_bcast_36; the family's test_datapath.py
             holds it bit for bit to the ACC datapath mirror

    Per attention head h in [0,4):
      O[i,t] = sum_s P[i,s] * V[s,t]
        i = query in [0,256)  (2 groups of 128 = the SIMD lanes)
        t = head channel in [0,36)   (global value channel = h*36 + t)
        s = key (contraction) in [0,256)

    LANES = QUERIES.  Standard matmul-broadcast template (V in the weight/scalar
    role, P key-major in the streamed-data role):
      MULT_RES[i] = P[i,s] * V[s,t]   ;  ACC[.FIRST] accumulates over s
      After 256 keys, R_ACC[i] = O[i,t] for the 128 queries of this group.
    No AGG, no collision: every cycle is a full-width scalar*vector MULT + ACC, and
    all 256 values of channel (h,t) live in R0++R1, so the scalar index s (0..255)
    selects V[s,t] directly with NO mid-loop reload.

    All XMEM operands are ROW numbers; one row is 128 FP32 lanes. The harness
    places and packs the regions (kernel.toml: b = P, a = V, y = O):
      P key-major (head-major): key s of head h is two rows, queries
        g*128..g*128+127 in row PBASE + h*512 + 2s + g.
      V channel-major: channel chan = h*36 + t is two rows, keys s = 0..127 in
        row VBASE + 2*chan (loaded into R0) and s = 128..255 in the next (R1).
    Output, channel-major: O[g*128 + local, chan] in lane `local` of row
      OBASE + 2*chan + g, one ACTIVATE.EXACT store per row.

    Timing (same as matmul_144x144's inner loop):
      LR slot runs first; XMEM + MULT then read LR values forwarded (post-ADD); BLT.S reads
      the cycle-start value (pre-ADD).  Single-cycle inner body:
        LD.MULT.CYCLIC P[h,g,s_live]; ADD data ptr; ADD s; MULT.RC.VE src=s_live; ACC; BLT.S
      Startup offsets (one-cycle pipeline align):
        data ptr  p_ptr = first_row - 2      (ADD +2 -> live = first_row)
        key index key_index = -1                  (ADD +1   -> live s = 0)
        BLT.S bound KEY_BOUND = 254  (first_index=0, width=256 -> 0+256-2=254)

    Registers are referred to below by the symbolic names defined in the
    register-name block. The assembler's Jinja2 preprocessor substitutes them
    before parsing, so the emitted binary is byte-identical to the raw form.
    NOTE: Jinja runs before comment stripping, so '#' comments must not
    contain Jinja delimiters -- the preprocessor would try to execute them.

    The register-name block below names every register. The ones the harness
    sets are kernel.toml's [abi] bindings (cr.<name>, lr.<name>); CR0 and CR1
    are the hardwired 0 and 1, and the other LRs are the program's own.

    Loop bounds (BLT.S reads the pre-ADD cycle-start value; the branch is taken
    while it is below the bound):
      KEY_BOUND  = 254          (key-loop bound: width 256, peeled + startup)
      CHAN_BOUND  = 35           (t-loop bound: 36 channels per head)
      HEAD_BOUND = 3            (head-loop bound: 4 heads)
      GROUP_BOUND = 1            (g-loop bound: 2 groups)

    ---------------------------------------------------------------------------
    Register names (Jinja2 preprocessor; pure source-level substitution)
    ---------------------------------------------------------------------------
-#}
{% set rc_slot0      = lr.rc_slot0  %}  {# const 0: R_CYCLIC index / mask_shift #}
{% set key_stride    = lr.pv_stride  %}  {# 2 = P key stride and V channel stride (rows) #}
{% set out_stride    = lr.out_row_stride  %}  {# 1 = output-row stride (rows) #}
{% set group_step    = lr.group_query_off  %}  {# 1 = query-group offset within a key (rows) #}
{% set p_ptr         = "LR4"  %}  {# row offset into P (key-major) #}
{% set key_index     = "LR5"  %}  {# contraction key index s -> selects V[s,t] #}
{% set head_index    = "LR6"  %}  {# head counter h #}
{% set chan_index    = "LR7"  %}  {# head-channel counter t #}
{% set key_last      = "LR8"  %}  {# 255 = KEY_BOUND + 1; BLT.S bound now that key_index's ADD sits a word ahead of the branch #}
{% set group_index   = "LR9"  %}  {# query-group counter g #}
{% set v_ptr_lo      = "LR10" %}  {# 2*chan       -> R0 = V[0:127,   chan] #}
{% set v_ptr_hi      = "LR11" %}  {# 2*chan + 1   -> R1 = V[128:255, chan] #}
{% set p_head_base   = "LR12" %}  {# h*512, P head base (rows) #}
{% set p_group_off   = "LR13" %}  {# g, P group offset (rows) #}
{% set out_ptr       = "LR14" %}  {# output row offset, += 1 per store #}

{% set ZERO          = "CR0"  %}  {# hardwired 0; const-0 and the R_CYCLIC load base #}
{% set ONE           = "CR1"  %}  {# hardwired read-only 1 #}
{% set P_BASE        = cr.b  %}  {# key-major scores base #}
{% set V_BASE        = cr.a  %}  {# channel-major value base #}
{% set OUT_BASE      = cr.y  %}  {# channel-major output base #}
{% set V_HI_OFF      = cr.v_hi_off  %}  {# 1 = R1 source offset within a V channel (rows) #}
{% set KEY_START     = cr.key_start  %}  {# -1 -> first live s = 0 #}
{% set P_HEAD_STRIDE = cr.p_head_stride  %}  {# 512 = P head stride (rows) #}
{% set KEY_BOUND     = cr.key_bound  %}  {# 254 = key-loop bound (width 256) #}
{% set CHAN_BOUND    = cr.chan_bound  %}  {# 35 = t-loop bound (36 channels/head) #}
{% set HEAD_BOUND    = cr.head_bound %}  {# 3 = head-loop bound (4 heads) #}
{% set GROUP_BOUND   = cr.group_bound %}  {# 1 = g-loop bound (2 query groups) #}
{% set DSTRUCT       = "CR15.L" %}  {# the DS half #}

    # key_last = KEY_BOUND + 1 (= 255). MULT reads R_CYCLIC from the
    # cycle-start value (issue #157), so each P key row is loaded an instruction word
    # before the MULT consuming it and key_index's ADD moves into s_loop_pre
    # (MULT reads it forwarded and must name the key just loaded). BLT.S reads the
    # cycle-start value, so with that ADD a word earlier it sees an already-incremented
    # index and needs the higher bound, or the last key is dropped.
    SET     {{ key_last }} {{ KEY_BOUND }};;
    ADD     {{ key_last }} {{ key_last }} {{ ONE }};;

    SET     {{ v_ptr_lo }} {{ ZERO }};;                       # R0 source offset = 2*chan = 0 rows
    SET     {{ v_ptr_hi }} {{ V_HI_OFF }};;                   # R1 source offset = 2*chan + 1 = 1 row
    SET     {{ p_head_base }} {{ ZERO }};;                    # P head base = 0
    SET     {{ out_ptr }} {{ ZERO }};;                        # output offset = 0

    SET     {{ head_index }}  {{ ZERO }};;                    # head counter (0..3)
h_loop:

    SET     {{ chan_index }}  {{ ZERO }};;                    # channel (t) counter (0..35)
t_loop:

    # -- load V[:, chan] into R0 (s=0..127) and R1 (s=128..255) ----------------
    LD.MULT R0 {{ v_ptr_lo }} {{ V_BASE }};;             # R0 = V[0:127,   chan]  (row VBASE + 2*chan)
    LD.MULT R1 {{ v_ptr_hi }} {{ V_BASE }};;             # R1 = V[128:255, chan]  (the next row)

    SET     {{ p_group_off }} {{ ZERO }};;                    # P group offset = g = 0
    SET     {{ group_index }}  {{ ZERO }};;                   # g counter (0..1)
g_loop:

    # -- key loop: s = 0..255 over the 128-query group g -----------------------
    # P[h,g,s] row = PBASE + h*512 + 2s + g.  data ptr p_ptr = start - 2.
    SET     {{ p_ptr }}  {{ P_BASE }};;                       # PBASE
    ADD     {{ p_ptr }}  {{ p_ptr }}  {{ p_head_base }};;     # + h*512
    ADD     {{ p_ptr }}  {{ p_ptr }}  {{ p_group_off }};;     # + g  -> P[h,g,s=0]
    SUB     {{ p_ptr }}  {{ p_ptr }}  {{ key_stride }};;      # - 2 startup (ADD +2 -> live s=0)
    SET     {{ key_index }}  {{ KEY_START }};;                # key index startup = -1 (ADD +1 -> s=0)

    # Prime s=0's P row (the MULT below consumes it from the cycle-start value).
    LD.MULT.CYCLIC {{ p_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    ADD {{ key_index }} {{ key_index }} {{ ONE }};;

    # Peeled first key (s=0): ACC.ADD.FIRST seeds R_ACC.
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC {{ p_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT.S {{ key_index }} {{ key_last }} s_loop_pre;;
    B s_done;;
s_loop_pre:
    ADD     {{ key_index }} {{ key_index }} {{ ONE }};;       # name the key just loaded
s_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }} {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC {{ p_ptr }} {{ ZERO }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT.S {{ key_index }} {{ key_last }} s_loop_pre;;
s_done:

    # -- store channel-major row O[g*128:+128, chan] ---------------------------
    ACTIVATE.EXACT identity {{ DSTRUCT }} {{ OUT_BASE }} {{ out_ptr }} ;;  # a row -> OBASE + 2*chan + g
    ADD     {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;    # advance the output row by 1

    ADD     {{ p_group_off }} {{ p_group_off }} {{ group_step }};; # next group: P offset += 1 row
    ADD     {{ group_index }}  {{ group_index }}  {{ ONE }};
    BLT.S {{ group_index }} {{ GROUP_BOUND }} g_loop;;

    ADD     {{ v_ptr_lo }} {{ v_ptr_lo }} {{ key_stride }};;  # next channel: R0 source += 2 rows
    ADD     {{ v_ptr_hi }} {{ v_ptr_hi }} {{ key_stride }};;  # next channel: R1 source += 2 rows
    ADD     {{ chan_index }}  {{ chan_index }}  {{ ONE }};
    BLT.S {{ chan_index }} {{ CHAN_BOUND }} t_loop;;

    ADD     {{ p_head_base }} {{ p_head_base }} {{ P_HEAD_STRIDE }};; # next head: P head base += 512 rows
    ADD     {{ head_index }}  {{ head_index }}  {{ ONE }};
    BLT.S {{ head_index }} {{ HEAD_BOUND }} h_loop;;

end:
    HALT;;
