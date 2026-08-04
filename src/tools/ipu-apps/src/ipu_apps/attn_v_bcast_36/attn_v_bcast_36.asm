# attn@V, key-major scores -> channel-major output  (Agent B, broadcast / no AGG)
#
# Per attention head h in [0,4):
#   O[i,t] = sum_s P[i,s] * V[s,t]
#     i = query in [0,256)  (2 groups of 128 = the SIMD lanes)
#     t = head channel in [0,36)   (global value channel = h*36 + t)
#     s = key (contraction) in [0,256)
#
# LANES = QUERIES.  Standard matmul-broadcast template (V in the weight/scalar
# role, P key-major in the streamed-data role):
#   mult_res[i] = P[i,s] * V[s,t]   ;  ACC[.FIRST] accumulates over s
#   After 256 keys, R_ACC[i] = O[i,t] for the 128 queries of this group.
# No AGG, no collision: every cycle is a full-width scalar*vector MULT + ACC, and
# all 256 values of channel (h,t) live in R0++R1, so the scalar index s (0..255)
# selects V[s,t] directly with NO mid-loop reload.
#
# Inputs (1 byte/element):
#   P key-major (head-major): P[i,s] at PBASE + h*65536 + s*256 + i.
#     Group g, key s -> 128 queries at PBASE + h*65536 + s*256 + g*128.
#   V channel-major: V[s, chan] at VBASE + chan*256 + s,  chan = h*36 + t.
#     Per channel the 256 values load into R0 (s=0..127) and R1 (s=128..255).
# Output (FP32 R_ACC, 512-byte group rows — transformer-matmul convention):
#   O[i,t] at OBASE + chan*1024 + g*512 + local*4,  i = g*128 + local.
#
# VLIW timing (same as matmul_144x144 inner loop):
#   LR slot runs first; XMEM + MULT then read LR values LIVE (post-ADD); BLT reads
#   the SNAPSHOT (pre-ADD).  Single-cycle inner body:
#     LDR_CYCLIC P[h,g,s_live]; ADD data ptr; ADD s; MULT.RC.VE src=s_live; ACC; BLT
#   Startup offsets (one-cycle pipeline align):
#     data ptr  p_ptr = first_addr - 256   (ADD +256 -> live = first_addr)
#     key index key_index = -1                  (ADD +1   -> live s = 0)
#     BLT bound KEY_BOUND = 254  (first_index=0, width=256 -> 0+256-2=254)
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
# pre-ADD snapshot; branch is taken while snapshot < bound):
#   KEY_BOUND  = 254          (key-loop bound: width 256, peeled + startup)
#   CHAN_BOUND  = 35           (t-loop bound: 36 channels per head)
#   HEAD_BOUND = 3            (head-loop bound: 4 heads)
#   GROUP_BOUND = 1            (g-loop bound: 2 groups)

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set rc_slot0      = "lr0"  %}  {# const 0: r_cyclic index / mask_shift #}
{% set key_stride    = "lr1"  %}  {# 256 = P key stride and V channel stride #}
{% set out_stride    = "lr2"  %}  {# 512 = output-row stride #}
{% set group_step    = "lr3"  %}  {# 128 = query-group offset within a key column #}
{% set p_ptr         = "lr4"  %}  {# byte offset into P (key-major) #}
{% set key_index     = "lr5"  %}  {# contraction key index s -> selects V[s,t] #}
{% set head_index    = "lr6"  %}  {# head counter h #}
{% set chan_index    = "lr7"  %}  {# head-channel counter t #}
{% set key_last      = "lr8"  %}  {# 255 = KEY_BOUND + 1; BLT bound now that key_index's ADD sits a word ahead of the branch #}
{% set group_index   = "lr9"  %}  {# query-group counter g #}
{% set v_ptr_lo      = "lr10" %}  {# chan*256      -> r0 = V[0:127,   chan] #}
{% set v_ptr_hi      = "lr11" %}  {# chan*256 +128 -> r1 = V[128:255, chan] #}
{% set p_head_base   = "lr12" %}  {# h*65536, P head base #}
{% set p_group_off   = "lr13" %}  {# g*128, P group offset #}
{% set out_ptr       = "lr14" %}  {# output byte offset, += 512 per store #}

{% set ZERO          = "cr0"  %}  {# hardwired 0; const-0 and r_cyclic XMEM base #}
{% set ONE           = "cr1"  %}  {# hardwired read-only 1 #}
{% set P_BASE        = "cr2"  %}  {# key-major scores base #}
{% set V_BASE        = "cr3"  %}  {# channel-major value base #}
{% set OUT_BASE      = "cr4"  %}  {# channel-major output base #}
{% set V_HI_OFF      = "cr5"  %}  {# 128 = r1 source offset within a V channel #}
{% set KEY_START     = "cr6"  %}  {# -1 -> first live s = 0 #}
{% set P_HEAD_STRIDE = "cr7"  %}  {# 65536 = P head stride #}
{% set KEY_BOUND     = "cr8"  %}  {# 254 = key-loop bound (width 256) #}
{% set CHAN_BOUND    = "cr9"  %}  {# 35 = t-loop bound (36 channels/head) #}
{% set HEAD_BOUND    = "cr10" %}  {# 3 = head-loop bound (4 heads) #}
{% set GROUP_BOUND   = "cr11" %}  {# 1 = g-loop bound (2 query groups) #}
{% set DSTRUCT       = "cr15" %}  {# reserved dstructure register #}

    # key_last = KEY_BOUND + 1 (= 255). MULT reads r_cyclic from the
    # start-of-cycle snapshot (issue #157), so each P key row is loaded a bundle
    # before the MULT consuming it and key_index's ADD moves into s_loop_pre
    # (MULT reads it LIVE and must name the key just loaded). BLT reads the
    # SNAPSHOT, so with that ADD a word earlier it sees an already-incremented
    # index and needs the higher bound, or the last key is dropped.
    SET     {{ key_last }} {{ KEY_BOUND }};;
    ADD     {{ key_last }} {{ key_last }} {{ ONE }};;

    SET     {{ v_ptr_lo }} {{ ZERO }};;                       # R0 source offset = chan*256 = 0
    SET     {{ v_ptr_hi }} {{ V_HI_OFF }};;                   # R1 source offset = chan*256 + 128 = 128
    SET     {{ p_head_base }} {{ ZERO }};;                    # P head base = 0
    SET     {{ out_ptr }} {{ ZERO }};;                        # output offset = 0

    SET     {{ head_index }}  {{ ZERO }};;                    # head counter (0..3)
h_loop:

    SET     {{ chan_index }}  {{ ZERO }};;                    # channel (t) counter (0..35)
t_loop:

    # -- load V[:, chan] into R0 (s=0..127) and R1 (s=128..255) ----------------
    LDR_MULT_REG r0 {{ v_ptr_lo }} {{ V_BASE }};;             # R0 = V[0:127,   chan]  (VBASE + chan*256)
    LDR_MULT_REG r1 {{ v_ptr_hi }} {{ V_BASE }};;             # R1 = V[128:255, chan]  (VBASE + chan*256 + 128)

    SET     {{ p_group_off }} {{ ZERO }};;                    # P group offset = g*128 = 0
    SET     {{ group_index }}  {{ ZERO }};;                   # g counter (0..1)
g_loop:

    # -- key loop: s = 0..255 over the 128-query group g -----------------------
    # P[h,g,s] address = PBASE + h*65536 + s*256 + g*128.  data ptr p_ptr = start-256.
    SET     {{ p_ptr }}  {{ P_BASE }};;                       # PBASE
    ADD     {{ p_ptr }}  {{ p_ptr }}  {{ p_head_base }};;     # + h*65536
    ADD     {{ p_ptr }}  {{ p_ptr }}  {{ p_group_off }};;     # + g*128  -> P[h,g,s=0]
    SUB     {{ p_ptr }}  {{ p_ptr }}  {{ key_stride }};;      # - 256 startup (ADD +256 -> live s=0)
    SET     {{ key_index }}  {{ KEY_START }};;                # key index startup = -1 (ADD +1 -> s=0)

    # Prime s=0's P row (the MULT below consumes it from the snapshot).
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }}; ADD {{ key_index }} {{ key_index }} {{ ONE }};;

    # Peeled first key (s=0): ACC.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT {{ key_index }} {{ key_last }} s_loop_pre;;
    B s_done;;
s_loop_pre:
    ADD     {{ key_index }} {{ key_index }} {{ ONE }};;       # name the key just loaded
s_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }}; ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ ZERO }} {{ rc_slot0 }}; ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT {{ key_index }} {{ key_last }} s_loop_pre;;
s_done:

    # -- store channel-major row O[g*128:+128, chan] ---------------------------
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};;
    STR_POST_AAQ_REG {{ out_ptr }} {{ OUT_BASE }};;                # 512B FP32 -> OBASE + chan*1024 + g*512
    ADD     {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;    # advance output-row offset by 512

    ADD     {{ p_group_off }} {{ p_group_off }} {{ group_step }};; # next group: P offset += 128
    ADD     {{ group_index }}  {{ group_index }}  {{ ONE }}; BLT {{ group_index }} {{ GROUP_BOUND }} g_loop;;

    ADD     {{ v_ptr_lo }} {{ v_ptr_lo }} {{ key_stride }};;  # next channel: R0 source += 256
    ADD     {{ v_ptr_hi }} {{ v_ptr_hi }} {{ key_stride }};;  # next channel: R1 source += 256
    ADD     {{ chan_index }}  {{ chan_index }}  {{ ONE }}; BLT {{ chan_index }} {{ CHAN_BOUND }} t_loop;;

    ADD     {{ p_head_base }} {{ p_head_base }} {{ P_HEAD_STRIDE }};; # next head: P head base += 65536
    ADD     {{ head_index }}  {{ head_index }}  {{ ONE }}; BLT {{ head_index }} {{ HEAD_BOUND }} h_loop;;

end:
    BKPT;;
