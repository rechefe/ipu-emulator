# Agent A — attn@V (AGG kernel), query-major scores -> channel-major output.
#
# Per head (4 heads, head_dim D=36, N=256 tokens):
#   O[i, t] = sum_s P[i, s] * V[s, t]      i,s in [0,256), t in [0,36)
#
# Layouts (channel-major activations, byte addresses):
#   P query-major  : P[i, s] at PBASE + h*65536 + i*256 + s
#                    (query i's 256 scores contiguous -> two 128-key chunks)
#   V channel-major: V[s, t] at VBASE + (h*36 + t)*256 + s
#                    (value channel's 256 keys contiguous -> two 128-key chunks)
#   O channel-major: O[i, t] at OBASE + (h*36 + t)*256 + i
#
# Compute model (post-merge AGG, commit d67c441):
#   Lanes = keys.  MULT.RC.VV: mult_res[s] = P[i,s] * V[s,t]  (element-wise,
#   R0 = P chunk, R_CYCLIC = V chunk).  AGG.SUM[.FIRST] dest=local reduces the
#   128 live mult_res lanes into R_ACC[local] -- collision-free, no ACC, no reset.
#   chunk 0 -> AGG.SUM.FIRST (clean write, keys 0..127 partial);
#   chunk 1 -> AGG.SUM       (snapshot R_ACC[local] from chunk 0 + keys 128..255).
#   V chunk reused across all 128 queries of a group; only P (R0) reloads.
#
# Loop nest:  for h in 0..3 { for t in 0..35 { for g in 0..1 {
#                 chunk 0 (keys 0..127):   for local 0..127: load P; MULT; AGG.SUM.FIRST
#                 chunk 1 (keys 128..255): for local 0..127: load P; MULT; AGG.SUM
#                 STR_ACC_REG -> O[g*128 .. g*128+127, t]  (512B FP32 column segment)
#             }}}
#
# Addressing is split offset(LR) + base(CR):
#   chan offset chan_off advances by 256 monotonically across all 144 (h,t) steps
#     (hc = h*36 + t runs 0..143 contiguously) -> single running V/O channel ptr.
#   head P offset head_p_off advances by 65536 per head.
#
# Inner-loop pipeline (live-read offsets, per matmul template):
#   LDR_MULT_REG offset is read LIVE -> ADD p_ptr +256 co-issues; start p_ptr at
#   (first P row addr - 256) so the first live load hits the first row.
#   AGG dest_slot is read from SNAPSHOT; INC agg_slot co-issues -> AGG sees 0,1,..,127.
#   BLT reads SNAPSHOT -> bound 127 gives exactly 128 bundles (dest 0..127).
#
# Output O is FP32 (R_ACC, 4 bytes/elem) stored as 512-byte group rows, like the
# transformer matmuls.  So O addressing uses FP32 strides (separate from the
# 1-byte input strides): O channel offset advances 1024/step (=2 groups*512),
# group g=1 is at +512.  Inputs P,V stay 1 byte/elem (256 stride, 128 chunk).
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
{% set p_ptr            = "lr0"  %}  {# P inner load offset (live +256 per query) #}
{% set v_ptr            = "lr1"  %}  {# V chunk offset within the current channel #}
{% set rc_slot0         = "lr2"  %}  {# const 0: r_cyclic index #}
{% set agg_slot         = "lr3"  %}  {# AGG dest slot = inner query counter 0..127 #}
{% set chan_off         = "lr4"  %}  {# value-channel offset, += 256 per (h,t) step #}
{% set out_chunk        = "lr5"  %}  {# O byte offset for the current group #}
{% set head_p_off       = "lr6"  %}  {# head P offset, += 65536 per head #}
{% set chan_index       = "lr7"  %}  {# head-channel counter t #}
{% set head_index       = "lr8"  %}  {# head counter h #}
{% set group_p_off      = "lr9"  %}  {# group P offset (head base, then + g*32768) #}
{% set out_chan_off     = "lr10" %}  {# O channel offset, += 1024 per (h,t) step #}
{% set p_query_stride   = "lr11" %}  {# 256 = P query stride (live add) #}

{% set ZERO             = "cr0"  %}  {# const 0 #}
{% set ONE              = "cr1"  %}  {# hardwired read-only 1 #}
{% set P_BASE           = "cr2"  %}  {# query-major P base #}
{% set V_BASE           = "cr3"  %}  {# channel-major V base #}
{% set OUT_BASE         = "cr4"  %}  {# channel-major O base #}
{% set P_QUERY_STRIDE   = "cr5"  %}  {# 256 = P query stride #}
{% set CHUNK_OFF        = "cr6"  %}  {# 128 = second key-chunk offset #}
{% set P_GROUP_STRIDE   = "cr7"  %}  {# 32768 = 128 queries x 256 (group g=1) #}
{% set P_HEAD_STRIDE    = "cr8"  %}  {# 65536 = P head stride #}
{% set INNER_BOUND      = "cr9"  %}  {# 127 = inner bound -> 128 AGG bundles #}
{% set CHAN_COUNT       = "cr10" %}  {# 36 = head channels t #}
{% set HEAD_COUNT       = "cr11" %}  {# 4 = heads #}
{% set OUT_GROUP_STRIDE = "cr12" %}  {# 512 = O group stride #}
{% set OUT_CHAN_STRIDE  = "cr13" %}  {# 1024 = O channel stride #}
{% set DSTRUCT          = "cr15" %}  {# reserved dstructure register #}

    SET {{ rc_slot0 }} {{ ZERO }};;                                # rc write index = 0 (const)
    SET {{ chan_off }} {{ ZERO }};;                                # value-channel offset = 0
    SET {{ head_p_off }} {{ ZERO }};;                              # head P offset = 0
    SET {{ head_index }} {{ ZERO }};;                              # head counter = 0
    SET {{ out_chan_off }} {{ ZERO }};;                            # O channel offset = 0
    SET {{ p_query_stride }} {{ P_QUERY_STRIDE }};;                # P query stride = 256

head_loop:
    SET {{ chan_index }} {{ ZERO }};;                              # t counter = 0
    ADD {{ group_p_off }} {{ head_p_off }} {{ ZERO }};;            # group P offset = head P offset (g=0)

t_loop:
    # ===================== group g = 0 (queries 0..127) =====================
    # ---- chunk 0: keys 0..127 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ ZERO }};;                    # V chunk0 offset = chan
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;  # R_CYCLIC = V[0..127, t]   (base VBASE)
    SUB {{ p_ptr }} {{ group_p_off }} {{ p_query_stride }};;       # P inner start = group P off - 256
    SET {{ agg_slot }} {{ ZERO }};;                                # dest/inner counter = 0
g0c0_loop:
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }}; ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }}; INC {{ agg_slot }} 1; MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }}; AGG.SUM.FIRST {{ agg_slot }} {{ DSTRUCT }}; BLT {{ agg_slot }} {{ INNER_BOUND }} g0c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ CHUNK_OFF }};;               # V chunk1 offset = chan + 128
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;  # R_CYCLIC = V[128..255, t]
    ADD {{ p_ptr }} {{ group_p_off }} {{ CHUNK_OFF }};;            # P chunk1 base = group P off + 128
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 256 startup
    SET {{ agg_slot }} {{ ZERO }};;
g0c1_loop:
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }}; ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }}; INC {{ agg_slot }} 1; MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }}; AGG.SUM {{ agg_slot }} {{ DSTRUCT }}; BLT {{ agg_slot }} {{ INNER_BOUND }} g0c1_loop;;

    ADD {{ out_chunk }} {{ out_chan_off }} {{ ZERO }};;            # O g=0 offset = O chan offset
    STR_ACC_REG {{ out_chunk }} {{ OUT_BASE }};;                   # O[0..127, t] = R_ACC   (base OBASE)

    # ===================== group g = 1 (queries 128..255) ===================
    # ---- chunk 0: keys 0..127 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ ZERO }};;                    # V chunk0 offset = chan (same channel)
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;
    ADD {{ p_ptr }} {{ group_p_off }} {{ P_GROUP_STRIDE }};;       # g=1 P base = group P off + 32768
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 256 startup
    SET {{ agg_slot }} {{ ZERO }};;
g1c0_loop:
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }}; ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }}; INC {{ agg_slot }} 1; MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }}; AGG.SUM.FIRST {{ agg_slot }} {{ DSTRUCT }}; BLT {{ agg_slot }} {{ INNER_BOUND }} g1c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ CHUNK_OFF }};;               # V chunk1 offset = chan + 128
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;
    ADD {{ p_ptr }} {{ group_p_off }} {{ P_GROUP_STRIDE }};;       # g=1 P base
    ADD {{ p_ptr }} {{ p_ptr }} {{ CHUNK_OFF }};;                  # + 128 (chunk1)
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 256 startup
    SET {{ agg_slot }} {{ ZERO }};;
g1c1_loop:
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }}; ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }}; INC {{ agg_slot }} 1; MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }}; AGG.SUM {{ agg_slot }} {{ DSTRUCT }}; BLT {{ agg_slot }} {{ INNER_BOUND }} g1c1_loop;;

    ADD {{ out_chunk }} {{ out_chan_off }} {{ OUT_GROUP_STRIDE }};; # O g=1 offset = O chan offset + 512
    STR_ACC_REG {{ out_chunk }} {{ OUT_BASE }};;                   # O[128..255, t] = R_ACC

    # ----- next t: advance value-channel offset (+256 in) and O offset (+1024), t++ -----
    ADD {{ chan_off }} {{ chan_off }} {{ P_QUERY_STRIDE }};;       # chan += 256
    ADD {{ out_chan_off }} {{ out_chan_off }} {{ OUT_CHAN_STRIDE }};; # O chan offset += 1024
    INC {{ chan_index }} 1;;                                       # t++
    BLT {{ chan_index }} {{ CHAN_COUNT }} t_loop;;

    # ----- next head: head P offset += 65536, head++ -----
    ADD {{ head_p_off }} {{ head_p_off }} {{ P_HEAD_STRIDE }};;    # head P offset += 65536
    INC {{ head_index }} 1;;
    BLT {{ head_index }} {{ HEAD_COUNT }} head_loop;;

end:
    BKPT;;
