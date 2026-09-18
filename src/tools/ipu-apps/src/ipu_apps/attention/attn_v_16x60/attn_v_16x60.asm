# attn@V (AGG kernel, Layer 5), query-major scores -> channel-major output.
#
# Layer:   L5
# Scope:   single-stream
# Layout:  unpacked
# Shape:   16tok x 60chan (head_dim), 4 heads
# Status:  validated
# Related: L5 port of attn_v_256x36 (L3) / attn_v_64x48 (L4); consumes
#          qk_scores_16x60's query-major output; attn_v_bcast_60 is the
#          key-major (no-AGG) sibling for the same layer
# Tests:   test_attn_v_16x60_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Per head (4 heads, head_dim D=60, N=16 tokens):
#   O[i, t] = sum_s P[i, s] * V[s, t]      i,s in [0,16), t in [0,60)
#
# L5 port of attn_v_256x36. The MAPPING is unchanged -- lanes are KEYS, P is
# query-major in R0, V is channel-major in R_CYCLIC, MULT.RC.VV multiplies them
# element-wise and AGG.SUM reduces the live lanes into one R_ACC slot. Only the
# counts move:
#   * D 36 -> 60 head channels (outer loop).
#   * N_TOK 256 -> 16. At L3 the 256 keys needed TWO 128-lane chunks
#     (AGG.SUM.FIRST for keys 0..127 then AGG.SUM for keys 128..255) and the 256
#     queries needed TWO 128-slot R_ACC groups -- four inner loops in all. At L5
#     all 16 keys fit in ONE chunk and all 16 queries in ONE R_ACC store, so a
#     SINGLE AGG.SUM.FIRST loop of 16 bundles produces a whole output row.
#     There is no second chunk, hence no AGG.SUM and no cross-chunk partial.
#
# ONE CHANNEL PER ROW: every P query row and every V channel row is a WHOLE
# 512-B row with 16 live FP32 lanes. cr15.valid_elements = 16 tells AGG how many
# lanes to reduce and ACTIVATE.QUANTIZE how many lanes to emit.
#
# Layouts (row addresses; XMEM operands are ROW numbers, issue #179):
#   P query-major  : P[i, s] at PBASE + h*P_HEAD_STRIDE + i*PV_STRIDE
#                    (query i's 16 scores = one row)
#   V channel-major: V[s, t] at VBASE + (h*60 + t)*PV_STRIDE
#                    (value channel's 16 keys = one row)
#   O channel-major: O[i, t] at OBASE + (h*60 + t)*O_CHAN_ROWS, lane i
#
# Compute model:
#   Lanes = keys.  MULT.RC.VV: mult_res[s] = P[i,s] * V[s,t]  (element-wise,
#   R0 = P row, R_CYCLIC = V row).  AGG.SUM.FIRST dest=local reduces the
#   16 live mult_res lanes into R_ACC[local] -- collision-free, no ACC, no reset.
#   V row is reused across all 16 queries; only P (R0) reloads.
#
# Loop nest:  for h in 0..3 { for t in 0..59 {
#                 for local 0..15: load P; MULT.RC.VV; AGG.SUM.FIRST
#                 ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG
#                   -> O[0..15, t]  (one whole row, 16 live FP32 lanes)
#             }}
#
# Addressing is split offset(LR) + base(CR):
#   chan offset chan_off advances by PV_STRIDE monotonically across all 240
#     (h,t) steps (hc = h*60 + t runs 0..239 contiguously) -> single running
#     V channel ptr, and out_chan_off likewise for O.
#   head P offset head_p_off advances by P_HEAD_STRIDE per head.
#
# Inner-loop pipeline (live-read offsets, per matmul template):
#   LDR_MULT_REG offset is read LIVE -> ADD p_ptr co-issues; start p_ptr at
#   (first P row - PV_STRIDE) so the first live load hits the first row.
#   AGG dest_slot is read from SNAPSHOT; INC agg_slot co-issues -> AGG sees
#   0,1,..,15.  BLT reads SNAPSHOT -> bound 15 gives exactly 16 bundles.
#
#   MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VV reads its R0 DATA from the
#   start-of-cycle snapshot, so it cannot consume a P row loaded by
#   LDR_MULT_REG in its own bundle -- it would see the previous R0. The inner
#   loop therefore primes query 0's P row before the loop, and every bundle
#   multiplies the row loaded last cycle while prefetching the next query's.
#   The V row in R_CYCLIC is loaded by LDR_CYCLIC_MULT_REG two bundles ahead of
#   the loop and never reloaded inside it, so it was already snapshot-safe.
#   agg_slot is unaffected -- both of its readers (AGG and BLT) take the
#   SNAPSHOT, so INC stays co-issued and the dest sequence is still 0..15.
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
{% set p_ptr            = "lr0"  %}  {# P inner load offset (live += PV_STRIDE per query) #}
{% set v_ptr            = "lr1"  %}  {# V row offset for the current channel #}
{% set rc_slot0         = "lr2"  %}  {# const 0: r_cyclic index #}
{% set agg_slot         = "lr3"  %}  {# AGG dest slot = inner query counter 0..15 #}
{% set chan_off         = "lr4"  %}  {# value-channel row offset, += PV_STRIDE per (h,t) #}
{% set head_p_off       = "lr6"  %}  {# head P row offset, += P_HEAD_STRIDE per head #}
{% set chan_index       = "lr7"  %}  {# head-channel counter t #}
{% set head_index       = "lr8"  %}  {# head counter h #}
{% set group_p_off      = "lr9"  %}  {# group P row offset (= head base; one group at L5) #}
{% set out_chan_off     = "lr10" %}  {# O channel row offset, += O_CHAN_ROWS per (h,t) #}
{% set p_query_stride   = "lr11" %}  {# P query stride in rows (live add) #}

{% set ZERO             = "cr0"  %}  {# const 0 #}
{% set ONE              = "cr1"  %}  {# hardwired read-only 1 #}
{% set P_BASE           = "cr2"  %}  {# query-major P base #}
{% set V_BASE           = "cr3"  %}  {# channel-major V base #}
{% set OUT_BASE         = "cr4"  %}  {# channel-major O base #}
{% set P_QUERY_STRIDE   = "cr5"  %}  {# P query stride / V channel stride (rows) #}
{% set P_HEAD_STRIDE    = "cr8"  %}  {# P head stride (rows) #}
{% set INNER_BOUND      = "cr9"  %}  {# 15 = inner bound -> 16 AGG bundles #}
{% set CHAN_COUNT       = "cr10" %}  {# 60 = head channels t #}
{% set HEAD_COUNT       = "cr11" %}  {# 4 = heads #}
{% set OUT_CHAN_STRIDE  = "cr13" %}  {# O channel stride (rows) #}
{% set DSTRUCT          = "cr15" %}  {# dstructure register: valid_elements = 16 #}

    SET {{ rc_slot0 }} {{ ZERO }};;                                # rc write index = 0 (const)
    SET {{ chan_off }} {{ ZERO }};;                                # value-channel row offset = 0
    SET {{ head_p_off }} {{ ZERO }};;                              # head P row offset = 0
    SET {{ head_index }} {{ ZERO }};;                              # head counter = 0
    SET {{ out_chan_off }} {{ ZERO }};;                            # O channel row offset = 0
    SET {{ p_query_stride }} {{ P_QUERY_STRIDE }};;                # P query stride (rows)

head_loop:
    SET {{ chan_index }} {{ ZERO }};;                              # t counter = 0
    ADD {{ group_p_off }} {{ head_p_off }} {{ ZERO }};;            # group P offset = head P offset

t_loop:
    # ---- the single key chunk (keys 0..15), single query group ----
    ADD {{ v_ptr }} {{ chan_off }} {{ ZERO }};;                    # V row offset = current channel
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;  # R_CYCLIC = V[0..15, t]
    SUB {{ p_ptr }} {{ group_p_off }} {{ p_query_stride }};;       # P inner start = group P off - stride
    SET {{ agg_slot }} {{ ZERO }};;                                # dest/inner counter = 0
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P row
inner_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM.FIRST {{ agg_slot }} {{ DSTRUCT }};
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT {{ agg_slot }} {{ INNER_BOUND }} inner_loop;;

    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_chan_off }} {{ OUT_BASE }};;  # O[0..15, t] = R_ACC

    # ----- next t: advance V channel row offset and O row offset, t++ -----
    ADD {{ chan_off }} {{ chan_off }} {{ P_QUERY_STRIDE }};;       # V chan row offset += PV_STRIDE
    ADD {{ out_chan_off }} {{ out_chan_off }} {{ OUT_CHAN_STRIDE }};; # O chan row offset += O_CHAN_ROWS
    INC {{ chan_index }} 1;;                                       # t++
    BLT {{ chan_index }} {{ CHAN_COUNT }} t_loop;;

    # ----- next head: head P row offset += P_HEAD_STRIDE, head++ -----
    ADD {{ head_p_off }} {{ head_p_off }} {{ P_HEAD_STRIDE }};;    # head P row offset += stride
    INC {{ head_index }} 1;;
    BLT {{ head_index }} {{ HEAD_COUNT }} head_loop;;

end:
    BKPT;;
