# Agent A — attn@V (AGG kernel), query-major scores -> channel-major output.
#
# Layer:   L3
# Scope:   single-stream
# Layout:  unpacked
# Shape:   256tok (2 groups x 128) x 36chan (head_dim), 4 heads
# Status:  validated
# Related: attn_v_64x48 (L4) / attn_v_16x60 (L5) are ports of this kernel;
#          consumes qk_scores_256x36's query-major output; attn_v_bcast_36
#          is the key-major (no-AGG) sibling for the same layer
# Tests:   test_attn_v_256x36_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Per head (4 heads, head_dim D=36, N=256 tokens):
#   O[i, t] = sum_s P[i, s] * V[s, t]      i,s in [0,256), t in [0,36)
#
# Layouts (channel-major activations; .asm operands are ROW numbers, issue #179):
#   P query-major  : P[i, s] at PBASE + h*512 + i*2 + s//128 rows
#                    (query i's 256 scores contiguous -> two 128-key chunks)
#   V channel-major: V[s, t] at VBASE + (h*36 + t)*2 + s//128 rows
#                    (value channel's 256 keys contiguous -> two 128-key chunks)
#   O channel-major: O[i, t] at OBASE + (h*36 + t)*2 + i//128 rows
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
#                 ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG
#                   -> O[g*128 .. g*128+127, t]  (512B FP32 column segment)
#             }}}
#
# Addressing is split offset(LR) + base(CR):
#   chan offset chan_off advances by 2 rows monotonically across all 144 (h,t)
#     steps (hc = h*36 + t runs 0..143 contiguously) -> single running V/O
#     channel ptr.
#   head P offset head_p_off advances by 512 rows per head.
#
# Inner-loop pipeline (live-read offsets, per matmul template):
#   LDR_MULT_REG offset is read LIVE -> ADD p_ptr +256 co-issues; start p_ptr at
#   (first P row addr - 256) so the first live load hits the first row.
#   AGG dest_slot is read from SNAPSHOT; INC agg_slot co-issues -> AGG sees 0,1,..,127.
#   BLT reads SNAPSHOT -> bound 127 gives exactly 128 bundles (dest 0..127).
#
#   MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VV reads its R0 DATA from the
#   start-of-cycle snapshot, so it cannot consume a P chunk loaded by
#   LDR_MULT_REG in its own bundle -- it would see the previous R0. Each of the
#   four inner loops therefore primes query 0's P chunk before the loop, and
#   every bundle multiplies the chunk loaded last cycle while prefetching the
#   next query's. Only R0 needed shifting: the V chunk in R_CYCLIC is loaded by
#   LDR_CYCLIC_MULT_REG two bundles ahead of the loop and never reloaded inside
#   it, so it was already snapshot-safe.
#   agg_slot is unaffected -- both of its readers (AGG and BLT) take the
#   SNAPSHOT, so INC stays co-issued and the dest sequence is still 0..127.
#
# Output O is FP32 (R_ACC) stored one group per row, like the transformer
# matmuls. All .asm operands are ROW numbers (issue #179), so O and the P/V
# inputs share the same row-unit addressing regardless of element width:
# O channel offset advances 2 rows/step (=2 groups), group g=1 is at +1 row.
# Inputs P,V advance 2 rows per query-stride step, 1 row per chunk.
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
{% set p_ptr            = "lr0"  %}  {# P inner load offset (live +2 rows per query) #}
{% set v_ptr            = "lr1"  %}  {# V chunk offset within the current channel #}
{% set rc_slot0         = "lr2"  %}  {# const 0: r_cyclic index #}
{% set agg_slot         = "lr3"  %}  {# AGG dest slot = inner query counter 0..127 #}
{% set chan_off         = "lr4"  %}  {# value-channel offset, += 2 rows per (h,t) step #}
{% set out_chunk        = "lr5"  %}  {# O row offset for the current group #}
{% set head_p_off       = "lr6"  %}  {# head P offset, += 512 rows per head #}
{% set chan_index       = "lr7"  %}  {# head-channel counter t #}
{% set head_index       = "lr8"  %}  {# head counter h #}
{% set group_p_off      = "lr9"  %}  {# group P offset (head base, then + g*256 rows) #}
{% set out_chan_off     = "lr10" %}  {# O channel offset, += 2 rows per (h,t) step #}
{% set p_query_stride   = "lr11" %}  {# 2 = P query stride, in rows (live add) #}

{% set ZERO             = "cr0"  %}  {# const 0 #}
{% set ONE              = "cr1"  %}  {# hardwired read-only 1 #}
{% set P_BASE           = "cr2"  %}  {# query-major P base #}
{% set V_BASE           = "cr3"  %}  {# channel-major V base #}
{% set OUT_BASE         = "cr4"  %}  {# channel-major O base #}
{% set P_QUERY_STRIDE   = "cr5"  %}  {# 2 = P query stride (rows) #}
{% set CHUNK_OFF        = "cr6"  %}  {# 1 = second key-chunk offset (rows) #}
{% set P_GROUP_STRIDE   = "cr7"  %}  {# 256 = 128 queries x 2 rows (group g=1) #}
{% set P_HEAD_STRIDE    = "cr8"  %}  {# 512 = P head stride (rows) #}
{% set INNER_BOUND      = "cr9"  %}  {# 127 = inner bound -> 128 AGG bundles #}
{% set CHAN_COUNT       = "cr10" %}  {# 36 = head channels t #}
{% set HEAD_COUNT       = "cr11" %}  {# 4 = heads #}
{% set OUT_GROUP_STRIDE = "cr12" %}  {# 1 = O group stride (rows) #}
{% set OUT_CHAN_STRIDE  = "cr13" %}  {# 2 = O channel stride (rows) #}
{% set DSTRUCT          = "cr15" %}  {# reserved dstructure register #}

    SET {{ rc_slot0 }} {{ ZERO }};;                                # rc write index = 0 (const)
    SET {{ chan_off }} {{ ZERO }};;                                # value-channel offset = 0
    SET {{ head_p_off }} {{ ZERO }};;                              # head P offset = 0
    SET {{ head_index }} {{ ZERO }};;                              # head counter = 0
    SET {{ out_chan_off }} {{ ZERO }};;                            # O channel offset = 0
    SET {{ p_query_stride }} {{ P_QUERY_STRIDE }};;                # P query stride = 2 rows

head_loop:
    SET {{ chan_index }} {{ ZERO }};;                              # t counter = 0
    ADD {{ group_p_off }} {{ head_p_off }} {{ ZERO }};;            # group P offset = head P offset (g=0)

t_loop:
    # ===================== group g = 0 (queries 0..127) =====================
    # ---- chunk 0: keys 0..127 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ ZERO }};;                    # V chunk0 offset = chan
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;  # R_CYCLIC = V[0..127, t]   (base VBASE)
    SUB {{ p_ptr }} {{ group_p_off }} {{ p_query_stride }};;       # P inner start = group P off - 2 rows
    SET {{ agg_slot }} {{ ZERO }};;                                # dest/inner counter = 0
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P chunk
g0c0_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM.FIRST {{ agg_slot }} {{ DSTRUCT }};
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT {{ agg_slot }} {{ INNER_BOUND }} g0c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ CHUNK_OFF }};;               # V chunk1 offset = chan + 1 row
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;  # R_CYCLIC = V[128..255, t]
    ADD {{ p_ptr }} {{ group_p_off }} {{ CHUNK_OFF }};;            # P chunk1 base = group P off + 1 row
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 2 rows startup
    SET {{ agg_slot }} {{ ZERO }};;
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P chunk
g0c1_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM {{ agg_slot }} {{ DSTRUCT }};
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT {{ agg_slot }} {{ INNER_BOUND }} g0c1_loop;;

    ADD {{ out_chunk }} {{ out_chan_off }} {{ ZERO }};;            # O g=0 offset = O chan offset
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_chunk }} {{ OUT_BASE }};;  # O[0..127, t] = R_ACC   (base OBASE)

    # ===================== group g = 1 (queries 128..255) ===================
    # ---- chunk 0: keys 0..127 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ ZERO }};;                    # V chunk0 offset = chan (same channel)
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;
    ADD {{ p_ptr }} {{ group_p_off }} {{ P_GROUP_STRIDE }};;       # g=1 P base = group P off + 256 rows
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 2 rows startup
    SET {{ agg_slot }} {{ ZERO }};;
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P chunk
g1c0_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM.FIRST {{ agg_slot }} {{ DSTRUCT }};
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT {{ agg_slot }} {{ INNER_BOUND }} g1c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ CHUNK_OFF }};;               # V chunk1 offset = chan + 1 row
    LDR_CYCLIC_MULT_REG {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;
    ADD {{ p_ptr }} {{ group_p_off }} {{ P_GROUP_STRIDE }};;       # g=1 P base
    ADD {{ p_ptr }} {{ p_ptr }} {{ CHUNK_OFF }};;                  # + 1 row (chunk1)
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 2 rows startup
    SET {{ agg_slot }} {{ ZERO }};;
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P chunk
g1c1_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM {{ agg_slot }} {{ DSTRUCT }};
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT {{ agg_slot }} {{ INNER_BOUND }} g1c1_loop;;

    ADD {{ out_chunk }} {{ out_chan_off }} {{ OUT_GROUP_STRIDE }};; # O g=1 offset = O chan offset + 1 row
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_chunk }} {{ OUT_BASE }};;  # O[128..255, t] = R_ACC

    # ----- next t: advance value-channel offset (+2 rows) and O offset (+2 rows), t++ -----
    ADD {{ chan_off }} {{ chan_off }} {{ P_QUERY_STRIDE }};;       # chan += 2 rows
    ADD {{ out_chan_off }} {{ out_chan_off }} {{ OUT_CHAN_STRIDE }};; # O chan offset += 2 rows
    INC {{ chan_index }} 1;;                                       # t++
    BLT {{ chan_index }} {{ CHAN_COUNT }} t_loop;;

    # ----- next head: head P offset += 512 rows, head++ -----
    ADD {{ head_p_off }} {{ head_p_off }} {{ P_HEAD_STRIDE }};;    # head P offset += 512 rows
    INC {{ head_index }} 1;;
    BLT {{ head_index }} {{ HEAD_COUNT }} head_loop;;

end:
    BKPT;;
