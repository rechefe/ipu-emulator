# attn@V (AGG kernel), Layer 4 — query-major scores -> channel-major output.
#
# Layer:   L4
# Scope:   single-stream
# Layout:  unpacked
# Shape:   64tok x 48chan (head_dim), P=4 streams x 4 heads (16 blocks)
# Status:  validated
# Related: L4 port of attn_v_256x36 (L3); attn_v_16x60 is the L5 port;
#          consumes qk_scores_64x48's query-major output; attn_v_bcast_48
#          is the key-major (no-AGG) sibling for the same layer
# Tests:   test_attn_v_64x48_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Per (stream, head) block b in [0,16)  (P=4 streams x h=4 heads):
#   O[b, i, t] = sum_s P[b, i, s] * V[b, s, t]   i,s in [0,64), t in [0,48)
#
# Layer 4: d = 192, N = 64 tokens/stream, head_dim D = 48.
#
# This is the L4 port of attn_v_256x36; the compute model is carried over
# unchanged:
#   Lanes = keys.  MULT.RC.VV: mult_res[s] = P[i,s] * V[s,t]  (element-wise,
#   R0 = the query's score row, R_CYCLIC = the value channel's key column).
#   AGG.SUM.FIRST dest=local reduces the live mult_res lanes into R_ACC[local]
#   -- collision-free, no ACC, no reset.
#   The V channel is loaded once and reused across all 64 queries; only P (R0)
#   reloads each cycle.
#
# N = 64 <= LANES = 128 changes two things versus L3, both structural:
#   * A key axis is ONE row, so there is a single key chunk per channel. The
#     L3 chunk-0/chunk-1 split collapses and every AGG is AGG.SUM.FIRST -- one
#     clean write per query slot, with no cross-chunk accumulation to carry.
#   * A query group is 64 queries, so a single R_ACC store covers a whole
#     output channel: AGG dest slots run 0..63 and R_ACC lanes 64..127 stay
#     unused padding (the harness crops them; rows are never shared).
#   AGG.SUM.FIRST is gated by DSTRUCT's valid_elements = N_TOK (64), set by the
#   harness, so the reduction structurally sees only the 64 real key lanes --
#   it does not depend on the trailing 64 lanes of P and V being zero.
#
# head_dim = 48 is a LOOP COUNT here (the t loop), not a lane count -- it needs
# no padding.
#
# Layouts (all .asm operands are ROW numbers, issue #179):
#   P query-major : P[b, i, s] on row (b*64 + i), keys 0..63 in the row's
#                   leading lanes.
#   V channel-major: V[b, s, t] on row (b*48 + t), keys 0..63 leading.
#   O channel-major: O[b, i, t] on row (b*48 + t), queries 0..63 leading.
#
# Loop nest:  for b in 0..15 { for t in 0..47 {
#                 load V channel -> R_CYCLIC
#                 for i in 0..63: load P row; MULT.RC.VV; AGG.SUM.FIRST i
#                 ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG
#             }}
#
# Addressing is split offset(LR) + base(CR):
#   chan_off advances by one row monotonically across all 768 (b,t) steps
#     (b*48 + t runs 0..767 contiguously) -> a single running V/O channel ptr,
#     shared because V and O have identical per-block row layouts.
#   p_block_off advances by 64 rows per block.
#
# Inner-loop pipeline (per the matmul template):
#   LDR_MULT_REG offset is read LIVE -> ADD p_ptr +1 row co-issues; start p_ptr
#     at (first P row - 1) so the first live load hits the first row.
#   AGG dest_slot is read from SNAPSHOT; INC agg_slot co-issues -> AGG sees
#     0,1,..,63.
#   BLT reads SNAPSHOT -> bound 63 gives exactly 64 bundles (dest 0..63).
#
#   MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VV reads its R0 DATA from the
#   start-of-cycle snapshot, so it cannot consume a P row loaded by
#   LDR_MULT_REG in its own bundle -- it would see the previous R0. The inner
#   loop therefore primes query 0's P row before the loop, and every bundle
#   multiplies the row loaded last cycle while prefetching the next query's.
#   The V channel in R_CYCLIC is loaded two bundles ahead of the loop and never
#   reloaded inside it, so it is already snapshot-safe.
#   agg_slot is unaffected -- both of its readers (AGG and BLT) take the
#   SNAPSHOT, so INC stays co-issued and the dest sequence is still 0..63.
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
{% set p_ptr            = "lr0"  %}  {# P inner load offset (live +1 row per query) #}
{% set rc_slot0         = "lr2"  %}  {# const 0: r_cyclic index #}
{% set agg_slot         = "lr3"  %}  {# AGG dest slot = inner query counter 0..63 #}
{% set chan_off         = "lr4"  %}  {# V/O channel row offset, += 1 per (b,t) step #}
{% set chan_index       = "lr7"  %}  {# head-channel counter t #}
{% set blk_index        = "lr8"  %}  {# block counter b (stream, head) #}
{% set p_block_off      = "lr9"  %}  {# P block row offset, += 64 per block #}
{% set p_query_stride   = "lr11" %}  {# 1 = P query stride, in rows (live add) #}

{% set ZERO             = "cr0"  %}  {# hardwired read-only 0 #}
{% set ONE              = "cr1"  %}  {# hardwired read-only 1 #}
{% set P_BASE           = "cr2"  %}  {# query-major P base #}
{% set V_BASE           = "cr3"  %}  {# channel-major V base #}
{% set OUT_BASE         = "cr4"  %}  {# channel-major O base #}
{% set P_QUERY_STRIDE   = "cr5"  %}  {# 1 = P query stride (rows) #}
{% set P_BLOCK_STRIDE   = "cr8"  %}  {# 64 = P rows per (stream, head) block #}
{% set INNER_BOUND      = "cr9"  %}  {# 63 = inner bound -> 64 AGG bundles #}
{% set CHAN_COUNT       = "cr10" %}  {# 48 = head channels t per block #}
{% set BLOCK_COUNT      = "cr11" %}  {# 16 = P * N_HEAD blocks #}
{% set OUT_CHAN_STRIDE  = "cr12" %}  {# 1 = O channel stride (rows) #}
{% set DSTRUCT          = "cr15" %}  {# reserved dstructure register #}

    SET {{ rc_slot0 }} {{ ZERO }};;                                # rc write index = 0 (const)
    SET {{ chan_off }} {{ ZERO }};;                                # V/O channel offset = 0
    SET {{ p_block_off }} {{ ZERO }};;                             # P block offset = 0
    SET {{ blk_index }} {{ ZERO }};;                               # block counter = 0
    SET {{ p_query_stride }} {{ P_QUERY_STRIDE }};;                # P query stride = 1 row

blk_loop:
    SET {{ chan_index }} {{ ZERO }};;                              # t counter = 0

t_loop:
    # V channel -> R_CYCLIC (loaded once, reused by all 64 queries).
    LDR_CYCLIC_MULT_REG {{ chan_off }} {{ V_BASE }} {{ rc_slot0 }};;  # R_CYCLIC = V[b, 0..63, t]
    SUB {{ p_ptr }} {{ p_block_off }} {{ p_query_stride }};;       # P inner start = block base - 1 row
    SET {{ agg_slot }} {{ ZERO }};;                                # dest/inner counter = 0
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P row

i_loop:
    MULT.RC.VV {{ rc_slot0 }} r0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM.FIRST {{ agg_slot }} {{ DSTRUCT }};
    LDR_MULT_REG r0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT {{ agg_slot }} {{ INNER_BOUND }} i_loop;;

    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ chan_off }} {{ OUT_BASE }};;  # O[b, 0..63, t] = R_ACC

    # ----- next t: advance the shared V/O channel row offset, t++ -----
    ADD {{ chan_off }} {{ chan_off }} {{ OUT_CHAN_STRIDE }};;      # channel row += 1
    INC {{ chan_index }} 1;;                                       # t++
    BLT {{ chan_index }} {{ CHAN_COUNT }} t_loop;;

    # ----- next block: P block offset += 64 rows, b++ -----
    ADD {{ p_block_off }} {{ p_block_off }} {{ P_BLOCK_STRIDE }};;
    INC {{ blk_index }} 1;;
    BLT {{ blk_index }} {{ BLOCK_COUNT }} blk_loop;;

end:
    BKPT;;
