# QK^T scores (Layer 4), query-major, per (stream, head) block:
#
# Layer:   L4
# Scope:   single-stream
# Layout:  unpacked
# Shape:   64tok x 48chan (head_dim), P=4 streams x 4 heads (16 blocks)
# Status:  validated
# Related: L4 port of qk_scores_256x36 (L3); qk_scores_16x60 is the L5 port;
#          feeds attn_v_64x48 (query-major chain); attn_scores_km_64x48 is
#          the key-major sibling for the same layer
# Tests:   test_qk_scores_64x48_wide (src/tools/ipu-apps/BUILD.bazel)
#
#   S[b, i, s] = sum_c Q[b, i, c] * K[b, s, c]   contraction over head_dim c = 0..47
#
# Layer 4: d = 192, N = 64 tokens/stream, P = 4 streams, h = 4 heads,
# head_dim D = 48, so there are P*h = 16 independent (stream, head) blocks and
# each is a 64-query x 64-key x 48-channel score problem.
#
# This is the L4 port of qk_scores_256x36. The mapping is carried over
# unchanged -- broadcast template, no AGG, no MULT.RC.VV score form:
#   K channel-major: K[b, s, c] on row (b*48 + c), keys 0..63 in the first 64
#     lanes of the row (N = 64 <= LANES = 128, so a channel column is ONE row
#     and there is exactly ONE key group -- the 2-group split of the L3 kernel
#     collapses away).
#   Q staged query-major by the harness (gather of the strided channels):
#     QROW[b, i] = Q[b, i, 0..47] contiguous on row (b*64 + i). This lets r0
#     hold one query's 48 channels with a single LDR_MULT_REG, exactly the
#     matmul broadcast template (scalar from r0 indexed by c).
#
# Broadcast template (mirrors qk_scores_256x36, width 48):
#   per query i:  r0 = QROW[b, i]           (48 scalars Q[b, i, 0..47])
#   per channel:  r_cyclic = K[b, 0..63, c]     (64 keys' channel-c column)
#                 MULT.RC.VE: scalar Q[i,c] (= r0[c]) x vector r_cyclic
#                   -> mult_res[s] = Q[i,c] * K[s,c]
#                 ACC.ADD.FIRST (c=0) / ACC.ADD      -> R_ACC[s] += Q[i,c] * K[s,c]
#   after 48 channels R_ACC[s] = S[b, i, s] for the 64 keys of the block.
#   ACTIVATE.QUANTIZE identity + STR_POST_AAQ_REG
#                 -> one whole row per query (the leading 64 FP32 lanes are the
#                    scores; the harness crops them in teardown -- rows are
#                    never shared between queries).
#
# head_dim = 48 needs no padding: it is the contraction LOOP COUNT (bound
# lr6 = D-2 = 46), not a lane count. Lanes here are keys.
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
# the start-of-cycle snapshot while keeping the LR index live, so it cannot
# consume a K column loaded by LDR_CYCLIC_MULT_REG in its own bundle -- it
# would see the previous column. `;;` ends one VLIW word = one cycle = one
# snapshot, so a load and a MULT in the same bundle always run in the same
# cycle regardless of textual order; co-issuing is fine, consuming the
# same-cycle load is not.
#   Each query therefore primes c=0's column, then every loop body multiplies
#   the column loaded last cycle while prefetching the next one.
#   k_ptr (the load offset) is read LIVE by the load, so its ADD stays
#   co-issued with the load it feeds -- and because the load now runs one
#   bundle ahead, k_ptr naturally addresses the NEXT channel's column.
#   chan_index (the r0 scalar selector) is read LIVE by MULT and must name the
#   channel just loaded, so its ADD moves into a c_loop_pre block that falls
#   through into the loop body. That changes what BLT sees: with the ADD in the
#   same word, BLT read the pre-ADD snapshot; with the ADD one word earlier it
#   reads the already-incremented value and would exit a channel early. The
#   loop bound is therefore raised by one, in-kernel, into chan_last
#   (= chan_bound + 1 = D-1 = 47), leaving the harness's lr6 untouched.
#   Each query re-SETs k_ptr and chan_index, so no pipeline state crosses a
#   query or block boundary.
#
# Loop nest:  for b in 0..15 { for i in 0..63 {
#                 prime c=0; for c in 0..47: MULT.RC.VE + ACC
#                 ACTIVATE.QUANTIZE + STR_POST_AAQ_REG
#             }}
#
# Addressing is split offset(LR) + base(CR) and all .asm operands are ROW
# numbers (issue #179). q_ptr and out_ptr advance monotonically across all
# 16 blocks (block b's queries are rows b*64 .. b*64+63 in both regions), so
# only k_block_off needs a per-block step (+48 rows).
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
{% set rc_slot0    = "lr0"  %}  {# const 0: r_cyclic write-index / mask_shift #}
{% set k_stride    = "lr2"  %}  {# 1 = rows per K channel column #}
{% set out_stride  = "lr3"  %}  {# 1 = rows per query (one key group) #}
{% set k_ptr       = "lr4"  %}  {# row offset into K, walks head channels c #}
{% set chan_index  = "lr5"  %}  {# head-channel index c -> selects Q[i,c] in r0 #}
{% set chan_bound  = "lr6"  %}  {# 46 = contraction bound (first=0, width=48) #}
{% set chan_last   = "lr11" %}  {# 47 = chan_bound + 1; BLT bound now that chan_index's ADD sits a word ahead of the branch #}
{% set out_ptr     = "lr7"  %}  {# row offset into S, += out_stride per query #}
{% set q_ptr       = "lr8"  %}  {# row offset into staged QROW, += qrow_stride #}
{% set q_index     = "lr9"  %}  {# query counter i within the block #}
{% set q_limit     = "lr10" %}  {# 64 = N queries per block #}
{% set qrow_stride = "lr12" %}  {# 1 = QROW rows per query #}
{% set k_block_off = "lr13" %}  {# K row offset of the current block, += 48 #}
{% set blk_index   = "lr14" %}  {# block counter b (stream, head) #}
{% set k_start     = "lr15" %}  {# k_block_off + K_START; per-query k_ptr seed #}

{% set ZERO        = "cr0"  %}  {# hardwired read-only 0 #}
{% set K_BASE      = "cr2"  %}  {# channel-major K base #}
{% set ONE         = "cr1"  %}  {# hardwired read-only 1 #}
{% set S_BASE      = "cr3"  %}  {# query-major score output base #}
{% set K_START     = "cr5"  %}  {# -1 row startup skew (first live = block base) #}
{% set CHAN_START  = "cr7"  %}  {# -1 -> first live c = 0 #}
{% set CHAN_BOUND  = "cr8"  %}  {# 46 = D - 2 #}
{% set QROW_BASE   = "cr9"  %}  {# staged query-major Q base #}
{% set BLK_LIMIT   = "cr10" %}  {# 16 = P * N_HEAD blocks #}
{% set K_BLOCK     = "cr11" %}  {# 48 = K rows per (stream, head) block #}
{% set DSTRUCT     = "cr15" %}  {# reserved dstructure register #}

    # chan_last = chan_bound + 1 (= D-1 = 47). See the snapshot note in the
    # header: chan_index's ADD now sits one word ahead of the BLT, so the branch
    # compares an already-incremented index and needs the higher bound.
    ADD {{ chan_last }} {{ chan_bound }} {{ ONE }};;

blk_loop:
    SET {{ q_index }} {{ ZERO }};;                        # query counter = 0
    ADD {{ k_start }} {{ k_block_off }} {{ K_START }};;   # per-query k_ptr seed = block base - 1 row

q_loop:
    LDR_MULT_REG r0 {{ q_ptr }} {{ QROW_BASE }};;         # r0 = QROW[b, i] = Q[b, i, 0..47] (rest pad)

    ADD {{ k_ptr }} {{ k_start }} {{ ZERO }};;            # K-data startup: block base - 1 row (LR copy)
    SET {{ chan_index }} {{ CHAN_START }};;               # channel fixed_idx startup: -1

    # Prime c=0's K column: MULT reads r_cyclic from the snapshot, so the column
    # it consumes must be loaded a cycle earlier (see header).
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;

    # Peeled first channel (c=0): ACC.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_pre;;
    B after_c;;

c_loop_pre:
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;     # name the channel just loaded

c_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ chan_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ k_ptr }} {{ K_BASE }} {{ rc_slot0 }};
    ADD {{ k_ptr }} {{ k_ptr }} {{ k_stride }};
    BLT {{ chan_index }} {{ chan_last }} c_loop_pre;;

after_c:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ out_ptr }} {{ S_BASE }};;  # store row -> S[b, i, keys 0..63]
    ADD {{ out_ptr }} {{ out_ptr }} {{ out_stride }};;    # advance output row

    ADD {{ q_ptr }} {{ q_ptr }} {{ qrow_stride }};
    ADD {{ q_index }} {{ q_index }} {{ ONE }};;  # next query
    BLT {{ q_index }} {{ q_limit }} q_loop;;

    ADD {{ k_block_off }} {{ k_block_off }} {{ K_BLOCK }};;   # next block: K base += 48 rows
    ADD {{ blk_index }} {{ blk_index }} {{ ONE }};;
    BLT {{ blk_index }} {{ BLK_LIMIT }} blk_loop;;

end:
    BKPT;;
