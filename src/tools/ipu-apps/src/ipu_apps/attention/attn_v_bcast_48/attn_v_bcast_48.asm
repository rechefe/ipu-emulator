# attn@V (broadcast / no AGG), Layer 4 — key-major scores -> channel-major output
#
# Layer:   L4
# Scope:   single-stream
# Layout:  unpacked
# Shape:   64tok x 48chan (head_dim), P=4 streams x 4 heads (16 blocks)
# Status:  validated
# Related: L4 port of attn_v_bcast_36 (L3); attn_v_bcast_60 is the L5 port;
#          consumes attn_scores_km_64x48's key-major output; attn_v_64x48
#          is the query-major (AGG) sibling for the same layer
# Tests:   test_attn_v_bcast_48_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Per (stream, head) block b in [0,16)  (P=4 streams x h=4 heads):
#   O[b, i, t] = sum_s P[b, i, s] * V[b, s, t]
#     i = query in [0,64)   (the SIMD lanes)
#     t = head channel in [0,48)   (global value channel = b*48 + t)
#     s = key (contraction) in [0,64)
#
# Layer 4: d = 192, N = 64 tokens/stream, head_dim D = 48. head_dim = 48 is a
# LOOP COUNT here (the t loop), not a lane count -- it needs no padding.
#
# This is the L4 port of attn_v_bcast_36. LANES = QUERIES and the standard
# matmul-broadcast template is carried over unchanged (V in the weight/scalar
# role, key-major P in the streamed-data role):
#   mult_res[i] = P[i,s] * V[s,t]   ;  ACC.ADD[.FIRST] accumulates over s
#   After 64 keys, R_ACC[i] = O[i,t] for the 64 queries of the block.
# NO AGG and no collision: every cycle is a full-width scalar*vector MULT + ACC.
#
# N = 64 <= LANES = 128 changes two things versus L3, both structural:
#   * A query axis is ONE row, so the L3 g=0/g=1 query-group split collapses to
#     a single group -- one ACC chain and one store per (block, channel).
#   * All 64 keys of a value channel fit in R0 alone, so the L3 R0++R1 pair
#     (which spanned 256 keys) collapses to a single LDR_MULT_REG r0. The
#     scalar index s (0..63) then selects V[s,t] directly with NO mid-loop
#     reload.
#
# Layouts (all .asm operands are ROW numbers, issue #179):
#   P key-major   : P[b, i, s] on row (b*64 + s), queries 0..63 in the row's
#                   leading lanes. This is exactly what attn_scores_km_64x48
#                   writes -- key (b,s) owns a whole row of query scores.
#   V channel-major: V[b, s, t] on row (b*48 + t), keys 0..63 leading.
#   O channel-major: O[b, i, t] on row (b*48 + t), queries 0..63 leading.
#   V and O match the query-major sibling attn_v_64x48 exactly; only P's
#   orientation differs, which is the whole point of the key-major chain.
#   The trailing 64 lanes of every P and V row are zero padding, so the products
#   in lanes 64..127 contribute nothing and the harness crops them away.
#
# Loop nest:  for b in 0..15 { for t in 0..47 {
#                 load V[b,:,t] -> r0 (all 64 keys, the scalar source)
#                 prime key 0's P row; for s in 0..63: MULT.RC.VE + ACC
#                 ACTIVATE.QUANTIZE + STR_POST_AAQ_REG
#             }}
#
# Counts: 16 blocks x 48 channels x 64 keys = 49152 MULT+ACC bundles + 768
#         stores.
#
# Addressing is split offset(LR) + base(CR):
#   chan_off advances by one row monotonically across all 768 (b,t) steps
#     (b*48 + t runs 0..767 contiguously) -> a single running V/O channel ptr,
#     shared because V and O have identical per-block row layouts.
#   p_block_off advances by 64 rows per block; p_ptr re-seeds from it per
#     channel, so no pipeline state crosses a channel or block boundary.
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
# the start-of-cycle snapshot while keeping the LR index live, so it cannot
# consume a P row loaded by LDR_CYCLIC_MULT_REG in its own bundle -- it would
# see the previous row. `;;` ends one VLIW word = one cycle = one snapshot, so a
# load and a MULT in the same bundle always run in the same cycle regardless of
# textual order; co-issuing is fine, consuming the same-cycle load is not.
#   Each channel therefore primes key 0's P row, then every loop body
#   multiplies the row loaded last cycle while prefetching the next one.
#   p_ptr (the load offset) is read LIVE by the load, so its ADD stays co-issued
#   with the load it feeds -- and because the load now runs one bundle ahead,
#   p_ptr naturally addresses the NEXT key's row.
#   key_index (the r0 scalar selector) is read LIVE by MULT and must name the
#   key just loaded, so its ADD moves into an s_loop_pre block that falls
#   through into the loop body. That changes what BLT sees: with the ADD in the
#   same word, BLT read the pre-ADD snapshot; with the ADD one word earlier it
#   reads the already-incremented value and would drop the last key. The loop
#   bound is therefore raised by one, in-kernel, into key_last
#   (= KEY_BOUND + 1 = N-1 = 63), leaving the harness's CR untouched.
#   r0 (the V channel) is loaded two bundles ahead of the loop and never
#   reloaded inside it, so it is already snapshot-safe.
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
{% set rc_slot0    = "lr0"  %}  {# const 0: r_cyclic index / mask_shift #}
{% set key_stride  = "lr1"  %}  {# 1 = P key stride, in rows #}
{% set p_ptr       = "lr4"  %}  {# row offset into key-major P, += key_stride #}
{% set key_index   = "lr5"  %}  {# contraction key index s -> selects V[s,t] in r0 #}
{% set blk_index   = "lr6"  %}  {# block counter b (stream, head) #}
{% set chan_index  = "lr7"  %}  {# head-channel counter t #}
{% set key_last    = "lr8"  %}  {# 63 = KEY_BOUND + 1; BLT bound now that key_index's ADD sits a word ahead of the branch #}
{% set chan_off    = "lr9"  %}  {# V/O channel row offset, += 1 per (b,t) step #}
{% set p_block_off = "lr12" %}  {# P block row offset, += 64 per block #}
{% set p_start     = "lr13" %}  {# p_block_off + P_START; per-channel p_ptr seed #}

{% set ZERO        = "cr0"  %}  {# hardwired read-only 0 #}
{% set ONE         = "cr1"  %}  {# hardwired read-only 1 #}
{% set P_BASE      = "cr2"  %}  {# key-major score base #}
{% set V_BASE      = "cr3"  %}  {# channel-major value base #}
{% set OUT_BASE    = "cr4"  %}  {# channel-major output base #}
{% set P_START     = "cr5"  %}  {# -1 row startup skew (first live = block base) #}
{% set KEY_START   = "cr6"  %}  {# -1 -> first live s = 0 #}
{% set P_BLOCK     = "cr7"  %}  {# 64 = P rows per (stream, head) block #}
{% set KEY_BOUND   = "cr8"  %}  {# 62 = N - 2 (key-loop bound, width 64) #}
{% set CHAN_COUNT  = "cr10" %}  {# 48 = head channels t per block #}
{% set BLOCK_COUNT = "cr11" %}  {# 16 = P * N_HEAD blocks #}
{% set CHAN_STRIDE = "cr12" %}  {# 1 = V/O channel stride (rows) #}
{% set DSTRUCT     = "cr15" %}  {# reserved dstructure register #}

    # key_last = KEY_BOUND + 1 (= N-1 = 63). See the snapshot note in the
    # header: key_index's ADD now sits one word ahead of the BLT, so the branch
    # compares an already-incremented index and needs the higher bound.
    SET {{ key_last }} {{ KEY_BOUND }};;
    ADD {{ key_last }} {{ key_last }} {{ ONE }};;

    SET {{ chan_off }} {{ ZERO }};;                        # V/O channel offset = 0
    SET {{ p_block_off }} {{ ZERO }};;                     # P block offset = 0
    SET {{ blk_index }} {{ ZERO }};;                       # block counter = 0

blk_loop:
    SET {{ chan_index }} {{ ZERO }};;                      # t counter = 0
    ADD {{ p_start }} {{ p_block_off }} {{ P_START }};;    # per-channel p_ptr seed = block base - 1 row

t_loop:
    # V channel -> r0: all 64 keys of channel (b,t), the broadcast scalars.
    LDR_MULT_REG r0 {{ chan_off }} {{ V_BASE }};;          # r0 = V[b, 0..63, t]

    ADD {{ p_ptr }} {{ p_start }} {{ ZERO }};;             # P-data startup: block base - 1 row (LR copy)
    SET {{ key_index }} {{ KEY_START }};;                  # fixed_idx s startup: -1

    # Prime key 0's P row: MULT reads r_cyclic from the snapshot, so the row it
    # consumes must be loaded a cycle earlier (see header).
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    ADD {{ key_index }} {{ key_index }} {{ ONE }};;

    # Peeled first key (s=0): ACC.FIRST seeds r_acc.
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT {{ key_index }} {{ key_last }} s_loop_pre;;
    B s_done;;

s_loop_pre:
    ADD {{ key_index }} {{ key_index }} {{ ONE }};;        # name the key just loaded

s_loop:
    MULT.RC.VE {{ rc_slot0 }} {{ key_index }} 0 {{ rc_slot0 }} {{ DSTRUCT }};
    ACC.ADD;
    LDR_CYCLIC_MULT_REG {{ p_ptr }} {{ P_BASE }} {{ rc_slot0 }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ key_stride }};
    BLT {{ key_index }} {{ key_last }} s_loop_pre;;

s_done:
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ chan_off }} {{ OUT_BASE }};;  # O[b, 0..63, t] = R_ACC

    ADD {{ chan_off }} {{ chan_off }} {{ CHAN_STRIDE }};;  # next channel row (V and O share the offset)
    ADD {{ chan_index }} {{ chan_index }} {{ ONE }};;
    BLT {{ chan_index }} {{ CHAN_COUNT }} t_loop;;

    ADD {{ p_block_off }} {{ p_block_off }} {{ P_BLOCK }};;   # next block: P base += 64 rows
    ADD {{ blk_index }} {{ blk_index }} {{ ONE }};;
    BLT {{ blk_index }} {{ BLOCK_COUNT }} blk_loop;;

end:
    BKPT;;
