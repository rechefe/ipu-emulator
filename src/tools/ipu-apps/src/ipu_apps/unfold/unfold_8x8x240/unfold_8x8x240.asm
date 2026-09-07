# Unfold 8x8x240 -> 4 stride-2 streams of [16, 240] channel-major FP32   (L5)
#
# Layer:   L5
# Scope:   single-stream (per-stripe; emits the 4 spatial streams)
# Layout:  unpacked
# Shape:   8x8x240 spatial input -> 4 streams x [16tok, 240chan]
# Status:  validated
# Related: L5 sibling of unfold_32x32x144 (L3) / unfold_16x16x192 (L4),
#          geometry re-derived not ported (1 stripe, elements_in_row=16);
#          requires the caller to pre-permute input rows per
#          ipu_apps.unfold.unfold_8x8x240._ROW_PACK_ORDER (see
#          kernel_docs/kernel_layer_map.md); feeds layernorm_16x240
# Tests:   test_unfold_8x8x240_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Rearranges an 8x8x240 spatial tensor into 4 stride-2 decimated streams,
# channel-major. The streams are the standard stride-2 space-to-depth
# decomposition -- stream s takes phase (s // 2, s % 2) of every 2x2 block.
# They are NOT image quadrants; on/on_inv are just ACC.STRIDE's even/odd
# selector encodings.
#
# Geometry derived for L5, not copied from L3/L4:
#   L3 (32x32x144): 8 stripes x (4 spatial rows x 32 cols), elements_in_row=32
#   L4 (16x16x192): 2 stripes x (8 spatial rows x 16 cols), elements_in_row=16
#   L5 (8x8x240):   1 stripe  x (8 spatial rows x  8 cols), elements_in_row=16
#
# W=8 is not an encodable elements_in_row (only 16/32/64), so a whole 8x8
# channel (64 elements) sits in the first 64 lanes of ONE row and ACC.STRIDE
# views it as 4 rows x 16 cols -- each view row holding TWO spatial rows.
# The harness packs spatial rows in the order [0|2] [1|3] [4|6] [5|7] so that
# ACC.STRIDE's even/odd view-row split lines up with the even/odd SPATIAL row
# split. See the harness docstring for why row-major packing would be wrong.
#
# Input:
#   1 stripe x 240 channels; row ch at SRC_BASE + ch, 64 valid lanes + padding.
#
# Output (per-stream channel-major FP32, one channel per row):
#   4 streams x 240 rows x 512 bytes, stream s based at DST_s.
#   Stream s, ch c: at DST_s + c.
#
#   *** Each row carries 16 valid FP32 tokens (64 bytes); lanes 16..127 are
#   *** STALE and must be ignored. A stream fills only 16 of the 32 elements
#   *** of one ACC.STRIDE slot, ACTIVATE.QUANTIZE identity stages all 128 lanes
#   *** of r_acc, and STR_POST_AAQ_REG writes the full 512-byte row.
#   Rows are NEVER shared between channels -- the harness teardown crops each
#   row to its 16-token prefix.
#
# Pass-through multiply: MULT.RC.VV with r_cyclic preloaded to 1.0 and the
#   stripe row in r0 (via LDR_MULT_REG r0):
#     MULT.RC.VV lr0 r0 0 lr0: r0[i] x r_cyclic[i] = stripe[i] x 1.0 = stripe[i].
#   r_cyclic[0..127] = 1.0 (loaded once at startup, never overwritten).
#
# Stream definitions (acc.stride with elements_in_row=16, all into slot 0):
#   stream 0: acc.stride 16 on     on     -> even cols, even spatial rows
#   stream 1: acc.stride 16 on_inv on     -> odd  cols, even spatial rows
#   stream 2: acc.stride 16 on     on_inv -> even cols, odd  spatial rows
#   stream 3: acc.stride 16 on_inv on_inv -> odd  cols, odd  spatial rows
#   Each selects 16 valid tokens into r_acc lanes 0..15 (offset lr0 = slot 0).
#
# acc.stride enum encoding (from acc_stride_enums.py):
#   elements_in_row: "16" (enum 0)
#   horizontal: on=1 (even cols), on_inv=2 (odd cols)
#   vertical:   on=1 (even rows), on_inv=2 (odd rows)
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VV reads its r0 DATA from the
# start-of-cycle snapshot, so a MULT cannot consume a chunk loaded by an
# LDR_MULT_REG in its own bundle -- it would see the PREVIOUS contents of r0.
# `;;` ends one VLIW word = one cycle = one snapshot, so a load and a MULT in
# the same bundle always execute in the same cycle regardless of textual order;
# co-issuing them is fine, consuming the same-cycle load is not.
# Here there is a single stripe, so r0 holds the SAME channel row for all four
# streams: one load per channel, primed one bundle ahead, feeds all 4 MULTs.
# The load for channel ch+1 rides in the last store bundle of channel ch, which
# is why src_off's ADD must precede it.
#
# STR_POST_AAQ_REG reads its offset LIVE -- never co-issue an ADD on the offset
# LR in the same bundle as the store, or the row lands one slot late.
# dst_off advances in its own bundle at the end of the channel loop, clear of
# every store.
#
# Registers are referred to below by the symbolic names defined in the
# register-name block. The assembler's Jinja2 preprocessor substitutes them
# before parsing, so the emitted binary is byte-identical to the raw form.
# NOTE: Jinja runs before comment stripping, so '#' comments must not
# contain Jinja delimiters -- the preprocessor would try to execute them.
#
# Memory layout (rows; issue #179 -- XMEM operands are ROW numbers):
#   SRC:  1 x 240 rows
#   ONES: 1 row
#   DST:  4 x 240 rows, one channel per row

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set slot0      = "lr0"  %}  {# const 0: r_cyclic slot 0 / mask / acc.stride slot 0 #}
{% set src_off    = "lr4"  %}  {# src row offset within the stripe, += 1 per ch #}
{% set src_stride = "lr5"  %}  {# 1 = src stride per channel (rows) #}
{% set dst_stride = "lr6"  %}  {# 1 = dst stride per channel (rows) #}
{% set dst_off    = "lr8"  %}  {# dst row offset = ch #}
{% set ch_index   = "lr10" %}  {# channel counter 0..239 #}
{% set ch_limit   = "lr11" %}  {# 240 = C #}

{% set SRC_BASE   = "cr0"  %}  {# stripe base (row 0) #}
{% set ONE        = "cr1"  %}  {# hardwired read-only 1 #}
{% set ONES_BASE  = "cr8"  %}  {# one row of 1.0, for r_cyclic init #}
{% set DST_S0     = "cr9"  %}  {# stream 0 output base #}
{% set DST_S1     = "cr10" %}  {# stream 1 output base #}
{% set DST_S2     = "cr11" %}  {# stream 2 output base #}
{% set DST_S3     = "cr12" %}  {# stream 3 output base #}
{% set DSTRUCT    = "cr15" %}  {# reserved dstructure register #}

    LDR_CYCLIC_MULT_REG {{ slot0 }} {{ ONES_BASE }} {{ slot0 }};;   # r_cyclic[0..127] = 1.0

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..239)
# ---------------------------------------------------------------------------

    LDR_MULT_REG        r0 {{ src_off }} {{ SRC_BASE }};;   # prime: channel 0

ch_loop:

    # One stripe row in r0 serves all four streams. Each stream MULTs it
    # through by 1.0 and decimates with its own (h, v) selector pair into
    # r_acc slot 0, then stages and stores in one word.

    # -- Stream 0  (h=on=even cols, v=on=even spatial rows) ------------------
    MULT.RC.VV {{ slot0 }} r0 0 {{ slot0 }} {{ DSTRUCT }};
    ACC.STRIDE 16 on on {{ slot0 }};;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ dst_off }} {{ DST_S0 }};;

    # -- Stream 1  (h=on_inv=odd cols, v=on=even spatial rows) ---------------
    MULT.RC.VV {{ slot0 }} r0 0 {{ slot0 }} {{ DSTRUCT }};
    ACC.STRIDE 16 on_inv on {{ slot0 }};;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ dst_off }} {{ DST_S1 }};;

    # -- Stream 2  (h=on=even cols, v=on_inv=odd spatial rows) ---------------
    MULT.RC.VV {{ slot0 }} r0 0 {{ slot0 }} {{ DSTRUCT }};
    ACC.STRIDE 16 on on_inv {{ slot0 }};;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ dst_off }} {{ DST_S2 }};;

    # -- Stream 3  (h=on_inv=odd cols, v=on_inv=odd spatial rows) ------------
    MULT.RC.VV {{ slot0 }} r0 0 {{ slot0 }} {{ DSTRUCT }};
    ACC.STRIDE 16 on_inv on_inv {{ slot0 }};;
    # src_off must advance BEFORE the load that uses the next channel's offset:
    # LR sub-slots run ahead of LOAD within a word, so this ADD gets its own word.
    ADD                 {{ src_off }} {{ src_off }} {{ src_stride }};;
    # Prefetch channel ch+1 into r0 for the next iteration's first MULT.
    # dst_off is read LIVE by the store, so it is NOT touched in this word.
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    STR_POST_AAQ_REG {{ dst_off }} {{ DST_S3 }};
    LDR_MULT_REG r0 {{ src_off }} {{ SRC_BASE }};;

    # -- Advance pointers; loop ----------------------------------------------
    ADD                 {{ dst_off }} {{ dst_off }} {{ dst_stride }};;
    ADD                 {{ ch_index }} {{ ch_index }} {{ ONE }};;
    BLT                 {{ ch_index }} {{ ch_limit }} ch_loop;;

end:
    BKPT;;
