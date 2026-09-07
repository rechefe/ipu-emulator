# Fold 8x8x240: 4 stride-2 channel-major streams -> spatial 8x8x240   (L5, inverse of unfold)
#
# Layer:   L5
# Scope:   single-stream (consumes the 4 stride-2 spatial streams)
# Layout:  unpacked
# Shape:   4 streams x [16tok, 240chan] -> 8x8x240 spatial output (naive, unpacked)
# Status:  new (L5 sibling of fold_16x16x192/fold_32x32x144; geometry
#          re-derived, not ported)
# Related: exact algebraic inverse of unfold_8x8x240's DECIMATION (not its
#          on-disk input packing -- see the OUTPUT LAYOUT note below for why
#          this kernel's output is naive [H, W] row-major, NOT
#          unfold_8x8x240._ROW_PACK_ORDER-permuted). Verified
#          fold(unfold(pack_input_rows(x))) == x bit-for-bit (mod FP32
#          rounding from the x1.0 pass-through multiply, which is exact) in
#          test_fold_8x8x240_wide.py.
# Tests:   test_fold_8x8x240_wide (src/tools/ipu-apps/BUILD.bazel)
#
# Rearranges 4 stride-2 decimated streams (the standard stride-2
# space-to-depth decomposition unfold_8x8x240 produces -- stream s takes
# phase (s // 2, s % 2) of every 2x2 block, NOT an image quadrant) back into
# an 8x8x240 spatial tensor, one row per channel, 64 valid lanes in NAIVE
# row-major order (lane = row*8 + col).
#
# Input (per-stream channel-major FP32, matches unfold_8x8x240's raw,
# UNCROPPED per-channel row output -- i.e. the ".rows.bin" sibling that
# kernel's own teardown() writes, not its cropped [N_STREAMS,N_OUT,N_TOK]
# convenience array):
#   4 streams x 240 rows x 512 bytes, stream s based at SRC_s.
#   Stream s, ch c: at SRC_s + c row.
#   *** Only the first 16 FP32 lanes (64 bytes) of each row are valid data;
#   *** lanes 16..127 are unfold's stale r_acc padding and MUST be ignored --
#   this kernel never reads past lane 15 of a loaded stream row.
#   Stream s row layout: 16 tokens in row-major order over the 4x4 decimated
#   grid (dr, dc), i.e. src_lane = dr*4 + dc, dr,dc in 0..3 -- confirmed
#   empirically (see below), NOT dependent on unfold_8x8x240's internal
#   _ROW_PACK_ORDER input-packing trick.
#
# Output (naive spatial, NOT unfold_8x8x240's packed INPUT layout):
#   240 rows x 512 bytes; row ch holds channel ch's 8x8 grid in the first 64
#   lanes, TRUE row-major order (lane = row*8 + col); lanes 64..127 are
#   whatever ACTIVATE.QUANTIZE/ACC.RESHAPE happened to leave in r_acc
#   (never written by this kernel's 8 ACC.RESHAPE calls per channel) and are
#   NOT part of this kernel's output contract.
#
# OUTPUT LAYOUT -- a deliberate departure from L3/L4's convention, decided
# and verified BEFORE writing this .asm (per the task brief's explicit
# open question): L3 (fold_32x32x144) and L4 (fold_16x16x192) both produce
# output that matches unfold's OWN INPUT layout byte-for-byte. For L5 that
# would mean reproducing unfold_8x8x240._ROW_PACK_ORDER's view-row
# permutation on the way OUT. But ACC.RESHAPE's dest indices are arbitrary
# bytes in [0, 127] -- unlike ACC.STRIDE's fixed view-row/view-col
# structure -- so there is no structural reason fold's scatter has to target
# the packed layout; it can target TRUE spatial (row, col) positions
# directly. Both options were derived and brute-force partition-checked in
# Python before writing this kernel:
#   packed-output dest table (would mirror unfold's input contract): the 64
#     valid destination lanes partition cleanly, uniform +32 per-call
#     stride, call-0 bases 0/1/16/17 for streams 0/1/2/3.
#   naive-output dest table (chosen): ALSO partitions cleanically with a
#     uniform +32 per-call stride, call-0 bases 0/1/8/9 for streams 0/1/2/3
#     -- see the ACC.RESHAPE MAPPING section below.
# Both are equally simple to encode; naive row-major was chosen because it
# is a genuine, more directly useful [H, W] spatial tensor (no downstream
# consumer has to know about _ROW_PACK_ORDER to interpret fold's output),
# and because the task's stated hypothesis -- that fold does not need an
# output-side inverse of the input-side packing trick -- holds. Fold's
# SOURCE-side interpretation still implicitly accounts for
# _ROW_PACK_ORDER: it consumes unfold_8x8x240's OUTPUT streams, which unfold
# itself already de-packed into real-coordinate stride-2 phases via its
# (h, v) ACC.STRIDE selectors -- fold never sees a packed row directly.
#
# STREAM -> REAL (row, col) MAPPING -- verified empirically, NOT assumed
# (per the task brief and the bug #1 lesson from fold_16x16x192: never trust
# textual/derived layout claims about a MULT-snapshot-affected kernel's
# stream contents without checking the real emulator output). A per-channel,
# per-lane marker tensor (value = ch*1e6 + row*1000 + col) was packed with
# unfold_8x8x240.pack_input_rows and run through the REAL unfold_8x8x240
# kernel binary; the resulting stream outputs were decoded lane-by-lane and
# matched EXACTLY the naive expectation
# ``x[ch, r_ph::2, c_ph::2].reshape(-1)`` for stream s (r_ph=s//2, c_ph=s%2)
# -- i.e. unfold_8x8x240's own (h, v) ACC.STRIDE selectors already fully
# resolve _ROW_PACK_ORDER's view-row permutation internally; nothing about
# it leaks into the stream's real-coordinate meaning. This matches (and
# empirically re-confirms, rather than just trusts) what
# test_unfold_8x8x240_wide.py's own reference computation already assumes.
#
# ACC.RESHAPE MAPPING (derived by inverting the stride-2 decimation with the
# naive dest-lane formula lane = row*8+col; verified by brute-force
# partition check in Python before writing this kernel):
#   Each stream contributes 16 tokens (a 4x4 decimated grid, row-major:
#   src_lane = dr*4+dc) to 16 of the destination row's 64 valid lanes, via 2
#   ACC.RESHAPE calls of 8 elements each:
#     source (same shape for every stream): [8c, 8c+1, ..., 8c+7] for call
#       c in 0..1 (source table is simply [0..7] then [8..15], no
#       stripe/tg-style window shifting needed -- there is only one stripe).
#     dest, stream s, call c: base(s) + step, where
#       base(s0)=0, base(s1)=1, base(s2)=8, base(s3)=9 (s1=s0+1, s2=s0+8,
#       s3=s0+9), and the per-call dest tables (relative to base) are:
#         call0 = base + [0,2,4,6,16,18,20,22]
#         call1 = base + [32,34,36,38,48,50,52,54]   (call0 + 32, uniform)
#   These 8 (stream, call) dest tables partition destination lanes 0..63
#   exactly once each -- confirmed by brute-force enumeration in Python
#   before writing this kernel (64 unique destination indices, range 0..63,
#   zero overlap). Lanes 64..127 are never targeted by any ACC.RESHAPE call
#   in this kernel (correctly -- there are only 64 real spatial elements per
#   channel).
#
# Building the index tables (per user directive: never assume an LR holds
# anything but garbage at kernel start -- unlike CRs, which setup()
# legitimately initializes via state.regfile.set_cr):
#   SOURCE table (LRD2 = LR3:LR2) is rebuilt via two SETs from cr2/cr3
#   (packed bytes [0,1,2,3] / [4,5,6,7]) at the start of every stream's
#   2-call block, then walked forward +8 per call via ADDBI.
#
#   DEST table (LRD4 = LR5:LR4) is rebuilt via two SETs from cr4/cr5
#   (stream-0's call-0 table, packed bytes [0,2,4,6] / [16,18,20,22]) then,
#   for streams 1/2/3, walked to that stream's base via a single ADDBI
#   (+1, +8, +9 respectively). Across the 2 calls of one stream, the table
#   is walked +32 (uniform, unlike fold_32x32x144's +16/+48/+16 -- L5's
#   simpler 4x4 decimated grid produces a uniform stride here, closer to
#   fold_16x16x192's pattern). The dest table is rebuilt once per stream per
#   channel purely for auditability (correctness-first per the task brief; a
#   cycle-tuned version could cache it across channels).
#
# CRs:
#   cr13 = DST_BASE_ROW            (output base, ch 0..239)
#   -- NOTE: CR0 is ALSO read-only hardwired to 0 (not just CR1 -- see
#      CR_READ_ONLY_INITIAL_VALUES in ipu_config.py); unfold_8x8x240 gets
#      away with naming its SOURCE base cr0 only because that base happens
#      to equal 0 anyway. Fold's DESTINATION base is DST_BASE_ROW (nonzero,
#      sitting after the 4-stream source region), so it cannot use cr0 --
#      same trap fold_16x16x192/fold_32x32x144 document. cr13 is used
#      instead (arbitrary choice among the free CRs).
#   cr8  = ONES_BASE                (128 elements of dtype 1.0, for r_cyclic init)
#   cr9  = SRC_0                    (stream 0 input base, row 0)
#   cr10 = SRC_1                    (stream 1 input base, row 240)
#   cr11 = SRC_2                    (stream 2 input base, row 480)
#   cr12 = SRC_3                    (stream 3 input base, row 720)
#   cr2  = source table lo  bytes [0,1,2,3]      (LRD2 low word,  via SET)
#   cr3  = source table hi  bytes [4,5,6,7]      (LRD2 high word, via SET)
#   cr4  = stream-0 dest table lo bytes [0,2,4,6]        (LRD4 low word,  via SET)
#   cr5  = stream-0 dest table hi bytes [16,18,20,22]    (LRD4 high word, via SET)
#   cr1  = 1     (read-only hardwired constant; channel-loop increment)
#
# LRs (preset by harness):
#   lr0  = 0    (const: r_cyclic slot 0)
#   lrd2 = LR2:LR3  (working source-lane-index table for ACC.RESHAPE)
#   lrd4 = LR4:LR5  (working dest-lane-index table for ACC.RESHAPE)
#   lr6  = 0    (src row offset within a stream; += 1 per channel)
#   lr7  = 1    (src stride per channel, in rows)
#   lr8  = 0    (dst row offset = ch; += 1 per channel)
#   lr9  = 1    (dst stride per channel, in rows)
#   lr10 = 0    (ch counter, 0..239)
#   lr11 = 240  (loop limit = C)
#
# Memory layout (ROW numbers -- .asm XMEM operands are rows, issue #179):
#   SRC:  4 streams x 240 rows                      (s0, s1, s2, s3)
#   ONES: 1 row
#   DST:  240 rows
#
# MULT SNAPSHOT CONTRACT (issue #157, inherited from fold_16x16x192 and
# fold_32x32x144): a MULT.RC.VV instruction reads its Ra DATA (r0) from the
# start-of-cycle snapshot, so it cannot consume a chunk loaded by an
# LDR_MULT_REG issued in the very same bundle -- co-issuing a load and a
# MULT in one bundle is fine only when the MULT consumes a PREVIOUSLY loaded
# chunk. This kernel primes the first load before ch_loop and keeps every
# subsequent LDR one bundle ahead of the MULT that consumes it. Rotation:
# stream0 block prefetches stream1's row (same channel), stream1 prefetches
# stream2, stream2 prefetches stream3, and stream3 prefetches stream0 of the
# NEXT channel (advancing lr6 first, in its own word, before that load).
#
# STR_POST_AAQ_REG reads its offset operand LIVE -- never co-issue an ADD on
# the offset LR (lr8) in the same bundle as the store, or the row lands one
# slot late. Here lr8 only advances in its own bundle at the very end of the
# channel loop, well clear of the store.
{% set groups = [
     ("s0", 0, "cr9",  "cr10"),
     ("s1", 1, "cr10", "cr11"),
     ("s2", 8, "cr11", "cr12"),
     ("s3", 9, "cr12", "cr9"),
   ] %}

    LDR_CYCLIC_MULT_REG lr0 cr8 lr0;;       # r_cyclic[0..127] = 1.0 (dtype-specific)

    LDR_MULT_REG        r0 lr6 cr9;;        # prime: stream 0 row of ch 0

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..239)
# ---------------------------------------------------------------------------

ch_loop:

{% for g in groups %}
{%- set name = g[0] %}
{%- set dest_base = g[1] %}
{%- set next_cr = g[3] %}
    # -- Stream {{ name }}: dest table = base{% if dest_base != 0 %} + {{ dest_base }}{% endif %} -----------------------------------
    SET                 lr2 cr2;
    SET                 lr3 cr3;;           # source table <- [0,1,2,3,4,5,6,7]
    SET                 lr4 cr4;
    SET                 lr5 cr5;;           # dest table   <- stream-0 base
{%- if dest_base != 0 %}
    ADDBI               lrd4 {{ dest_base }};;      # dest table <- {{ name }} base
{%- endif %}
{%- if name == "s3" %}
    # lr6 (src offset) must advance BEFORE the load that uses next channel's
    # offset: LR sub-slots run ahead of LOAD within a word (same hazard
    # fold_16x16x192/fold_32x32x144 document for their src-offset LR), so
    # this ADD gets its own word.
    ADD                 lr6 lr6 lr7;;       # src offset: next channel (+1 row)
{%- endif %}
    LDR_MULT_REG r0 lr6 {{ next_cr }};
    MULT.RC.VV lr0 r0 0 lr0 cr15;;          # multiply {{ name }} row (prefetch next stream)
    ACC.RESHAPE lrd2 lrd4 0;;               # call0: src[0..7] -> dest[{{ dest_base }},{{ dest_base+2 }},{{ dest_base+4 }},{{ dest_base+6 }},{{ dest_base+16 }},{{ dest_base+18 }},{{ dest_base+20 }},{{ dest_base+22 }}]
    ADDBI               lrd2 8;
    ADDBI               lrd4 32;;
    ACC.RESHAPE lrd2 lrd4 0;;               # call1: src[8..15] -> dest[+32 of call0]
{% endfor %}
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG         lr8 cr13;;     # ch <- DST_BASE_ROW + ch (current channel)

    # -- Advance pointers; loop -------------------------------------------------
    ADD                 lr8 lr8 lr9;;       # dst offset: next channel
    ADD                 lr10 lr10 cr1;;
    BLT                 lr10 lr11 ch_loop;; # loop while ch < 240

end:
    BKPT;;
