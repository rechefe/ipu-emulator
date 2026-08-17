# =============================================================================
# BOOTCAMP DRILL 2/3 -- conv.asm
# Difficulty: ** (adds MULT + ACC "for real" -- a genuine multiply-accumulate
#                 sliding window, not the multiply-by-one placeholder from
#                 drill 1)
# =============================================================================
#
# WHAT THIS PROGRAM DOES
# -----------------------
# A single-channel, 1D, 3-tap "valid" convolution (stride 1, no padding) over
# `num_rows` independent rows of 128 INT8 elements each:
#
#     out[r][p] = w[-1]*in[r][p-1] + w[0]*in[r][p] + w[1]*in[r][p+1]
#
# computed for every output position `p` that has a full 3-element
# neighbourhood, i.e. p = 1 .. 126 (positions 0 and 127 would need an
# out-of-bounds neighbour, so this drill skips them rather than pad/mask them
# -- see "WHY NO MASKING" below). This is exactly the sliding-window
# multiply-accumulate at the heart of every convolutional layer -- just
# reduced to one channel, one small kernel, and one dimension so the
# MULT.RC.VE + ACC.ADD/ACC.ADD.FIRST idiom is easy to see.
#
# THE KEY TRICK: A "SHIFT" IS FREE ON THIS DATAPATH
# ----------------------------------------------------
# MULT.RC.VE multiplies ALL 128 lanes of R_CYCLIC, starting at element index
# `rc_idx`, by one scalar. Critically, `rc_idx` does not have to be 0 -- if
# you pass rc_idx = -1, lane i of the result is R_CYCLIC[i - 1] * scalar; if
# you pass rc_idx = +1, lane i of the result is R_CYCLIC[i + 1] * scalar.
# In other words, changing rc_idx shifts the ENTIRE 128-lane vector for free,
# with no separate "shift" instruction. So one 3-tap conv is just THREE
# MULT.RC.VE + ACC calls, each with a different rc_idx (-1, 0, +1) and a
# different scalar tap weight -- no per-output-element loop needed at all:
#
#     rc_idx=-1, scalar=w[-1]:  ACC.ADD.FIRST   -> r_acc[i]  = w[-1]*in[i-1]
#     rc_idx= 0, scalar=w[0] :  ACC.ADD         -> r_acc[i] += w[0]*in[i]
#     rc_idx=+1, scalar=w[1] :  ACC.ADD         -> r_acc[i] += w[1]*in[i+1]
#
# After these three cycles, r_acc[i] holds the full 3-tap sum for EVERY lane
# i in one shot -- the "sliding window" comes from the shifted rc_idx, not
# from looping over output positions.
#
# WHY NO MASKING (the ISA has a masking mechanism, but this drill skips it)
# -----------------------------------------------------------------------------
# R_CYCLIC is a 512-element ring; a load only fills elements 0..127 (one
# "slot") with real row data, so reading rc_idx=-1 wraps around and reads
# element 511 (never written -- garbage) into lane 0, and rc_idx=+1 wraps lane
# 127 to read element 0 (real data, but the WRONG neighbour -- there is no
# element 128). Real conv kernels use R_MASK + mask_shift to zero out exactly
# these wrapped/out-of-range lanes (see conv_first_layer for that machinery).
# This drill sidesteps it entirely: lanes 0 and 127 of r_acc are simply never
# read back by the harness (only output positions 1..126 are stored/checked)
# -- a genuine "valid" convolution, just implemented by discarding the two
# edge lanes in software instead of masking them in hardware. Drill 3
# (softmax) is where you'll meet real multi-CR/AAQ machinery; this drill
# stays scoped to MULT + ACC.
#
# PIPELINE TIMING (same rule as drill 1 -- see residual_add.asm for the full
# explanation): MULT reads R_CYCLIC from the cycle-start SNAPSHOT, so the row
# must be loaded ONE CYCLE before the first MULT.RC.VE that reads it.
#
# TAP WEIGHTS: SCALAR, VIA CR REGISTERS
# ----------------------------------------
# MULT.RC.VE's scalar source can be a CR register directly (its low byte is
# read as the scalar) -- no LR indirection needed, since these tap weights
# are small compile-time-ish constants for the whole run, not per-row data.
# The harness sets CR8/CR9/CR10 to w[-1]/w[0]/w[1].
#
# XMEM ADDRESSING: rows, not bytes (see residual_add.asm's note -- identical
# convention here). One 128-element INT8 row is one XMEM row; STR_ACC_REG's
# 512-byte (4-row) output write needs its own output row stride, same as
# drill 1.
#
# CR REGISTERS (set by the harness):
#   cr0  = 0            (hardware-reserved: always 0)
#   cr1  = 1             (hardware-reserved: always 1; unused directly here)
#   cr2  = input base row
#   cr3  = output base row
#   cr4  = 1             (input row step)
#   cr5  = 4             (output row step -- STR_ACC_REG always writes 512 B/4 rows)
#   cr6  = num_rows (loop bound)
#   cr8  = w[-1]  (left tap weight, low byte)
#   cr9  = w[0]   (center tap weight, low byte)
#   cr10 = w[1]   (right tap weight, low byte)
#
# LR REGISTERS:
#   lr0  = 0   (R_CYCLIC slot index -- always slot 0)
#   lr1  = input row cursor / row counter (0 .. num_rows-1)
#   lr2  = output row cursor (0, 4, 8, ... -- steps by cr5)
#   lr3  = num_rows, as an LR (BLT's bound source)
#   lr8  = rc_idx for the LEFT tap  (constant -1, held once outside the loop)
#   lr9  = rc_idx for the CENTER tap (constant 0)
#   lr10 = rc_idx for the RIGHT tap  (constant +1)

# =============================================================================
# Initialization
# =============================================================================
    SET     lr0 cr0;                 # lr0 = 0 -- R_CYCLIC slot index
    SET     lr1 cr0;;                # lr1 = 0 -- row cursor / counter starts at 0

    SET     lr2 cr0;                 # lr2 = 0 -- output row cursor starts at 0
    SET     lr3 cr6;;                # lr3 = num_rows (loop bound, as an LR)

    # Constant rc_idx values for the three taps: -1, 0, +1. SUB/ADD against
    # LR0(=0) is just a convenient way to get a -1/+1 constant into an LR
    # (there's no immediate-load instruction for arbitrary signed values).
    SUB     lr8 lr0 cr1;              # lr8  = 0 - 1 = -1 (left tap rc_idx)
    SET     lr9 cr0;                  # lr9  = 0        (center tap rc_idx)
    ADD     lr10 lr0 cr1;;            # lr10 = 0 + 1 = +1 (right tap rc_idx)

    # Prime the pipeline: load row 0 into R_CYCLIC now so it's ready for the
    # first MULT.RC.VE next cycle (see the pipelining note above).
    LDR_CYCLIC_MULT_REG lr1 cr2 lr0;;

# =============================================================================
# Main loop: one iteration per input row -> one 3-tap conv over all 128 lanes
# =============================================================================
loop:
    # Row r is already in R_CYCLIC (loaded last cycle). LEFT tap: shift by -1,
    # multiply by w[-1], and OVERWRITE r_acc (ACC.ADD.FIRST -- this is the
    # first of the three taps, so there is no previous partial sum to keep).
    MULT.RC.VE      lr8 cr8 0 lr0 cr15;
    ACC.ADD.FIRST;;

    # CENTER tap: shift 0 (no shift), multiply by w[0], accumulate.
    MULT.RC.VE      lr9 cr9 0 lr0 cr15;
    ACC.ADD;;

    # RIGHT tap: shift by +1, multiply by w[1], accumulate. r_acc now holds
    # the full 3-tap sum for every lane. While this runs, also advance lr1 to
    # the next row and preload it -- keeping the pipeline full across
    # iterations, exactly like drill 1.
    MULT.RC.VE      lr10 cr10 0 lr0 cr15;
    ACC.ADD;
    ADD             lr1 lr1 cr4;;

    LDR_CYCLIC_MULT_REG lr1 cr2 lr0;;

    # Store the full 128-lane r_acc (INT32) for this row. The harness will
    # only look at output lanes 1..126 -- lanes 0 and 127 hold the
    # wrapped-neighbour values described above and are not valid conv output.
    STR_ACC_REG     lr2 cr3;;

    # Bump the output cursor, then test the loop bound against lr1 (already
    # advanced to r+1 above) -- same two-separate-words rule as drill 1: COND
    # reads its operands from the cycle-start snapshot, so a same-word write
    # would be compared against the stale value.
    ADD             lr2 lr2 cr5;;
    BLT             lr1 lr3 loop;;

end:
    BKPT;;
