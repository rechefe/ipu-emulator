# =============================================================================
# EXERCISE SOLUTION -- conv5_solution.asm  (based on ../conv.asm)
# =============================================================================
# 5-tap "valid" 1D convolution (stride 1, no padding):
#
#     out[r][p] = w[-2]*in[r][p-2] + w[-1]*in[r][p-1] + w[0]*in[r][p]
#                 + w[1]*in[r][p+1] + w[2]*in[r][p+2]
#
# for output positions p = 2 .. 125 (two edge lanes on each side would need
# an out-of-bounds neighbour, same "skip the edges" convention as
# ../conv.asm -- see that file's "WHY NO MASKING" note).
#
# Do not read this until you've attempted exercise/conv5_stub.asm yourself --
# see that file for the task description and self-check instructions.
#
# CR REGISTERS (set by the harness -- exercise/__init__.py):
#   cr0..cr10 : same as conv.asm (see that file's CR map)
#   cr11 = w[-2]  (outer-left tap weight, low byte)
#   cr12 = w[2]   (outer-right tap weight, low byte)
#
# LR REGISTERS: same as conv.asm, plus
#   lr11 = rc_idx for the OUTER-LEFT tap  (constant -2)
#   lr12 = rc_idx for the OUTER-RIGHT tap (constant +2)

# =============================================================================
# Initialization
# =============================================================================
    SET     lr0 cr0;                 # lr0 = 0 -- R_CYCLIC slot index
    SET     lr1 cr0;;                # lr1 = 0 -- row cursor / counter starts at 0

    SET     lr2 cr0;                 # lr2 = 0 -- output row cursor starts at 0
    SET     lr3 cr6;;                # lr3 = num_rows (loop bound, as an LR)

    # Constant rc_idx values for the five taps: -2, -1, 0, +1, +2.
    SUB     lr8 lr0 cr1;              # lr8  = 0 - 1 = -1 (tap -1 rc_idx)
    SET     lr9 cr0;                  # lr9  = 0        (center tap rc_idx)
    ADD     lr10 lr0 cr1;;            # lr10 = 0 + 1 = +1 (tap +1 rc_idx)

    SUB     lr11 lr8 cr1;             # lr11 = -1 - 1 = -2 (tap -2 rc_idx)
    ADD     lr12 lr10 cr1;;           # lr12 = +1 + 1 = +2 (tap +2 rc_idx)

    # Prime the pipeline: load row 0 into R_CYCLIC now so it's ready for the
    # first MULT.RC.VE next cycle (see the pipelining note in conv.asm).
    LDR_CYCLIC_MULT_REG lr1 cr2 lr0;;

# =============================================================================
# Main loop: one iteration per input row -> one 5-tap conv over all 128 lanes
# =============================================================================
loop:
    # OUTER-LEFT tap: shift by -2, multiply by w[-2], and OVERWRITE r_acc
    # (ACC.ADD.FIRST -- this is the first of the five taps).
    MULT.RC.VE      lr11 cr11 0 lr0 cr15;
    ACC.ADD.FIRST;;

    # LEFT tap: shift by -1, multiply by w[-1], accumulate.
    MULT.RC.VE      lr8 cr8 0 lr0 cr15;
    ACC.ADD;;

    # CENTER tap: shift 0 (no shift), multiply by w[0], accumulate.
    MULT.RC.VE      lr9 cr9 0 lr0 cr15;
    ACC.ADD;;

    # RIGHT tap: shift by +1, multiply by w[1], accumulate.
    MULT.RC.VE      lr10 cr10 0 lr0 cr15;
    ACC.ADD;;

    # OUTER-RIGHT tap: shift by +2, multiply by w[2], accumulate. r_acc now
    # holds the full 5-tap sum for every lane. While this runs, also advance
    # lr1 to the next row and preload it -- keeping the pipeline full across
    # iterations, exactly like conv.asm.
    MULT.RC.VE      lr12 cr12 0 lr0 cr15;
    ACC.ADD;
    ADD             lr1 lr1 cr4;;

    LDR_CYCLIC_MULT_REG lr1 cr2 lr0;;

    # Store the full 128-lane r_acc (INT32) for this row. The harness only
    # looks at output lanes 2..124 -- lanes 0, 1, 126, 127 hold
    # wrapped-neighbour values and are not valid conv output.
    STR_ACC_REG     lr2 cr3;;

    ADD             lr2 lr2 cr5;;
    BLT             lr1 lr3 loop;;

end:
    BKPT;;
