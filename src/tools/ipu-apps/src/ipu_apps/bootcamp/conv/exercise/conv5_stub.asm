# =============================================================================
# EXERCISE -- conv5_stub.asm  (based on ../conv.asm)
# =============================================================================
#
# YOUR TASK
# ---------
# ../conv.asm computes a 3-tap "valid" convolution:
#
#     out[r][p] = w[-1]*in[r][p-1] + w[0]*in[r][p] + w[1]*in[r][p+1]
#
# Widen it to a 5-tap convolution:
#
#     out[r][p] = w[-2]*in[r][p-2] + w[-1]*in[r][p-1] + w[0]*in[r][p]
#                 + w[1]*in[r][p+1] + w[2]*in[r][p+2]
#
# Recall the key trick from conv.asm: MULT.RC.VE's rc_idx is a FREE
# whole-vector shift -- rc_idx=-1 reads R_CYCLIC shifted left by one lane,
# rc_idx=+1 shifted right by one, etc. So a 5-tap conv is just FIVE
# MULT.RC.VE + ACC pairs (rc_idx = -2, -1, 0, +1, +2) instead of three.
#
# WHAT YOU NEED TO ADD
# ----------------------
# 1. Two more CR tap-weight registers: cr11 = w[-2], cr12 = w[2]
#    (the harness -- exercise/__init__.py -- already sets these for you,
#    mirroring how conv.asm's harness sets cr8/cr9/cr10 for the 3 middle
#    taps; you only need to touch the .asm).
# 2. Two more constant rc_idx LRs: lr11 = -2, lr12 = +2 (lr8/lr9/lr10 already
#    hold -1/0/+1, same as conv.asm -- see the "SUB/SET/ADD" trio below).
# 3. Two more MULT.RC.VE / ACC.ADD pairs in the loop body, at rc_idx=-2 and
#    rc_idx=+2, using tap weights cr11 and cr12.
#
# The valid output range SHRINKS from p=1..126 to p=2..125 (two edge lanes
# on each side are now wrapped/out-of-range instead of one) -- the harness's
# VALID_LO/VALID_HI already reflect this, you don't need to change anything
# outside this .asm.
#
# Find the "# TODO" markers below and fill them in. Everything else (LR
# bookkeeping, XMEM addressing, pipelining) is identical to conv.asm.
#
# HOW TO CHECK YOUR WORK
# -----------------------
#   PYTHONPATH=... python -m pytest \
#     src/tools/ipu-apps/test/test_bootcamp_conv_exercise.py -v
# test_conv5_solution_matches_numpy exercises the answer key
# (exercise/conv5_solution.asm) against a numpy 5-tap valid-conv reference
# (p = 2..125). Compare your filled-in stub's behaviour against it.

# =============================================================================
# Initialization
# =============================================================================
    SET     lr0 cr0;                 # lr0 = 0 -- R_CYCLIC slot index
    SET     lr1 cr0;;                # lr1 = 0 -- row cursor / counter starts at 0

    SET     lr2 cr0;                 # lr2 = 0 -- output row cursor starts at 0
    SET     lr3 cr6;;                # lr3 = num_rows (loop bound, as an LR)

    # Constant rc_idx values for the three MIDDLE taps: -1, 0, +1 (unchanged
    # from conv.asm).
    SUB     lr8 lr0 cr1;              # lr8  = 0 - 1 = -1 (tap -1 rc_idx)
    SET     lr9 cr0;                  # lr9  = 0        (center tap rc_idx)
    ADD     lr10 lr0 cr1;;            # lr10 = 0 + 1 = +1 (tap +1 rc_idx)

    # TODO: synthesize the two new OUTER rc_idx constants the same way conv.asm
    # builds -1/+1 above (SUB/ADD against lr0=0 and cr1=1 won't directly give
    # you +/-2 in one instruction -- think about what you can add/subtract
    # against an LR that already holds +/-1 to reach +/-2).
    #   lr11 = -2  (tap -2 rc_idx)
    #   lr12 = +2  (tap +2 rc_idx)

    # Prime the pipeline: load row 0 into R_CYCLIC now so it's ready for the
    # first MULT.RC.VE next cycle (see the pipelining note in conv.asm).
    LDR_CYCLIC_MULT_REG lr1 cr2 lr0;;

# =============================================================================
# Main loop: one iteration per input row -> one 5-tap conv over all 128 lanes
# =============================================================================
loop:
    # TODO: eventually you'll want the outer-LEFT tap (rc_idx=-2,
    # weight=cr11) to be the FIRST MULT.RC.VE/ACC pair (ACC.ADD.FIRST, since
    # it should start the sum). As scaffolding, the LEFT tap below still
    # does ACC.ADD.FIRST for now (unchanged from conv.asm) so this stub
    # keeps working as a plain 3-tap conv until you add the outer taps --
    # once you add the outer-left tap, MOVE the ACC.ADD.FIRST there and
    # change this LEFT tap to ACC.ADD (it becomes the second of five taps,
    # not the first).
    #   MULT.RC.VE      lr11 cr11 0 lr0 cr15;
    #   ACC.ADD.FIRST;;

    # LEFT tap: shift by -1, multiply by w[-1]. Starts the sum (ACC.ADD.FIRST)
    # -- see the TODO above: this must become ACC.ADD once you add the
    # outer-left tap in front of it.
    MULT.RC.VE      lr8 cr8 0 lr0 cr15;
    ACC.ADD.FIRST;;

    # CENTER tap: shift 0 (no shift), multiply by w[0], accumulate.
    MULT.RC.VE      lr9 cr9 0 lr0 cr15;
    ACC.ADD;;

    # RIGHT tap: shift by +1, multiply by w[1], accumulate.
    MULT.RC.VE      lr10 cr10 0 lr0 cr15;
    ACC.ADD;;

    # TODO: add the outer-RIGHT tap here (rc_idx=+2, weight=cr12), accumulate
    # into r_acc (ACC.ADD, not ACC.ADD.FIRST -- four taps already summed).
    # While this runs, also advance lr1 to the next row and preload it --
    # keeping the pipeline full across iterations, exactly like conv.asm.
    #   MULT.RC.VE      lr12 cr12 0 lr0 cr15;
    #   ACC.ADD;
    ADD             lr1 lr1 cr4;;

    LDR_CYCLIC_MULT_REG lr1 cr2 lr0;;

    # Store the full 128-lane r_acc (INT32) for this row. The harness only
    # looks at output lanes 2..124 (valid range for a 5-tap conv) -- lanes
    # 0, 1, 126, 127 hold wrapped-neighbour garbage and are not valid output.
    STR_ACC_REG     lr2 cr3;;

    ADD             lr2 lr2 cr5;;
    BLT             lr1 lr3 loop;;

end:
    BKPT;;
