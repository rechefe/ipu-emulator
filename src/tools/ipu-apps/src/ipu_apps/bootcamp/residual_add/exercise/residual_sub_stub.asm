# =============================================================================
# EXERCISE -- residual_sub_stub.asm  (based on ../residual_add.asm)
# =============================================================================
#
# YOUR TASK
# ---------
# This is a copy of residual_add.asm, which computes out[v] = A[v] + B[v].
# Change it to compute out[v] = A[v] - B[v] instead.
#
# You need to change EXACTLY ONE instruction: the ACC op that combines the
# B[v] load (the second of the two MULT.RC.VE / ACC pairs in the loop body)
# currently reads `ACC.ADD`. Swap it for `ACC.SUB` (see instruction_spec.py --
# it exists, mirroring ACC.ADD/ACC.ADD.FIRST). Find the "# TODO" marker below
# and replace it with the correct instruction.
#
# Everything else -- the pipelining, the LR bookkeeping, the XMEM addressing
# -- is identical to residual_add.asm and does not need to change.
#
# HOW TO CHECK YOUR WORK
# -----------------------
# 1. Make your edit below.
# 2. Copy your fixed body into (or otherwise compare it against)
#    exercise/residual_sub_solution.asm.
# 3. Run the self-check test:
#      PYTHONPATH=... python -m pytest \
#        src/tools/ipu-apps/test/test_bootcamp_residual_add_exercise.py -v
#    test_residual_sub_solution_matches_numpy exercises the solution file
#    (the answer key) against `a.astype(np.int32) - b.astype(np.int32)`.
#    Point your own fixed-up stub at that same test's `_run()` helper (or
#    just diff your result against residual_sub_solution.asm) to confirm.
#
# See ../residual_add.asm for the full pipelining / addressing explanation --
# it is not repeated here since nothing about it changes.

# =============================================================================
# Initialization
# =============================================================================
    SET     lr0 cr0;                 # lr0 = 0 -- the only R_CYCLIC slot we use
    SET     lr1 cr0;;                # lr1 = 0 -- input cursor starts at vector 0

    SET     lr2 cr0;;                # lr2 = 0 -- output cursor starts at vector 0
    SET     lr3 cr6;;                # lr3 = num_vectors (loop bound, as an LR)

    # Prime the pipeline: load A[0] into R_CYCLIC now so it's ready for the
    # first MULT.RC.VE (which runs next cycle, per the timing note above).
    LDR_CYCLIC_MULT_REG lr1 cr2 lr0;;

# =============================================================================
# Main loop: one iteration per 128-element vector
# =============================================================================
loop:
    # A[v] is already sitting in R_CYCLIC (loaded last cycle, or primed
    # above). Copy it into r_acc with ACC.ADD.FIRST (overwrite -- we do NOT
    # want whatever r_acc held from the previous loop iteration). In the same
    # word, kick off the load of B[v] -- it lands in R_CYCLIC next cycle.
    LDR_CYCLIC_MULT_REG lr1 cr3 lr0;
    MULT.RC.VE          lr0 cr1 0 lr0 cr15;
    ACC.ADD.FIRST;;

    # B[v] is now in R_CYCLIC. Multiply-by-1 again, then combine into r_acc.
    #
    # TODO: this drill wants out[v] = A[v] - B[v], not A[v] + B[v]. The
    # instruction below (ACC.ADD) currently ADDS B[v] into r_acc. Replace it
    # with the correct ACC op so r_acc ends up holding A[v] - B[v] instead.
    ADD                  lr1 lr1 cr5;;

    LDR_CYCLIC_MULT_REG  lr1 cr2 lr0;
    MULT.RC.VE           lr0 cr1 0 lr0 cr15;
    ACC.ADD;;
    # ^^^^^^^ TODO: fix me -- this still computes A[v] + B[v]. See the task
    # description at the top of this file.

    # Store the INT32 result (r_acc is always 128 x INT32 = 512 bytes,
    # regardless of the INT8 input width) to the output row for this vector.
    STR_ACC_REG          lr2 cr4;;

    ADD                  lr2 lr2 cr7;;
    BLT                  lr1 lr3 loop;;

end:
    BKPT;;
