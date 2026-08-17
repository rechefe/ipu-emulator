# =============================================================================
# EXERCISE SOLUTION -- residual_sub_solution.asm  (based on ../residual_add.asm)
# =============================================================================
# Computes out[v] = A[v] - B[v] instead of A[v] + B[v]. The only change vs.
# ../residual_add.asm is the second ACC op in the loop body: ACC.ADD -> ACC.SUB.
#
# Do not read this until you've attempted exercise/residual_sub_stub.asm
# yourself -- see that file for the task description and self-check
# instructions.

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

    # B[v] is now in R_CYCLIC. Multiply-by-1 again and ACC.SUB (subtract from
    # the running r_acc, not accumulate) so r_acc = A[v] - B[v]. While that's
    # happening, also advance lr1 to the NEXT vector and preload A[v+1] so
    # it's ready when the loop comes back around.
    ADD                  lr1 lr1 cr5;;

    LDR_CYCLIC_MULT_REG  lr1 cr2 lr0;
    MULT.RC.VE           lr0 cr1 0 lr0 cr15;
    ACC.SUB;;

    # Store the INT32 result (r_acc is always 128 x INT32 = 512 bytes,
    # regardless of the INT8 input width) to the output row for this vector.
    STR_ACC_REG          lr2 cr4;;

    ADD                  lr2 lr2 cr7;;
    BLT                  lr1 lr3 loop;;

end:
    BKPT;;
