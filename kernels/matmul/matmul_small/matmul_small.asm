{#- matmul_small.asm: C = A x W^T, row-major, K <= 128 and N <= 128 (E07-07).

    The template of the registry.toml entries matmul_64x64x64, matmul_128x64x64,
    matmul_128x64x128 and matmul_128x128 (one program, the E02 digest 61867660):
    the shape reaches the program only through the CRs.

    a: A, (M, K), one XMEM row per matrix row, K lanes valid (zero-padded).
    b: T = W^T, (K, N): row k is column k of the output-major weights W, N lanes
       valid. b is what LD.MULT.CYCLIC streams: one row per contraction step.
    y: C, (M, N), one XMEM row per matrix row (all 128 accumulators stored; the
       first N are C's columns).

    Per output row m: R0 = A[m]; then for k in 0..K-1, R_CYCLIC slot 0 = T[k]
    and MULT.RC.VE broadcasts the element A[m][k] (R0 at fixed_idx k) across
    T[k], accumulating into R_ACC. The first k is peeled with ACC.ADD.FIRST to
    seed R_ACC; the rest use ACC.ADD. MULT reads R_CYCLIC from the cycle-start
    snapshot (issue #157), so each k's row is loaded a word before its multiply.

    ABI (kernel.toml [abi]; every value counts ROWS or indices, not bytes):
      CR0  = a base (row 0: the hardwired 0)
      CR11 = b base                    CR2  = y base
      CR3  = 1   (fixed_idx step)      CR4  = 1   (a and b row step)
      CR5  = 1   (y row step)          CR6  = M   (outer-loop limit on the a row)
      CR7  = 0   (initial offsets, R_CYCLIC slot 0)
      CR8  = -1  (b row startup)       CR9  = -1  (fixed_idx startup)
      CR10 = K - 1 (inner-loop limit)

    LRs: LR0 = a row offset (m), LR1 = M, LR4 = b row offset (startup -1: the
    first live row is T[0]), LR5 = fixed_idx (k), LR6 = K - 1, LR7 = y row
    offset, LR12..LR14 = the steps; the program sets each. LR15 is the
    R_CYCLIC slot index 0 and is never written: the program reads its reset
    value (neither harness wrote it).
-#}

    SET                 LR12 CR3;;
    SET                 LR13 CR4;;
    SET                 LR14 CR5;;
    SET                 LR0 CR7;;
    SET                 LR1 CR6;;        # M: the outer-loop limit on the a row
    SET                 LR7 CR7;;

row_loop:
    LD.MULT        R0 LR0 CR0;;

    SET                 LR4 CR8;;
    SET                 LR5 CR9;;
    SET                 LR6 CR10;;

    # MULT reads r_cyclic from the start-of-cycle snapshot (issue #157), so the
    # chunk consumed by MULT.RC.VE must be loaded a cycle earlier than it is
    # used: prime k=0's load here, then each loop body loads the NEXT k's chunk
    # while multiplying the chunk loaded last cycle.
    # LR5 (R0 scalar-select index, read LIVE by MULT) must equal the CURRENT k on
    # the cycle MULT runs; LR4 (load offset, also read LIVE, by LDR) must already
    # hold the NEXT k's offset on that same cycle. LR sub-slots run before
    # LOAD/MULT within a word, so LR4's increment for k+1 must land in the word
    # BEFORE the load that consumes it -- it cannot share a word with that load.
    # LR5 shares a word with the MULT that consumes it, so MULT sees the k that
    # LR5 was just advanced to.
    ADD                 LR4 LR4 LR13;;            # LR4 = k=0's offset (0)
    LD.MULT.CYCLIC LR4 CR11 LR15;;           # load T[k=0] (LR4 unchanged this word)
    ADD                 LR4 LR4 LR13;
    ADD                 LR5 LR5 LR12;;            # LR4 -> k=1's offset; LR5 -> 0

    # Peeled first iteration (k=0): ACC.FIRST seeds the accumulator (replaces RESET_ACC).
    MULT.RC.VE          LR15 LR5 0 LR15 CR15;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC LR4 CR11 LR15;
    BNE                 LR5 LR6 k_loop_pre;;
    B                   after_k_loop;;

k_loop_pre:
    ADD                 LR4 LR4 LR13;
    ADD                 LR5 LR5 LR12;;            # LR4 -> next load offset; LR5 -> k just loaded

k_loop:
    MULT.RC.VE          LR15 LR5 0 LR15 CR15;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR11 LR15;
    BNE                 LR5 LR6 k_loop_pre;;

after_k_loop:
    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ         LR7 CR2;;
    ADD                 LR7 LR7 LR14;
    ADD                 LR0 LR0 LR13;;

    BREAK;;

    BLT.S                 LR0 LR1 row_loop;;

end:
    HALT;;
