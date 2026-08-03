# Matrix multiplication: C = A x W^T
# A: 64x64  input  (M=64 rows, K=64 cols)
# W: 64x64  weights output-major (N=64 rows, K=64 cols); transposed to T in XMEM
# C: 64x64  output (M=64 rows, N=64 cols, stored as 128 int32/fp32 per row)
#
# K=64: A rows zero-padded to 128 bytes. N=64: T[k] zero-padded to 128 bytes.
# Output stride 256 (N*4) packs only the valid first-half per row.
#
# New kernel: MULT.VE.CYCLIC replaces MULT.EV mem_bypass (mem_bypass removed in PR #69).
#
# Memory layout (set via CR registers):
#   cr0  = input  base (A: 64 rows x 128 bytes padded = 8192 bytes)
#   cr11 = weights base (T: 64 rows x 128 bytes padded = 8192 bytes)
#          (moved off CR1 — CR1 is now a read-only hardwired constant ≡ 1)
#   cr2 = output base  (C: 64 rows x 256 bytes packed = 16384 bytes)
#   cr3 = 1      (ADD step for fixed_idx)
#   cr4 = 128    (ADD step for weight/input strides)
#   cr5 = 256    (ADD step for output stride = N*4)
#   cr6 = 8192   (outer loop limit = M * 128 = 64 * 128)
#   cr7 = 0      (const zero)
#   cr8 = -128   (inner loop init: weight offset startup)
#   cr9 = -1     (inner loop init: fixed_idx startup)
#   cr10 = 63    (inner loop limit K-1 = 63)
#
# Differs from matmul_128x64x64.asm only in: cr6 = 8192 (M=64).

    SET                 lr12 cr3;;
    SET                 lr13 cr4;;
    SET                 lr14 cr5;;
    SET                 lr0 cr7;;
    SET                 lr1 cr6;;        # M * 128 = 64 * 128
    SET                 lr7 cr7;;

row_loop:
    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr4 cr8;;
    SET                 lr5 cr9;;
    SET                 lr6 cr10;;

    # MULT reads r_cyclic from the start-of-cycle snapshot (issue #157), so the
    # chunk consumed by MULT.RC.VE must be loaded a cycle earlier than it is
    # used: prime k=0's load here, then each loop body loads the NEXT k's chunk
    # while multiplying the chunk loaded last cycle.
    # lr5 (R0 scalar-select index, read LIVE by MULT) must equal the CURRENT k on
    # the cycle MULT runs; lr4 (load offset, also read LIVE, by LDR) must already
    # hold the NEXT k's offset on that same cycle. LR sub-slots run before
    # LOAD/MULT within a word, so lr4's increment for k+1 must land in the word
    # BEFORE the load that consumes it -- it cannot share a word with that load.
    # lr5 shares a word with the MULT that consumes it, so MULT sees the k that
    # lr5 was just advanced to.
    ADD                 lr4 lr4 lr13;;            # lr4 = k=0's offset (0)
    LDR_CYCLIC_MULT_REG lr4 cr11 lr15;;           # load T[k=0] (lr4 unchanged this word)
    ADD                 lr4 lr4 lr13;
    ADD                 lr5 lr5 lr12;;            # lr4 -> k=1's offset; lr5 -> 0

    # Peeled first iteration (k=0): ACC.FIRST seeds the accumulator (replaces RESET_ACC).
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG lr4 cr11 lr15;
    BNE                 lr5 lr6 k_loop_pre;;
    B                   after_k_loop;;

k_loop_pre:
    ADD                 lr4 lr4 lr13;
    ADD                 lr5 lr5 lr12;;            # lr4 -> next load offset; lr5 -> k just loaded

k_loop:
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr11 lr15;
    BNE                 lr5 lr6 k_loop_pre;;

after_k_loop:
    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 lr14;
    ADD                 lr0 lr0 lr13;;

    BREAK;;

    BLT                 lr0 lr1 row_loop;;

end:
    BKPT;;
