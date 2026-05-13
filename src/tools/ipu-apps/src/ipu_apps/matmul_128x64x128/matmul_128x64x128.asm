# Matrix multiplication: C = A x B
# A: 128x64  input  (M=128 rows, K=64 cols)
# B: 64x128  weights (K=64 rows, N=128 cols)
# C: 128x128 output (M=128 rows, N=128 cols, stored as 128 int32/fp32 accumulators per row)
#
# K=64 < SIMD width (128): each row of A is zero-padded to 128 bytes in XMEM.
#
# New kernel: MULT.VE.CYCLIC replaces MULT.EV mem_bypass (mem_bypass removed in PR #69).
#   New shape: A[m] in r0 (loaded once per row), T[k] rows in r_cyclic per inner step.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (A: 128 rows x 128 bytes padded = 16384 bytes)
#   cr1 = weights base (T: 64 rows x 128 bytes = 8192 bytes; T[k] = col k of W)
#   cr2 = output base  (C: 128 rows x 512 bytes = 65536 bytes)
#
# Identical to matmul_128x128.asm except lr6 = 63 (K-1 = 63).

    SET                 lr12 1;;
    SET                 lr13 128;;
    SET                 lr14 512;;
    SET                 lr0 0;;
    SET                 lr1 16384;;
    SET                 lr7 0;;

row_loop:
    RESET_ACC;;

    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr4 -128;;
    SET                 lr5 -1;;
    SET                 lr6 63;;         # K-1 = 63

k_loop:
    LDR_CYCLIC_MULT_REG lr4 cr1 lr15;
    ADD                 lr4 lr4 lr13;
    ADD                 lr5 lr5 lr12;
    MULT.VE.CYCLIC      lr15 0 lr15 lr5;
    ACC;
    BLT                 lr5 lr6 k_loop;;

    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 lr14;
    ADD                 lr0 lr0 lr13;;

    BREAK;;

    BLT                 lr0 lr1 row_loop;;

end:
    BKPT;;
