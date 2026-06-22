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
#   cr0  = input  base (A: 128 rows x 128 bytes padded = 16384 bytes)
#   cr11 = weights base (T: 64 rows x 128 bytes = 8192 bytes; T[k] = col k of W)
#          (moved off CR1 — CR1 is now a read-only hardwired constant ≡ 1)
#   cr2 = output base  (C: 128 rows x 512 bytes = 65536 bytes)
#   cr3 = 1      (ADD step for fixed_idx)
#   cr4 = 128    (ADD step for weight/input strides)
#   cr5 = 512    (ADD step for output stride)
#   cr6 = 16384  (outer loop limit = M * 128)
#   cr7 = 0      (const zero)
#   cr8 = -128   (inner loop init: weight offset startup)
#   cr9 = -1     (inner loop init: fixed_idx startup)
#   cr10 = 63    (inner loop limit K-1 = 63)
#
# Identical to matmul_128x128.asm except cr10 = 63 (K-1 = 63).

    SET                 lr12 cr3;;
    SET                 lr13 cr4;;
    SET                 lr14 cr5;;
    SET                 lr0 cr7;;
    SET                 lr1 cr6;;
    SET                 lr7 cr7;;

row_loop:
    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr4 cr8;;
    SET                 lr5 cr9;;
    SET                 lr6 cr10;;       # K-1 = 63

    # Peeled first iteration (k=0): ACC.FIRST seeds the accumulator (replaces RESET_ACC).
    LDR_CYCLIC_MULT_REG lr4 cr11 lr15;
    ADD                 lr4 lr4 lr13;
    ADD                 lr5 lr5 lr12;
    MULT.RC.VE          lr15 lr5 0 lr15;
    ACC.FIRST;
    BNE                 lr5 lr6 k_loop;;
    B                   after_k_loop;;

k_loop:
    LDR_CYCLIC_MULT_REG lr4 cr11 lr15;
    ADD                 lr4 lr4 lr13;
    ADD                 lr5 lr5 lr12;
    MULT.RC.VE          lr15 lr5 0 lr15;
    ACC;
    BNE                 lr5 lr6 k_loop;;

after_k_loop:
    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 lr14;
    ADD                 lr0 lr0 lr13;;

    BREAK;;

    BLT                 lr0 lr1 row_loop;;

end:
    BKPT;;
