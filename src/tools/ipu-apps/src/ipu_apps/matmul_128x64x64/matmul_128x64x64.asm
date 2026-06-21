# Matrix multiplication: C = A x W^T
# A: 128x64  input  (M=128 rows, K=64 cols)
# W: 64x64   weights output-major (N=64 rows, K=64 cols); transposed to T in XMEM
# C: 128x64  output (M=128 rows, N=64 cols, stored as 128 int32/fp32 per row)
#
# K=64: A rows zero-padded to 128 bytes. N=64: T[k] zero-padded to 128 bytes.
# Output stride 256 (N*4) packs only the valid first-half per row.
#
# New kernel: MULT.VE.CYCLIC replaces MULT.EV mem_bypass (mem_bypass removed in PR #69).
#
# Memory layout (set via CR registers):
#   cr0  = input  base (A: 128 rows x 128 bytes padded = 16384 bytes)
#   cr11 = weights base (T: 64 rows x 128 bytes padded = 8192 bytes)
#          (moved off CR1 — CR1 is now a read-only hardwired constant ≡ 1)
#   cr2 = output base  (C: 128 rows x 256 bytes packed = 32768 bytes)
#   cr3 = 1      (ADD step for fixed_idx)
#   cr4 = 128    (ADD step for weight/input strides)
#   cr5 = 256    (ADD step for output stride = N*4)
#   cr6 = 16384  (outer loop limit = M * 128)
#   cr7 = 0      (const zero)
#   cr8 = -128   (inner loop init: weight offset startup)
#   cr9 = -1     (inner loop init: fixed_idx startup)
#   cr10 = 63    (inner loop limit K-1 = 63)
#
# Differs from matmul_128x64x128.asm only in: cr5 = 256 (output stride = N*4).

    SET                 lr12 cr3;;
    SET                 lr13 cr4;;
    SET                 lr14 cr5;;       # output stride = N*4 = 256
    SET                 lr0 cr7;;
    SET                 lr1 cr6;;
    SET                 lr7 cr7;;

row_loop:
    RESET_ACC;;

    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr4 cr8;;
    SET                 lr5 cr9;;
    SET                 lr6 cr10;;

k_loop:
    LDR_CYCLIC_MULT_REG lr4 cr11 lr15;
    ADD                 lr4 lr4 lr13;
    ADD                 lr5 lr5 lr12;
    MULT.VE.CYCLIC      lr15 0 lr15 lr5;
    ACC;
    BLT                 lr5 lr6 k_loop;;

    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 lr14;    # +256: pack N=64 outputs contiguously
    ADD                 lr0 lr0 lr13;;

    BREAK;;

    BLT                 lr0 lr1 row_loop;;

end:
    BKPT;;
