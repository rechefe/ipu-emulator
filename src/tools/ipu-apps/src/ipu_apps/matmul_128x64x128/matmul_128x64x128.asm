# Matrix multiplication: C = A x B
# A: 128x64  input  (M=128 rows, K=64 cols)
# B: 64x128  weights (K=64 rows, N=128 cols)
# C: 128x128 output (M=128 rows, N=128 cols, stored as 128 int32/fp32 accumulators per row)
#
# K=64 < SIMD width (128): each row of A is zero-padded to 128 bytes in XMEM so
# ldr_cyclic_mult_reg (which always loads 128 bytes) works unchanged.
# The inner loop only accesses cyclic indices 0..63; the padded zeros are never read.
#
# No mask needed: N=128 = full SIMD width. lr15=0 (zero-init) selects mask slot 0
# with shift 0. r_mask is zero-initialized, so no accumulator outputs are zeroed.
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (A: 128 rows x 128 bytes padded = 16384 bytes)
#   cr1 = weights base (B: 64 rows x 128 bytes = 8192 bytes, row k = weights for input k)
#   cr2 = output base  (C: 128 rows x 512 bytes = 65536 bytes, 128 x int32/fp32 per row)
#
# This assembly is identical to matmul_128x128.asm except lr6 = 63 (K-1 = 63).

    # -- Initialization -------------------------------------------------------
    set                 lr0 0 ;;       # lr0: byte offset into A (row m * 128, padded stride)
    set                 lr1 16384 ;;   # lr1: outer loop limit = M * 128 = 128 * 128 (padded)
    set                 lr7 0 ;;       # lr7: byte offset into C (row m * 512)

row_loop:
    reset_acc;;                        # clear accumulator for new output row

    ldr_cyclic_mult_reg lr0 cr0 lr15;; # load A[m][0..63] + 64 zeros into r_cyclic (padded row m)

    # inner loop init: startup offset so load and mult index align correctly
    set                 lr4 -128;;     # lr4: weight byte offset (starts one step back)
    set                 lr5 -1;;       # lr5: k counter (-1 so first real k=0 after incr)
    set                 lr6 63;;       # lr6: k loop limit (K-1 = 63)  ← only change vs 128x128

k_loop:
    # One VLIW cycle: load B[k], advance pointers, multiply A[m][k] * B[k][0..127], accumulate
    ldr_mult_reg        mem_bypass lr4 cr1;   # load B[k][0..127] from address cr1+lr4
    incr                lr4 128;              # advance weight pointer to next row of B
    incr                lr5 1;                # advance k counter
    mult.ev             mem_bypass lr5 lr15 lr15; # A[m][k] (r_cyclic[k]) * B[k][0..127]
    acc;                                      # accumulate: C[m][n] += A[m][k] * B[k][n] for all n
    blt                 lr5 lr6 k_loop;;      # repeat for k=0..63

    str_acc_reg         lr7 cr2;;      # store 512-byte accumulator as output row m of C
    incr                lr7 512;       # advance output pointer (512 bytes = 128 * 4)
    incr                lr0 128;;      # advance input pointer to next padded row of A

    break;;                            # debug breakpoint (no-op in normal execution)

    blt                 lr0 lr1 row_loop;;  # repeat for m=0..127

end:
    bkpt;;                             # halt execution
