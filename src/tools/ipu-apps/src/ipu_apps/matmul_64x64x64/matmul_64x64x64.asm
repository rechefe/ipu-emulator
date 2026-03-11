# Matrix multiplication: C = A x W^T
# A: 64x64  input  (M=64 rows, K=64 cols)
# W: 64x64  weights output-major (N=64 rows, K=64 cols); transposed to T in XMEM
# C: 64x64  output (M=64 rows, N=64 cols, stored as 128 int32/fp32 per row)
#
# K=64 < SIMD width (128): A rows zero-padded to 128 bytes in XMEM.
# N=64 < SIMD width (128): T[k] zero-padded to 128 bytes; acc[64..127] = 0.
#
# Output packing: str_acc_reg always stores 512 bytes, but only first 256 (N*4) are
# valid. Use incr lr7 256 (not 512) to pack outputs contiguously (FC convention).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (A: 64 rows x 128 bytes padded = 8192 bytes)
#   cr1 = weights base (T: 64 rows x 128 bytes padded = 8192 bytes)
#   cr2 = output base  (C: 64 rows x 256 bytes packed = 16384 bytes)
#
# Differs from matmul_128x64x64.asm only in: lr1 = 8192 (M=64 rows × 128B).

    # -- Initialization -------------------------------------------------------
    set                 lr0 0 ;;       # lr0: byte offset into A (row m * 128, padded stride)
    set                 lr1 8192 ;;    # lr1: outer loop limit = M * 128 = 64 * 128
    set                 lr7 0 ;;       # lr7: byte offset into C (row m * 256, packed)

row_loop:
    reset_acc;;                        # clear accumulator for new output row

    ldr_cyclic_mult_reg lr0 cr0 lr15;; # load A[m][0..63] + 64 zeros into r_cyclic (padded row m)

    # inner loop init: startup offset so load and mult index align correctly
    set                 lr4 -128;;     # lr4: weight byte offset (starts one step back)
    set                 lr5 -1;;       # lr5: k counter (-1 so first real k=0 after incr)
    set                 lr6 63;;       # lr6: k loop limit (K-1 = 63)

k_loop:
    # One VLIW cycle: load T[k], advance pointers, multiply A[m][k] * T[k][0..127], accumulate
    ldr_mult_reg        mem_bypass lr4 cr1;   # load T[k][0..127] from address cr1+lr4
    incr                lr4 128;              # advance weight pointer to next row of T
    incr                lr5 1;                # advance k counter
    mult.ev             mem_bypass lr5 lr15 lr15; # A[m][k] (r_cyclic[k]) * T[k][0..127]
    acc;                                      # accumulate: C[m][n] += A[m][k] * T[k][n]
    blt                 lr5 lr6 k_loop;;      # repeat for k=0..63

    str_acc_reg         lr7 cr2;;      # store 512-byte accumulator (first 256B = C[m][0..63])
    incr                lr7 256;       # advance by 256 bytes (N*4) — pack outputs contiguously
    incr                lr0 128;;      # advance input pointer to next padded row of A

    break;;                            # debug breakpoint (no-op in normal execution)

    blt                 lr0 lr1 row_loop;;  # repeat for m=0..63

end:
    bkpt;;                             # halt execution
