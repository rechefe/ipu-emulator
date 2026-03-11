# Matrix multiplication: C = A x B
# A: 128x128 input  (M=128 rows, K=128 cols)
# B: 128x128 weights (K=128 rows, N=128 cols)
# C: 128x128 output (M=128 rows, N=128 cols, stored as 128 int32/fp32 accumulators per row)
#
# Algorithm: for each output row m, load A[m][0..127] into r_cyclic, then
# for each k=0..127 load B[k][0..127] into mem_bypass and accumulate
# A[m][k] * B[k][n] for all n simultaneously (SIMD over N outputs).
#
# Memory layout (set via CR registers):
#   cr0 = input  base  (A: 128 rows x 128 bytes = 16384 bytes)
#   cr1 = weights base (B: 128 rows x 128 bytes = 16384 bytes, row k = weights for input k)
#   cr2 = output base  (C: 128 rows x 512 bytes = 65536 bytes, 128 x int32/fp32 per row)
#
# Inner loop uses a one-cycle startup offset (lr4=-128, lr5=-1):
#   cycle 1: loads from cr1-128 (zeros, harmless) and reads cyclic[-1] (don't care)
#   cycles 2..129: correctly computes k=0..127

    # -- Initialization -------------------------------------------------------
    set                 lr0 0 ;;       # lr0: byte offset into A (row m * 128), starts at row 0
    set                 lr1 16384 ;;   # lr1: loop limit = M * K = 128 * 128 bytes
    set                 lr7 0 ;;       # lr7: byte offset into C (row m * 512), starts at row 0

row_loop:
    reset_acc;;                        # clear accumulator for new output row

    ldr_cyclic_mult_reg lr0 cr0 lr15;; # load A[m][0..127] into r_cyclic (input row m)

    # inner loop init: startup offset so load and mult index align correctly
    set                 lr4 -128;;     # lr4: weight byte offset (cr1 + lr4 = B[k], starts one step back)
    set                 lr5 -1;;       # lr5: k counter (-1 so first real k=0 after incr)
    set                 lr6 127;;      # lr6: k loop limit (K-1 = 127)

k_loop:
    # One VLIW cycle: load B[k], advance pointers, multiply A[m][k] * B[k][0..127], accumulate
    ldr_mult_reg        mem_bypass lr4 cr1;   # load B[k][0..127] from address cr1+lr4
    incr                lr4 128;              # advance weight pointer to next row of B
    incr                lr5 1;                # advance k counter
    mult.ev             mem_bypass lr5 lr15 lr15; # A[m][k] (scalar from r_cyclic[k]) * B[k][0..127]
    acc;                                      # accumulate: C[m][n] += A[m][k] * B[k][n] for all n
    blt                 lr5 lr6 k_loop;;      # repeat for k=0..127

    str_acc_reg         lr7 cr2;;      # store 512-byte accumulator as output row m of C
    incr                lr7 512;       # advance output pointer to next row (512 bytes = 128 * 4)
    incr                lr0 128;;      # advance input pointer to next row of A (128 bytes)

    break;;                            # debug breakpoint (no-op in normal execution)

    blt                 lr0 lr1 row_loop;;  # repeat for m=0..127

end:
    bkpt;;                             # halt execution
