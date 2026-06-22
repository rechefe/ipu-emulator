# Matrix multiplication: C = A x B
# A: 128x128 input  (M=128 rows, K=128 cols)
# B: 128x128 weights (K=128 rows, N=128 cols)
# C: 128x128 output (M=128 rows, N=128 cols, stored as 128 int32/fp32 accumulators per row)
#
# New kernel: MULT.VE.CYCLIC replaces MULT.EV mem_bypass (mem_bypass removed in PR #69).
#   Old shape: T[k] in mem_bypass (vector), A[m][k] scalar from r_cyclic via cyclic index.
#   New shape: A[m] in r0 (loaded once per output row via LDR_MULT_REG r0),
#              T[k] rows loaded into r_cyclic (LDR_CYCLIC_MULT_REG) per inner-loop step.
#              MULT.VE.CYCLIC selects scalar A[m][k] from r0 via fixed_idx=k.
#
# Memory layout (set via CR registers):
#   cr0  = input  base (A: 128 rows x 128 bytes = 16384 bytes)
#   cr11 = weights base (T: 128 rows x 128 bytes = 16384 bytes; T[k] = col k of W)
#          (moved off CR1 — CR1 is now a read-only hardwired constant ≡ 1)
#   cr2 = output base  (C: 128 rows x 512 bytes = 65536 bytes, 128 x int32/fp32 per row)
#   cr3 = 1      (ADD step for fixed_idx)
#   cr4 = 128    (ADD step for weight/input strides)
#   cr5 = 512    (ADD step for output stride)
#   cr6 = 16384  (outer loop limit = M * 128)
#   cr7 = 0      (const zero: initial offsets and r_cyclic slot 0)
#   cr8 = -128   (inner loop init: weight offset startup)
#   cr9 = -1     (inner loop init: fixed_idx startup)
#   cr10 = 127   (inner loop limit K-1)
#
# LRs:
#   lr0  = byte offset into A (row m * 128)
#   lr1  = outer loop limit = M * 128 = 16384
#   lr4  = weight byte offset; init=-128 (startup: first live = 0 = T[k=0] after ADD)
#   lr5  = fixed_idx k; init=-1 (startup: first live = 0 after ADD)
#   lr6  = k loop limit K-1 = 127; BLT exits when snap >= 127
#   lr7  = byte offset into C (row m * 512)
#   lr12 = 1   (ADD step for fixed_idx)
#   lr13 = 128 (ADD step for weight/input strides)
#   lr14 = 512 (ADD step for output stride)
#   lr15 = 0   (const: r_cyclic slot 0 for LDR_CYCLIC_MULT_REG and MULT.VE.CYCLIC)

    SET                 lr12 cr3;;
    SET                 lr13 cr4;;
    SET                 lr14 cr5;;
    SET                 lr0 cr7;;
    SET                 lr1 cr6;;
    SET                 lr7 cr7;;

row_loop:
    LDR_MULT_REG        r0 lr0 cr0;;     # r0 = A[m][0..127]

    SET                 lr4 cr8;;        # first live = 0 = T[k=0]
    SET                 lr5 cr9;;        # first live = 0
    SET                 lr6 cr10;;       # BNE: exit when live k == 127; last live = 127

    # Peeled first iteration (k=0): ACC.FIRST seeds the accumulator (replaces RESET_ACC).
    LDR_CYCLIC_MULT_REG lr4 cr11 lr15;  # XMEM: r_cyclic[0] = T[k][0..127]
    ADD                 lr4 lr4 lr13;    # LR0 : weight offset += 128
    ADD                 lr5 lr5 lr12;    # LR1 : fixed_idx += 1
    MULT.RC.VE          lr15 lr5 0 lr15; # MULT: A[m][live k] × T[k][0..127]
    ACC.FIRST;                           # ACC : seed accumulator (= reset + accumulate)
    BNE                 lr5 lr6 k_loop;; # if K>1 run remaining k; else fall through
    B                   after_k_loop;;

k_loop:
    LDR_CYCLIC_MULT_REG lr4 cr11 lr15;  # XMEM: r_cyclic[0] = T[k][0..127]
    ADD                 lr4 lr4 lr13;    # LR0 : weight offset += 128
    ADD                 lr5 lr5 lr12;    # LR1 : fixed_idx += 1
    MULT.RC.VE          lr15 lr5 0 lr15; # MULT: A[m][live k] × T[k][0..127]
    ACC;                                 # ACC : accumulate
    BNE                 lr5 lr6 k_loop;; # loop while live k != 127

after_k_loop:
    STR_ACC_REG         lr7 cr2;;        # store 512 B → C[m]
    ADD                 lr7 lr7 lr14;    # out ptr += 512
    ADD                 lr0 lr0 lr13;;   # input ptr += 128

    BREAK;;

    BLT                 lr0 lr1 row_loop;;

end:
    BKPT;;
