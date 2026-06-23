# Agent D — kQᵀ → key-major attention scores, one head (D=36, N=256)
#
# S[i, s] = sum_c Q[i, c] * K[s, c]      i,s in [0,256), c in [0,36)
#   Lanes = queries (i); outer loop = key (s); contraction = head channel (c).
#
# Memory layout (one head; harness offsets by head h before load):
#   Q: channel-major     Q[i, c] at QBASE + c*256 + i
#      A channel column (128 queries, one channel c, group g) is contiguous:
#      QBASE + c*256 + g*128  → loads straight into R_CYCLIC (vector operand).
#   K: key-major scratch K[s, 0:35] at KBASE_KM + s*128 (36 ch padded to 128 B)
#      Loaded once per key s into R0 (scalar operand, indexed by c).
#   S: key-major words   S[i, s] at SBASE + (s*256 + i)*4
#      (s,g=0) → SBASE + s*1024;  (s,g=1) → SBASE + s*1024 + 512
#      One running 512-B-stride pointer covers g=0, g=1, next key contiguously.
#
# Compute (matmul template: scalar from R0 indexed by c × vector in R_CYCLIC):
#   MULT.RC.VE rc_idx=0, src=lr5 (=c → R0[c]) → mult_res[i] = Q[i,c]·K[s,c]
#   c==0: ACC.FIRST  else: ACC   → R_ACC[i] += Q[i,c]·K[s,c]
#   After 36 channels: R_ACC[i] = S[i,s] for the 128 queries of this group.
#   STR_ACC_REG → 512 B (128 × int32/fp32) key-major score row.  No AGG.
#
# Counts/head: 256 keys × 2 groups × 36 channels = 18432 MULT+ACC bundles
#              + 512 stores.
#
# CRs (harness-set):
#   cr0 = QBASE        (data base for LDR_CYCLIC_MULT_REG)
#   cr2 = SBASE        (output base for STR_ACC_REG)
#   cr9 = KBASE_KM     (key-major K base for LDR_MULT_REG into R0)
#   cr5 = -256         (g=0 channel-column startup: first live = 0 after +256)
#   cr6 = -128         (g=1 channel-column startup: first live = 128 after +256)
#   cr7 = -1           (fixed_idx c startup: first live = 0 after +1)
#   cr8 = 34           (c-loop bound: first=0, width=36 → 0+36-2 = 34)
#   cr1 = 1            (read-only hardwired constant ≡ 1)
# LRs (harness-set persistent):
#   lr0  = 0    (R_CYCLIC write/read index 0)
#   lr2  = 256  (channel stride: 256 B between consecutive channels in Q)
#   lr3  = 512  (output store stride: 512 B per (s,g) row)
#   lr6  = 34   (c-loop bound)
#   lr7  = 0    (output byte pointer, += 512 per store)
#   lr8  = -128 (key byte offset into K; += 128 per key, first live = 0)
#   lr9  = 0    (key counter s)
#   lr10 = 256  (key-loop limit = N)
#   lr12 = 128  (key stride into K scratch)

s_loop:
    ADD lr8 lr8 lr12;;                  # key byte offset += 128 (first live = 0)
    LDR_MULT_REG r0 lr8 cr9;;           # r0[0..127] = K[s, 0:35] + zeros

    # -- query group 0 (queries 0..127) -------------------------------------
    SET lr4 cr5;;                       # channel-column startup: -256
    SET lr5 cr7;;                       # fixed_idx c startup: -1

    # Peeled first channel (c=0): ACC.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 c_loop_g0;;
    B after_c_g0;;

c_loop_g0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 c_loop_g0;;

after_c_g0:
    STR_ACC_REG lr7 cr2;;               # store S[0:128, s] (key-major row, g=0)
    ADD lr7 lr7 lr3;;                   # output ptr += 512

    # -- query group 1 (queries 128..255) -----------------------------------
    SET lr4 cr6;;                       # g=1 channel-column startup: -128 → first live = 128
    SET lr5 cr7;;                       # fixed_idx c startup: -1

    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 c_loop_g1;;
    B after_c_g1;;

c_loop_g1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 c_loop_g1;;

after_c_g1:
    STR_ACC_REG lr7 cr2;;               # store S[128:256, s] (key-major row, g=1)
    ADD lr7 lr7 lr3;;                   # output ptr += 512

    ADD lr9 lr9 cr1; BLT lr9 lr10 s_loop;;   # next key

end:
    BKPT;;
