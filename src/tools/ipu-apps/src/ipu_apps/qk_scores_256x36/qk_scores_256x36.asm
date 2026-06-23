# QKᵀ scores (Agent C), one attention head:
#   S[i, s] = sum_c Q[i, c] * K[s, c]   contraction over head_dim c = 0..35
#
# One head: N = 256 tokens (queries = keys), head_dim D = 36.
#   Q, K logically channel-major. K is loaded channel-major as given:
#     K[s, c] at K_BASE + c*256 + s  (a full channel column = 256 contiguous
#     bytes, loaded as two 128-key chunks: g=0 → +0, g=1 → +128).
#   Q is STAGED query-major by the harness (gather of the strided channels):
#     QROW[i] = Q[i, 0..35] contiguous at QROW_BASE + i*QROW_STRIDE.
#     This lets r0 hold one query's 36 channels with a single LDR_MULT_REG,
#     exactly the matmul broadcast template (scalar from r0 indexed by c).
#
# Broadcast template (mirrors matmul_144x144_x128 k-loop1, width 36):
#   per query i:  r0 = QROW[i]              (36 scalars Q[i, 0..35])
#   per channel:  r_cyclic = K[g*128:+128, c]   (128 keys' channel-c column)
#                 MULT.RC.VE: scalar Q[i,c] (= r0[c]) × vector r_cyclic
#                   → mult_res[s] = Q[i,c] * K[s,c]
#                 ACC.FIRST (c=0) / ACC      → R_ACC[s] += Q[i,c] * K[s,c]
#   after 36 channels R_ACC[s] = S[i, s] for the 128 keys of group g.
#   STR_ACC_REG → S[i, g] (raw R_ACC, 512 B / 128 FP32|INT32 lanes, query-major)
#
# No AGG. Scores are stored RAW (full precision) so softmax (Agent A) reads
# unquantized scores; this matches every other matmul kernel (STR_ACC_REG).
#
# CRs (set by harness):
#   cr0 = K_BASE                       (data base)
#   cr9 = QROW_BASE                    (staged query-major Q; off read-only cr1)
#   cr3 = S_BASE                       (group 0 output base)
#   cr4 = S_BASE + 512                 (group 1 output base)
#   cr5 = -256                         (g=0 K-data startup: first live = 0)
#   cr6 = -128                         (g=1 K-data startup: first live = 128)
#   cr7 = -1                           (channel fixed_idx startup: first live = 0)
#   cr8 = 34                           (contraction bound: first=0,width=36 → 34)
#   cr1 = 1                            (hardwired read-only constant)
# LRs (set by harness):
#   lr0  = 0      (r_cyclic write-index 0 and mask_shift 0)
#   lr2  = 256    (K data stride: 256 bytes/channel)
#   lr3  = 1024   (output stride: 1024 bytes/query = 2 groups × 512)
#   lr6  = 34     (contraction BLT bound)
#   lr7  = 0      (output query byte offset)
#   lr8  = 0      (Q-row byte offset, += QROW_STRIDE each query)
#   lr9  = 0      (query counter)
#   lr10 = 256    (query-loop limit = N)
#   lr12 = 512    (QROW_STRIDE = 512 bytes per query)

q_loop:
    LDR_MULT_REG r0 lr8 cr9;;            # r0 = QROW[i] = Q[i, 0..35] (rest pad)

    # -- key group 0 ---------------------------------------------------------
    SET lr4 cr5;;                        # g=0 K-data startup: -256
    SET lr5 cr7;;                        # channel fixed_idx startup: -1

    # Peeled first channel (c=0): ACC.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 c_loop_g0;;
    B after_c_g0;;

c_loop_g0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 c_loop_g0;;

after_c_g0:
    STR_ACC_REG lr7 cr3;;                # store 512B → S[i, keys 0..127]

    # -- key group 1 ---------------------------------------------------------
    SET lr4 cr6;;                        # g=1 K-data startup: -128
    SET lr5 cr7;;                        # channel fixed_idx startup: -1

    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 c_loop_g1;;
    B after_c_g1;;

c_loop_g1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 c_loop_g1;;

after_c_g1:
    STR_ACC_REG lr7 cr4;;                # store 512B → S[i, keys 128..255]
    ADD lr7 lr7 lr3;;                    # advance output ptr (+1024)

    ADD lr8 lr8 lr12; ADD lr9 lr9 cr1;;  # next query: Q ptr += 512, i++
    BLT lr9 lr10 q_loop;;

end:
    BKPT;;
