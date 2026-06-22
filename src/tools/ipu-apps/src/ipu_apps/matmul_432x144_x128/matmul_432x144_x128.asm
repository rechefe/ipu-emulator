# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]
#
# D: interleaved channel-major [K=144 channels, 2 tg, 128 tokens]
#    Row (k, tg) at DATA_BASE + k*256 + tg*128
# W: output-major [432 out_ch, 144 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*256
#    W[j, 128..143] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
# C: grouped channel-major [2 tg, 432 out_ch, 128 tokens], FP32 accumulators
#    Row (j, tg) at OUTPUT_BASE + tg*N_OUT*512 + j*512
#
# Algorithm: load W[j,:] into r0/r1 (once per j); load D[k,tg] into r_cyclic per k.
#   MULT.VE.CYCLIC: r0[fixed_idx] x r_cyclic[:] → 128 outputs per cycle.
#   k=0..127 uses r0[k]; k=128..143 uses r1[k-128] (fixed_idx=128..143 auto-reads r1).
#
# CRs: cr0=DATA_BASE, cr9=WEIGHTS_BASE (moved off read-only CR1), cr2=WEIGHTS_BASE+128,
#       cr3=OUTPUT_BASE (tg=0), cr4=OUTPUT_BASE+N_OUT*512 (tg=1)
#       cr5=-256 (tg=0 data startup), cr6=-128 (tg=1 data startup)
#       cr7=-1 (k-loop1 fixed_idx startup), cr8=127 (k-loop2 fixed_idx startup)
# LRs (preset by harness):
#   lr0=0    (const: r_cyclic write-index 0)
#   lr2=256  (data stride: 256 bytes/channel)
#   lr3=512  (output stride: 512 bytes/j)
#   lr6=126  (k-loop1 bound: first_index=0, width=128 → 0+128-2=126)
#   lr7=0    (output pointer, incremented by lr3 each j)
#   lr8=0    (weight byte offset, incremented by lr12 each j)
#   lr9=0    (j counter)
#   lr10=432 (j-loop limit)
#   lr11=142 (k-loop2 bound: first_index=128, width=16 → 128+16-2=142)
#   lr12=256 (W_STRIDE = 256 bytes per j)
#
# Memory layout:
#   DATA:    2 × 144 × 128 B =  36 864 B (0x00000..0x08FFF)
#   WEIGHTS: 432 rows × 256 B = 110 592 B (0x10000..0x2AFFF)
#   OUTPUT:  2 × 432 × 512 B = 442 368 B  (0x30000..0x9BFFF)

j_loop:
    LDR_MULT_REG r0 lr8 cr9;;          # r0[0..127] = W[j, 0..127]
    LDR_MULT_REG r1 lr8 cr2;;          # r1[0..127] = W[j, 128..143] + zeros

    # -- token group 0 -------------------------------------------------------
    SET lr4 cr5;;                       # tg=0 startup offset: -256
    SET lr5 cr7;;                       # k-loop1 fixed_idx startup: -1

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 k_loop1_tg0;;
    B after_k_tg0;;

k_loop1_tg0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_loop1_tg0;;

after_k_tg0:
    SET lr5 cr8;;                       # k-loop2 fixed_idx startup: 127 → first live=128 (r1[0])

k_loop2_tg0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr11 k_loop2_tg0;;

    STR_ACC_REG lr7 cr3;;               # store 512B → OUTPUT[j, tg=0]

    # -- token group 1 -------------------------------------------------------
    SET lr4 cr6;;                       # tg=1 startup offset: -128
    SET lr5 cr7;;                       # k-loop1 fixed_idx startup: -1

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 k_loop1_tg1;;
    B after_k_tg1;;

k_loop1_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_loop1_tg1;;

after_k_tg1:
    SET lr5 cr8;;

k_loop2_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr11 k_loop2_tg1;;

    STR_ACC_REG lr7 cr4;;               # store 512B → OUTPUT[j, tg=1]
    ADD lr7 lr7 lr3;;                   # advance output ptr

    ADD lr8 lr8 lr12; ADD lr9 lr9 cr1;;   # next j: weight offset += W_STRIDE, j++
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
