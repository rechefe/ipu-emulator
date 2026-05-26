# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]
#
# D: interleaved channel-major [K=144 channels, 2 tg, 128 tokens]
#    Row (k, tg) at DATA_BASE + k*256 + tg*128
# W: output-major [144 out_ch, 144 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*256
#    W[j, 128..143] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
# C: grouped channel-major [2 tg, 144 out_ch, 128 tokens], FP32 accumulators
#    Row (j, tg) at OUTPUT_BASE + tg*N_OUT*512 + j*512
#
# Algorithm: load W[j,:] into r0/r1 (once per j); load D[k,tg] into r_cyclic per k.
#   MULT.VE.CYCLIC: r0[fixed_idx] x r_cyclic[:] → 128 outputs per cycle.
#   k=0..127 uses r0[k] (fixed_idx=k); k=128..143 uses r1[k-128].
#
# CRs: cr0=DATA_BASE, cr1=WEIGHTS_BASE, cr2=WEIGHTS_BASE+128,
#       cr3=OUTPUT_BASE (tg=0), cr4=OUTPUT_BASE+N_OUT*512 (tg=1)
#       cr5=-256 (tg=0 data startup), cr6=-128 (tg=1 data startup)
#       cr7=-1 (k-loop1 fixed_idx startup), cr8=127 (k-loop2 fixed_idx startup)
# LRs (preset by harness):
#   lr0=0    (const: r_cyclic write-index 0)
#   lr2=256  (data stride: 256 bytes/channel)
#   lr3=512  (output stride: 512 bytes/j)
#   lr6=127  (k-loop1 bound: loop while lr5 < 127, last real k=127)
#   lr7=0    (output pointer, incremented by lr3 each j)
#   lr8=0    (weight byte offset, incremented by lr12 each j)
#   lr9=0    (j counter)
#   lr10=144 (j-loop limit)
#   lr11=143 (k-loop2 bound: loop while lr5 < 143, last real k=143)
#   lr12=256 (W_STRIDE = 256 bytes per j)
#
# One-cycle startup offset pattern (pipeline alignment):
#   k-loop startup: lr4 = first_real_addr - stride, lr5 = -1
#   First cycle: XMEM loads from (cr0 + lr4_start) which may be negative/invalid
#                but r_cyclic[lr5_start = -1 mod 512 = 511] is unused slot → harmless
#   After LR increments: lr4 = first_real_addr, lr5 = 0 → k=0 mult uses correct data
#   tg=0: lr4 start=-256 → first real load at 0 (D[k=0,tg=0])
#   tg=1: lr4 start=-128 → first real load at 128 (D[k=0,tg=1])
#   k-loop2 startup: lr4 continues from k-loop1 end (naturally at k=128 addr)
#                    lr5 reset to 127 → first live fixed_idx=128 (reads r1[0])

j_loop:
    LDR_MULT_REG r0 lr8 cr1;;          # r0[0..127] = W[j, 0..127]
    LDR_MULT_REG r1 lr8 cr2;;          # r1[0..127] = W[j, 128..143] + zeros

    # -- token group 0 -------------------------------------------------------
    RESET_ACC;;
    SET lr4 cr5;;                       # tg=0 startup offset: -256
    SET lr5 cr7;;                       # k-loop1 fixed_idx startup: -1

k_loop1_tg0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_loop1_tg0;;

    SET lr5 cr8;;                       # k-loop2 fixed_idx startup: 127 → first live=128 (r1[0])

k_loop2_tg0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr11 k_loop2_tg0;;

    STR_ACC_REG lr7 cr3;;               # store 512B → OUTPUT[j, tg=0]

    # -- token group 1 -------------------------------------------------------
    RESET_ACC;;
    SET lr4 cr6;;                       # tg=1 startup offset: -128
    SET lr5 cr7;;                       # k-loop1 fixed_idx startup: -1

k_loop1_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_loop1_tg1;;

    SET lr5 cr8;;

k_loop2_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr11 k_loop2_tg1;;

    STR_ACC_REG lr7 cr4;;               # store 512B → OUTPUT[j, tg=1]
    ADD lr7 lr7 lr3;;                   # advance output ptr

    ADD lr8 lr8 lr12; ADD lr9 lr9 1;;   # next j: weight offset += W_STRIDE, j++
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
