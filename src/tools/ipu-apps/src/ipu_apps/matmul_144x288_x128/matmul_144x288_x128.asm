# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]
#
# D: interleaved channel-major [K=288 channels, 2 tg, 128 tokens]
#    Row (k, tg) at DATA_BASE + k*256 + tg*128
# W: output-major [144 out_ch, 288 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*384
#    W[j, 128..255] at WEIGHTS_BASE + j*384 + 128
#    W[j, 256..287] at WEIGHTS_BASE + j*384 + 256 (padded to 128 bytes)
# C: grouped channel-major [2 tg, 144 out_ch, 128 tokens], FP32 accumulators
#    Row (j, tg) at OUTPUT_BASE + tg*N_OUT*512 + j*512
#
# Algorithm: K=288 split into three 128-element chunks per tg.
#   Per chunk: load W[j, chunk*128..(chunk+1)*128-1] into r0, run 128 k-steps.
#   MULT.VE.CYCLIC r0[lr5] x r_cyclic[:] — lr5 cycles 0..127 per chunk.
#   lr4 (data pointer) is NOT reset between chunks; advances continuously.
#   lr5 reset to -1 at the start of each chunk (combined with LDR_MULT_REG).
#
# CRs: cr0=DATA_BASE, cr1=WEIGHTS_BASE, cr2=WEIGHTS_BASE+128, cr3=WEIGHTS_BASE+256,
#       cr4=OUTPUT_BASE (tg=0), cr5=OUTPUT_BASE+N_OUT*512 (tg=1)
#       cr6=-256 (tg=0 data startup), cr7=-128 (tg=1 data startup)
#       cr8=-1   (per-chunk fixed_idx startup, reset before every chunk)
# LRs (preset by harness):
#   lr0=0    (const: r_cyclic write-index 0)
#   lr2=256  (data stride: 256 bytes/channel)
#   lr3=512  (output stride: 512 bytes/j)
#   lr6=127  (per-chunk k-loop bound: loop while lr5 < 127, last real k=127)
#   lr7=0    (output pointer, incremented by lr3 each j)
#   lr8=0    (weight byte offset, incremented by lr12 each j)
#   lr9=0    (j counter)
#   lr10=144 (j-loop limit)
#   lr12=384 (W_STRIDE = 384 bytes per j)
#
# Memory layout:
#   DATA:    288 × 2 × 128 B =  73 728 B (0x00000..0x11FFF)
#   WEIGHTS: 144 rows × 384 B =  55 296 B (0x20000..0x2D7FF)
#   OUTPUT:  2 × 144 × 512 B = 147 456 B  (0x40000..0x63FFF)

j_loop:
    RESET_ACC;;
    SET lr4 cr6; LDR_MULT_REG r0 lr8 cr1;;  # tg=0 startup; r0 = W[j, 0..127]
    SET lr5 cr8;;                            # chunk0 fixed_idx startup: -1

k_chunk0_tg0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk0_tg0;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;  # chunk1 fixed_idx startup; r0 = W[j, 128..255]

k_chunk1_tg0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk1_tg0;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr3;;  # chunk2 fixed_idx startup; r0 = W[j, 256..287]+zeros

k_chunk2_tg0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk2_tg0;;

    STR_ACC_REG lr7 cr4;;                   # store 512B → OUTPUT[j, tg=0]

    RESET_ACC;;
    SET lr4 cr7; LDR_MULT_REG r0 lr8 cr1;;  # tg=1 startup; r0 = W[j, 0..127]
    SET lr5 cr8;;

k_chunk0_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk0_tg1;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;

k_chunk1_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk1_tg1;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr3;;

k_chunk2_tg1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk2_tg1;;

    STR_ACC_REG lr7 cr5;;                   # store 512B → OUTPUT[j, tg=1]
    ADD lr7 lr7 lr3;;                       # advance output ptr

    ADD lr8 lr8 lr12; ADD lr9 lr9 1;;       # next j: weight offset += W_STRIDE, j++
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
