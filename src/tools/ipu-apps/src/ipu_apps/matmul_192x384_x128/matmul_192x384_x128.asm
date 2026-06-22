# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 4 FFN2)
#
# Single token group (N_TOK=64 <= 128): one accumulate+store pass per output j.
#
# D: channel-major [K=384 channels, 128 tokens]  (64 valid, padded to 128)
#    Row k at DATA_BASE + k*128
# W: output-major [192 out_ch, 384 in_ch], NO transposition; 3 chunks of <=128 bytes
#    W[j, 0..127] at WEIGHTS_BASE + j*384 + 0
#    W[j, 128..255] at WEIGHTS_BASE + j*384 + 128
#    W[j, 256..383] at WEIGHTS_BASE + j*384 + 256
# C: channel-major [192 out_ch, 64 tokens] FP32, packed at OUTPUT_BASE + j*256
#
# lr4 (data ptr) advances continuously by 128 across chunks; lr5 reset to -1 per chunk.
# Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT).
#
# CRs: cr0=DATA_BASE, cr9=WEIGHTS_BASE, cr2=WB+128, cr3=WB+256, cr5=OUTPUT_BASE, cr6=-128 (data startup), cr8=-1 (chunk startup)
# LRs: lr0=0, lr2=128 (data stride), lr3=256 (output stride), lr6=126 (width-128 bound),
#      lr7=0 (out ptr), lr8=0 (weight offset), lr9=0 (j), lr10=192 (j limit),
#      lr11=126 (tail-chunk bound, width=128), lr12=384 (W_STRIDE)
#
# Memory layout:
#   DATA:    384 x 128 B      =   49152 B (0x00000..0x0BFFF)
#   WEIGHTS: 192 rows x 384 B =   73728 B (0x10000..0x21FFF)
#   OUTPUT:  192 rows x 256 B =   49152 B (0x30000..0x3BFFF)

j_loop:
    RESET_ACC;;
    SET lr4 cr6; LDR_MULT_REG r0 lr8 cr9;;   # data startup -128; r0 = W[j, chunk0]
    SET lr5 cr8;;                            # chunk0 fixed_idx startup: -1

k_chunk0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk0;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;   # chunk1 startup; r0 = W[j, chunk1]

k_chunk1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk1;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr3;;   # chunk2 startup; r0 = W[j, chunk2]

k_chunk2:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk2;;

    STR_ACC_REG lr7 cr5;;                    # store 512B -> OUTPUT[j] (first 256B valid)
    ADD lr7 lr7 lr3;;                        # advance output ptr (packed)

    ADD lr8 lr8 lr12; ADD lr9 lr9 1;;        # next j
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
