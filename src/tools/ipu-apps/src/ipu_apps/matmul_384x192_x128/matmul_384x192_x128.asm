# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 4 FFN1)
#
# Single token group (N_TOK=64 <= 128): one accumulate+store pass per output j.
#
# D: channel-major [K=192 channels, 128 tokens]  (64 valid, padded to 128)
#    Row k at DATA_BASE + k*128
# W: output-major [384 out_ch, 192 in_ch], NO transposition; 2 chunks of <=128 bytes
#    W[j, 0..127] at WEIGHTS_BASE + j*256 + 0
#    W[j, 128..191] at WEIGHTS_BASE + j*256 + 128 (padded to 128)
# C: channel-major [384 out_ch, 64 tokens] FP32, packed at OUTPUT_BASE + j*256
#
# lr4 (data ptr) advances continuously by 128 across chunks; lr5 reset to -1 per chunk.
# Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT).
#
# CRs: cr0=DATA_BASE, cr9=WEIGHTS_BASE, cr2=WB+128, cr5=OUTPUT_BASE, cr6=-128 (data startup), cr8=-1 (chunk startup)
# LRs: lr0=0, lr2=128 (data stride), lr3=256 (output stride), lr6=126 (width-128 bound),
#      lr7=0 (out ptr), lr8=0 (weight offset), lr9=0 (j), lr10=384 (j limit),
#      lr11=62 (tail-chunk bound, width=64), lr12=256 (W_STRIDE)
#
# Memory layout:
#   DATA:    192 x 128 B      =   24576 B (0x00000..0x05FFF)
#   WEIGHTS: 384 rows x 256 B =   98304 B (0x10000..0x27FFF)
#   OUTPUT:  384 rows x 256 B =   98304 B (0x30000..0x47FFF)

j_loop:
    SET lr4 cr6; LDR_MULT_REG r0 lr8 cr9;;   # data startup -128; r0 = W[j, chunk0]
    SET lr5 cr8;;                            # chunk0 fixed_idx startup: -1

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 lr6 k_chunk0;;
    B after_chunk0;;

k_chunk0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr6 k_chunk0;;

after_chunk0:
    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;   # chunk1 startup; r0 = W[j, chunk1]

k_chunk1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 lr11 k_chunk1;;

    STR_ACC_REG lr7 cr5;;                    # store 512B -> OUTPUT[j] (first 256B valid)
    ADD lr7 lr7 lr3;;                        # advance output ptr (packed)

    ADD lr8 lr8 lr12; ADD lr9 lr9 cr1;;        # next j
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
