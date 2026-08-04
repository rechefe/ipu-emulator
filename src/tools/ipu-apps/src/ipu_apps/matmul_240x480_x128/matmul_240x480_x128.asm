# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 5 FFN2)
#
# Single token group (N_TOK=16 <= 128): one accumulate+store pass per output j.
#
# D: channel-major [K=480 channels, 128 tokens]  (16 valid, padded to 128)
#    Row k at DATA_BASE + k*128
# W: output-major [240 out_ch, 480 in_ch], NO transposition; 4 chunks of <=128 bytes
#    W[j, 0..127] at WEIGHTS_BASE + j*512 + 0
#    W[j, 128..255] at WEIGHTS_BASE + j*512 + 128
#    W[j, 256..383] at WEIGHTS_BASE + j*512 + 256
#    W[j, 384..479] at WEIGHTS_BASE + j*512 + 384 (padded to 128)
# C: channel-major [240 out_ch, 16 tokens] FP32, packed at OUTPUT_BASE + j*64
#
# lr4 (data ptr) advances continuously by 128 across chunks; lr5 reset to -1 per chunk.
# Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT).
#
# CRs: cr0=DATA_BASE, cr9=WEIGHTS_BASE, cr2=WB+128, cr3=WB+256, cr4=WB+384, cr5=OUTPUT_BASE, cr6=-128 (data startup), cr8=-1 (chunk startup)
# LRs: lr0=0, lr2=128 (data stride), lr3=64 (output stride), lr6=126 (width-128 bound),
#      lr7=0 (out ptr), lr8=0 (weight offset), lr9=0 (j), lr10=240 (j limit),
#      lr11=94 (tail-chunk bound, width=96), lr12=512 (W_STRIDE)
#
# Memory layout:
#   DATA:    480 x 128 B      =   61440 B (0x00000..0x0EFFF)
#   WEIGHTS: 240 rows x 512 B =  122880 B (0x10000..0x2DFFF)
#   OUTPUT:  240 rows x 64 B =   15360 B (0x30000..0x33BFF)

j_loop:
    SET lr4 cr6; LDR_MULT_REG r0 lr8 cr9;;   # data startup -128; r0 = W[j, chunk0]
    SET lr5 cr8;;                            # chunk0 fixed_idx startup: -1
    SUB lr5 lr5 cr1;;                        # biased to -2 (load runs a bundle ahead)

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    # Prime k=0's chunk one bundle ahead: MULT.RC.VE reads r_cyclic from the
    # start-of-cycle snapshot (issue #157), so it cannot consume a chunk loaded
    # in its OWN bundle -- it would multiply against the previous row. chunk1+
    # must NOT re-prime: their first row is already in flight from the previous
    # chunk's trailing prefetch.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;;

    MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    BLT lr5 lr6 k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    BLT lr5 lr6 k_chunk0;;

after_chunk0:

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;   # chunk1 startup; r0 = W[j, chunk1]

k_chunk1:
    MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    BLT lr5 lr6 k_chunk1;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr3;;   # chunk2 startup; r0 = W[j, chunk2]

k_chunk2:
    MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    BLT lr5 lr6 k_chunk2;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr4;;   # chunk3 startup; r0 = W[j, chunk3]

k_chunk3:
    MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    BLT lr5 lr11 k_chunk3;;

    # Hardware store path (see docs/content/specs/stage-aaq-str.md section 7.0):
    # AaQ and STR are consecutive pipeline stages WITHIN one VLIW word
    # (CTRL -> MULT -> ACC -> AaQ -> STR), so STR consumes this cycle's AaQ
    # result and the store is free. `identity` + valid_elements=128 (the CR
    # default) makes it a lane-for-lane FP32 copy of r_acc in wide mode.
    ACTIVATE.QUANTIZE identity cr15; STR_POST_AAQ_REG lr7 cr5;;                    # store 512B -> OUTPUT[j] (first 64B valid)
    ADD lr7 lr7 lr3;;                        # advance output ptr (packed)

    ADD lr8 lr8 lr12; ADD lr9 lr9 cr1;;        # next j
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
