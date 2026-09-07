# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 5 OutProj)
#
# Layer:   L5
# Scope:   single-stream
# Layout:  unpacked
# Shape:   N_TOK=16, K=240, N=240
# Status:  validated
# Related: L5 OutProj (single-stream). Sibling family: matmul_480x240_x128
#          (FFN1, K=240->480, silu), matmul_720x240_x128 (QKV, K=240->720),
#          matmul_240x480_x128 (FFN2, K=480->240). Consumed by
#          proj_outproj_240_p4, which wraps the same arithmetic across all 4
#          pixel streams in one invocation, and by
#          test_seam_layernorm_matmul_xmem_direct_l5 (L5 layernorm->matmul
#          seam) plus the "L5 matmul output-crop fix" (row-based, full-row
#          OUTPUT_ROW_BYTES=512, per kernel_docs/kernel_layer_map.md).
# Tests:   //src/tools/ipu-apps:test_matmul_240x240_x128_wide
#
# Computes the L5 OutProj matmul: 240 input channels contracted down to 240
# output channels, over 16 tokens (a single token group; all .asm operands
# here are ROW numbers, one row = 128 lanes, per issue #179). K=240 splits
# into two weight-row chunks (0..127 from r0, 128..239 from a reload); the
# data pointer advances one row at a time continuously across chunks while
# the per-chunk scalar index resets, with the first chunk's first iteration
# peeled to seed r_acc via ACC.ADD.FIRST and the remainder accumulated with
# ACC.ADD.
#
# Single token group (N_TOK=16 <= 128): one accumulate+store pass per output j.
#
# All .asm operands are ROW numbers (issue #179), one row = 128 lanes.
# D: channel-major [K=240 channels, 16 tokens], one channel per WHOLE row (16
#    valid, padded to 128).  Row k at DATA_BASE + k.
# W: output-major [240 out_ch, 240 in_ch], NO transposition; 2 chunks/row-pairs
#    per output channel.
#    W[j, 0..127] at WEIGHTS_BASE + j*2 + 0
#    W[j, 128..239] at WEIGHTS_BASE + j*2 + 1 (padded to a whole row)
# C: channel-major [240 out_ch, 16 tokens] FP32, one row per channel at
#    OUTPUT_BASE + j.
#
# lr4 (data ptr) advances continuously by 1 row across chunks; lr5 reset to -1 per chunk.
# Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT).
#
# CRs: cr0=DATA_BASE, cr9=WEIGHTS_BASE, cr2=WB+1 row, cr5=OUTPUT_BASE, cr6=-1 row (data startup), cr8=-1 (chunk startup)
# LRs: lr0=0, lr2=1 row (data stride), lr3=1 row (output stride), lr6=126 (width-128 bound),
#      lr7=0 (out ptr), lr8=0 (weight offset), lr9=0 (j), lr10=240 (j limit),
#      lr11=110 (tail-chunk bound, width=112), lr12=2 rows (W_STRIDE)
#
# Memory layout (row counts):
#   DATA:    240 rows
#   WEIGHTS: 480 rows (240 output channels x 2 rows/channel)
#   OUTPUT:  240 rows

j_loop:
    SET lr4 cr6;
    LDR_MULT_REG r0 lr8 cr9;;  # data startup -128; r0 = W[j, chunk0]
    SET lr5 cr8;;                            # chunk0 fixed_idx startup: -1
    SUB lr5 lr5 cr1;;                        # biased to -2 (load runs a bundle ahead)

    # Prime k=0's chunk one bundle ahead: MULT.RC.VE reads r_cyclic from the
    # start-of-cycle snapshot (issue #157), so it cannot consume a chunk loaded
    # in its OWN bundle -- it would multiply against the previous row. Each load
    # therefore runs one bundle ahead of the MULT that consumes it.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0;
    ADD lr4 lr4 lr2;
    ADD lr5 lr5 cr1;;

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    MULT.RC.VE lr0 lr5 0 lr0 cr15;
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0;
    ADD lr4 lr4 lr2;
    ADD lr5 lr5 cr1;
    BLT lr5 lr6 k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE lr0 lr5 0 lr0 cr15;
    ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0;
    ADD lr4 lr4 lr2;
    ADD lr5 lr5 cr1;
    BLT lr5 lr6 k_chunk0;;

after_chunk0:
    SET lr5 cr8;
    LDR_MULT_REG r0 lr8 cr2;;  # chunk1 startup; r0 = W[j, chunk1]

k_chunk1:
    MULT.RC.VE lr0 lr5 0 lr0 cr15;
    ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0;
    ADD lr4 lr4 lr2;
    ADD lr5 lr5 cr1;
    BLT lr5 lr11 k_chunk1;;

    # Hardware store path (see docs/content/specs/stage-aaq-str.md section 7.0):
    # AaQ and STR are consecutive pipeline stages WITHIN one VLIW word
    # (CTRL -> MULT -> ACC -> AaQ -> STR), so STR consumes this cycle's AaQ
    # result and the store is free. `identity` + valid_elements=128 (the CR
    # default) makes it a lane-for-lane FP32 copy of r_acc in wide mode.
    ACTIVATE.QUANTIZE identity cr15;
    STR_POST_AAQ_REG lr7 cr5;;  # store 512B -> OUTPUT[j] (first 64B valid)
    ADD lr7 lr7 lr3;;                        # advance output ptr (packed)

    ADD lr8 lr8 lr12;
    ADD lr9 lr9 cr1;;  # next j
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
