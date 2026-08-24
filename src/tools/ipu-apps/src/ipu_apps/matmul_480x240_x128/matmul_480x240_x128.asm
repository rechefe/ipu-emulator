# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 5 FFN1)
#
# Layer:   L5
# Scope:   single-stream
# Layout:  unpacked
# Shape:   N_TOK=16, K=240, N=480
# Status:  validated
# Related: L5 FFN1 (single-stream, 2x output expansion). Sibling family:
#          matmul_240x240_x128 (OutProj), matmul_720x240_x128 (QKV),
#          matmul_240x480_x128 (FFN2, contracts back down). Consumed by
#          proj_ffn1_240_p4, which wraps the same arithmetic (and the same
#          silu activation) across all 4 pixel streams in one invocation.
# Tests:   //src/tools/ipu-apps:test_matmul_480x240_x128_wide
#
# Computes the L5 FFN1 matmul: 240 input channels expanded to 480 hidden
# channels, over 16 tokens (single token group; ROW-numbered operands per
# issue #179). K=240 splits into two weight-row chunks with the data
# pointer advancing one row at a time continuously across chunks and the
# per-chunk scalar index resetting; the first chunk's first iteration is
# peeled to seed r_acc via ACC.ADD.FIRST, the rest accumulate with ACC.ADD.
# Per kernel_docs/kernel_layer_map.md, the store applies ACTIVATE.QUANTIZE
# silu (not identity) since FFN1 feeds directly into FFN2 with no other
# nonlinearity between them.
#
# Single token group (N_TOK=16 <= 128): one accumulate+store pass per output j.
#
# All .asm operands are ROW numbers (issue #179), one row = 128 lanes.
# D: channel-major [K=240 channels, 16 tokens], one channel per WHOLE row (16
#    valid, padded to 128).  Row k at DATA_BASE + k.
# W: output-major [480 out_ch, 240 in_ch], NO transposition; 2 chunks/row-pairs
#    per output channel.
#    W[j, 0..127] at WEIGHTS_BASE + j*2 + 0
#    W[j, 128..239] at WEIGHTS_BASE + j*2 + 1 (padded to a whole row)
# C: channel-major [480 out_ch, 16 tokens] FP32, one row per channel at
#    OUTPUT_BASE + j.
#
# lr4 (data ptr) advances continuously by 1 row across chunks; lr5 reset to -1 per chunk.
# Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT).
#
# CRs: cr0=DATA_BASE, cr9=WEIGHTS_BASE, cr2=WB+1 row, cr5=OUTPUT_BASE, cr6=-1 row (data startup), cr8=-1 (chunk startup)
# LRs: lr0=0, lr2=1 row (data stride), lr3=1 row (output stride), lr6=126 (width-128 bound),
#      lr7=0 (out ptr), lr8=0 (weight offset), lr9=0 (j), lr10=480 (j limit),
#      lr11=110 (tail-chunk bound, width=112), lr12=2 rows (W_STRIDE)
#
# Memory layout (row counts):
#   DATA:    240 rows
#   WEIGHTS: 960 rows (480 output channels x 2 rows/channel)
#   OUTPUT:  480 rows

j_loop:
    SET lr4 cr6;
    LDR_MULT_REG r0 lr8 cr9;;  # data startup -128; r0 = W[j, chunk0]
    SET lr5 cr8;;                            # chunk0 fixed_idx startup: -1
    SUB lr5 lr5 cr1;;                        # biased to -2 (load runs a bundle ahead)

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    # Prime k=0's chunk one bundle ahead: MULT.RC.VE reads r_cyclic from the
    # start-of-cycle snapshot (issue #157), so it cannot consume a chunk loaded
    # in its OWN bundle -- it would multiply against the previous row. chunk1+
    # must NOT re-prime: their first row is already in flight from the previous
    # chunk's trailing prefetch.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0;
    ADD lr4 lr4 lr2;
    ADD lr5 lr5 cr1;;

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
    # result and the store is free. `silu` (FFN1 nonlinearity) + valid_elements=128
    # (the CR default) applies x*sigmoid(x) lane-for-lane to r_acc in wide mode.
    ACTIVATE.QUANTIZE silu cr15;
    STR_POST_AAQ_REG lr7 cr5;;  # store 512B -> OUTPUT[j] (first 64B valid)
    ADD lr7 lr7 lr3;;                        # advance output ptr (packed)

    ADD lr8 lr8 lr12;
    ADD lr9 lr9 cr1;;  # next j
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
