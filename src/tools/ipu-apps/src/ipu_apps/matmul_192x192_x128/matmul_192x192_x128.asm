# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 4 OutProj)
#
# Single token group (N_TOK=64 <= 128): one accumulate+store pass per output j.
#
# D: channel-major [K=192 channels, 128 tokens]  (64 valid, padded to 128)
#    Row k at DATA_BASE + k*128
# W: output-major [192 out_ch, 192 in_ch], NO transposition; 2 chunks of <=128 bytes
#    W[j, 0..127] at WEIGHTS_BASE + j*256 + 0
#    W[j, 128..191] at WEIGHTS_BASE + j*256 + 128 (padded to 128)
# C: channel-major [192 out_ch, 64 tokens] FP32, packed at OUTPUT_BASE + j*256
#
# lr4 (data ptr) advances continuously by 128 across chunks; lr5 reset to -1 per chunk.
# Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT).
#
# MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
# the start-of-cycle snapshot while keeping the LR index live, so it cannot
# consume the chunk LDR_CYCLIC_MULT_REG loads in its own bundle. `;;` ends one
# VLIW word = one cycle = one snapshot, so a load and a MULT in the same bundle
# always run in the same cycle regardless of textual order -- co-issuing is
# fine, consuming the same-cycle load is not.
#   chunk0 primes k=0's row, then each loop body multiplies the row loaded last
#   cycle while prefetching the next. lr4 walks CONTINUOUSLY into chunk1 while
#   lr5 resets, so chunk1's first row is ALREADY IN FLIGHT from chunk0's
#   trailing prefetch: chunk1 must NOT re-prime.
#   lr5 keeps its ADD co-issued with the MULT and the BLT (moving it to its own
#   word would cost a cycle per contraction step). MULT reads lr5 LIVE and BLT
#   reads the SNAPSHOT, exactly as before, so lr6/lr11 stay valid; only chunk0's
#   startup is biased down by one (-1 -> -2) because the load now runs a bundle
#   ahead. chunk1's startup is NOT biased -- chunk0's trailing prefetch already
#   supplied that step of phase.
#
# CRs: cr0=DATA_BASE, cr9=WEIGHTS_BASE, cr2=WB+128, cr5=OUTPUT_BASE, cr6=-128 (data startup), cr8=-1 (chunk startup)
# LRs: lr0=0, lr2=128 (data stride), lr3=256 (output stride), lr6=126 (width-128 bound),
#      lr7=0 (out ptr), lr8=0 (weight offset), lr9=0 (j), lr10=192 (j limit),
#      lr11=62 (tail-chunk bound, width=64), lr12=256 (W_STRIDE)
#
# Memory layout:
#   DATA:    192 x 128 B      =   24576 B (0x00000..0x05FFF)
#   WEIGHTS: 192 rows x 256 B =   49152 B (0x10000..0x1BFFF)
#   OUTPUT:  192 rows x 256 B =   49152 B (0x20000..0x2BFFF)

j_loop:
    SET lr4 cr6; LDR_MULT_REG r0 lr8 cr9;;   # data startup -128; r0 = W[j, chunk0]
    SET lr5 cr8;;                            # chunk0 fixed_idx startup: -1
    SUB lr5 lr5 cr1;;                        # biased to -2 (load runs a bundle ahead)

    # Prime k=0's row for chunk0 (see the snapshot note in the header).
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;;

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    BLT lr5 lr6 k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    BLT lr5 lr6 k_chunk0;;

after_chunk0:
    # No re-prime: chunk1's first row is already in flight from chunk0's
    # trailing prefetch, and lr5 is NOT biased here for the same reason.
    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;   # chunk1 startup; r0 = W[j, chunk1]

k_chunk1:
    MULT.RC.VE lr0 lr5 0 lr0 cr15; ACC.ADD;
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 cr1;
    BLT lr5 lr11 k_chunk1;;

    ACTIVATE.QUANTIZE identity cr15; STR_POST_AAQ_REG lr7 cr5;;# store 512B -> OUTPUT[j] (first 256B valid)
    ADD lr7 lr7 lr3;;                        # advance output ptr (packed)

    ADD lr8 lr8 lr12; ADD lr9 lr9 cr1;;      # next j
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
