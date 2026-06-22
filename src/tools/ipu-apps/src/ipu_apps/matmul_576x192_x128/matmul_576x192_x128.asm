# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 4 QKV)
#
# Single token group (N_TOK=64 <= 128): one accumulate+store pass per output j.
#
# D: channel-major [K=192 channels, 128 tokens]  (64 valid tokens, padded to 128)
#    Row k at DATA_BASE + k*128
# W: output-major [576 out_ch, 192 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*256
#    W[j, 128..191] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
# C: channel-major [576 out_ch, 64 tokens] FP32, packed contiguously
#    Row j at OUTPUT_BASE + j*256  (256 = 64 tokens * 4 bytes)
#
# Algorithm: K=192 split into two chunks per j ([128, 64]).
#   Per chunk: load W[j, chunk*128..] into r0, run width k-steps.
#   MULT.VE.CYCLIC r0[lr5] x r_cyclic[:] — lr5 cycles 0..width-1 per chunk.
#   lr4 (data pointer) advances continuously by 128; NOT reset between chunks.
#   lr5 reset to -1 at the start of each chunk via SET lr5 cr8.
#
# Loop-bound formula (do-while, live MULT/XMEM, snapshot BLT):
#   body runs for live index [lr5_start+1 .. bound+1]; bound = width-2.
#   chunk0 width=128 -> bound 126 (lr6);  chunk1 width=64 -> bound 62 (lr11).
#
# CRs: cr0=DATA_BASE, cr9=WEIGHTS_BASE (off read-only CR1), cr2=WEIGHTS_BASE+128,
#       cr3=OUTPUT_BASE
#       cr6=-128 (data startup), cr8=-1 (per-chunk fixed_idx startup)
# LRs (preset by harness):
#   lr0=0    (const: r_cyclic write-index 0)
#   lr2=128  (data stride: 128 bytes/channel)
#   lr3=256  (output stride: 64 tokens * 4 bytes, packed)
#   lr6=126  (chunk0 bound: width=128)
#   lr7=0    (output pointer, incremented by lr3 each j)
#   lr8=0    (weight byte offset, incremented by lr12 each j)
#   lr9=0    (j counter)
#   lr10=576 (j-loop limit)
#   lr11=62  (chunk1 bound: width=64)
#   lr12=256 (W_STRIDE = 256 bytes per j)
#
# Memory layout:
#   DATA:    192 x 128 B       =  24 576 B (0x00000..0x05FFF)
#   WEIGHTS: 576 rows x 256 B  = 147 456 B (0x10000..0x33FFF)
#   OUTPUT:  576 rows x 256 B  = 147 456 B (0x40000..0x63FFF, +256B spill)

j_loop:
    RESET_ACC;;
    SET lr4 cr6; LDR_MULT_REG r0 lr8 cr9;;   # data startup -128; r0 = W[j, 0..127]
    SET lr5 cr8;;                            # chunk0 fixed_idx startup: -1

k_chunk0:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr6 k_chunk0;;

    SET lr5 cr8; LDR_MULT_REG r0 lr8 cr2;;   # chunk1 startup; r0 = W[j, 128..191]+zeros

k_chunk1:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr2; ADD lr5 lr5 1;
    MULT.VE.CYCLIC lr0 0 lr0 lr5; ACC; BLT lr5 lr11 k_chunk1;;

    STR_ACC_REG lr7 cr3;;                    # store 512B -> OUTPUT[j] (first 256B valid)
    ADD lr7 lr7 lr3;;                        # advance output ptr (packed, 256B)

    ADD lr8 lr8 lr12; ADD lr9 lr9 1;;        # next j: weight offset += W_STRIDE, j++
    BLT lr9 lr10 j_loop;;

end:
    BKPT;;
