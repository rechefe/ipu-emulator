# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]
#
# D: interleaved channel-major [144 channels, 2 tg, 128 tokens]
#    Row (k, tg) at DATA_BASE + k*256 + tg*128
# W: output-major [288 out_ch, 144 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*256
#    W[j, 128..143] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
# C: grouped channel-major [2 tg, 288 out_ch, 128 tokens], FP32 accumulators
#    Row (j, tg) at OUTPUT_BASE + tg*N_OUT*512 + j*512
#
# (No activation applied — Swish to be added when ISA support is available)
#
# CRs:
#   cr0 = DATA_BASE
#   cr1 = WEIGHTS_BASE
#   cr2 = WEIGHTS_BASE + 128   (second 128-byte chunk of weights)
#   cr3 = OUTPUT_BASE                    (tg=0 output)
#   cr4 = OUTPUT_BASE + N_OUT*512        (tg=1 output)
#
# LRs:
#   lr0 = 0   (const: r_cyclic slot-0 index, holds W[j, 0..127])
#   lr1 = 128 (const: r_cyclic slot-1 index, holds W[j, 128..143] + zeros)
#   lr4 = inner data pointer (reset per j/tg via set)
#   lr5 = inner cyclic index (reset per j/tg via set)
#   lr6 = 143 (K−1, inner loop bound)
#   lr7 = output pointer (+512 per j, shared by both tgs)
#   lr8 = weight byte offset (+256 per j)
#   lr9 = j counter (0..287)
#   lr10 = 288 (j-loop limit; using counter avoids 16-bit immediate overflow)
#
# One-cycle startup offset pattern (pipeline alignment):
#   tg=0: set lr4=-256, stride 256 → first real load = D[k=0, tg=0] at DATA_BASE+0
#   tg=1: set lr4=-128, stride 256 → first real load = D[k=0, tg=1] at DATA_BASE+128
#          (interleaved: D[k,tg] at k*256 + tg*128; startup = first_addr - stride)
#
# Memory layout:
#   DATA:    2 × 144 × 128 B = 36 864 B  (0x00000..0x08FFF)
#   WEIGHTS: 288 rows × 256 B =  73 728 B  (0x10000..0x21FFF)
#   OUTPUT:  2 × 288 × 512 B = 294 912 B  (0x30000..0x77FFF)
#   NOTE: OUTPUT_BASE = 0x30000, not 0x20000 — weights end at 0x22000,
#         so 0x20000 would overlap and corrupt weights for j ≥ 256.

    set                 lr0 0;;
    set                 lr1 128;;
    set                 lr6 143;;
    set                 lr7 0;;
    set                 lr8 0;;
    set                 lr9 0;;
    set                 lr10 288;;

j_loop:
    ldr_cyclic_mult_reg lr8 cr1 lr0;;    # r_cyclic[0..127]   = W[j, 0..127]
    ldr_cyclic_mult_reg lr8 cr2 lr1;;    # r_cyclic[128..255] = W[j, 128..143] + zeros

    # -- token group 0 -------------------------------------------------------
    reset_acc;;
    set                 lr4 -256;;         # startup offset: interleaved tg=0 stride=256
    set                 lr5 -1;;

k_loop_tg0:                                  # K+1 iterations: 1 dummy startup + K real (k=0..143)
    ldr_mult_reg        mem_bypass lr4 cr0;  # XMEM : mem_bypass = D[k, tg=0]  (128 bytes, bypasses cache)
    incr                lr4 256;             # LR_A : lr4 += 256  (stride to next channel)
    incr                lr5 1;               # LR_B : lr5 += 1    (cyclic index: −1 → 0 → … → 143)
    mult.ev             mem_bypass lr5 lr0 lr0; # MULT: 128 outputs = r_cyclic[lr5_old] × mem_bypass
    acc;                                     # ACC  : accumulator[0..127] += mult result  (FP32)
    blt                 lr5 lr6 k_loop_tg0;; # COND : branch while lr5 < 143; exit when lr5 == 143

    str_acc_reg         lr7 cr3;;            # store 512 B accumulator → OUTPUT[j, tg=0]

    # -- token group 1 -------------------------------------------------------
    reset_acc;;                              # clear accumulator for tg=1 pass (weights stay in r_cyclic)
    set                 lr4 -128;;           # startup: interleaved tg=1 first addr=128, stride=256, startup=128-256=-128
    set                 lr5 -1;;             # reset cyclic index for second pass

k_loop_tg1:                                  # identical structure; same k range, different tg offset
    ldr_mult_reg        mem_bypass lr4 cr0;  # XMEM : mem_bypass = D[k, tg=1]
    incr                lr4 256;             # LR_A : lr4 += 256
    incr                lr5 1;               # LR_B : lr5 += 1
    mult.ev             mem_bypass lr5 lr0 lr0; # MULT: r_cyclic[lr5_old] × mem_bypass
    acc;                                     # ACC  : accumulator += mult result
    blt                 lr5 lr6 k_loop_tg1;; # COND : branch while lr5 < 143

    str_acc_reg         lr7 cr4;;            # store 512 B accumulator → OUTPUT[j, tg=1]
    incr                lr7 512;;            # advance output ptr (once per j)

    incr                lr8 256;             # advance weight offset to W[j+1, :]  (W_STRIDE = 256 B)
    incr                lr9 1;;              # increment j counter
    blt                 lr9 lr10 j_loop;;    # loop while j < 288

end:
    bkpt;;
