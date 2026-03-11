# Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]
#
# D: channel-major [144 channels, 2 token-groups, 128 tokens]
#    Row (k, tg) at DATA_BASE + k*256 + tg*128  (256 bytes per channel, 128 per tg)
# W: output-major [432 out_ch, 144 in_ch], NO transposition
#    W[j, 0..127]   at WEIGHTS_BASE + j*256
#    W[j, 128..143] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
# C: channel-major [432 out_ch, 2 tg, 128 tokens], FP32 accumulators
#    Row (j, tg) at OUTPUT_BASE + (j*2 + tg)*512
#
# CRs:
#   cr0 = DATA_BASE
#   cr1 = WEIGHTS_BASE
#   cr2 = WEIGHTS_BASE + 128   (second 128-byte chunk of weights)
#   cr3 = OUTPUT_BASE
#
# LRs:
#   lr0 = 0    (const: r_cyclic slot-0 index)
#   lr1 = 128  (const: r_cyclic slot-1 index)
#   lr4 = inner data pointer (reset per j/tg via set)
#   lr5 = inner cyclic index (reset per j/tg via set)
#   lr6 = 143  (K-1, inner loop bound)
#   lr7 = output pointer (+512 per tg, so +1024 per j)
#   lr8 = weight byte offset (+256 per j)
#   lr9 = j counter (0..431)
#   lr10 = 432 (j-loop limit; using counter avoids 16-bit immediate overflow)
#
# One-cycle startup offset pattern (pipeline alignment):
#   Slot order within a compound instruction: XMEM → MULT → ACC → LR → COND
#   XMEM reads the current lr4 BEFORE LR increments it in the same cycle.
#   Likewise, MULT reads r_cyclic at lr5 BEFORE LR increments it.
#   So we initialise both one step behind the true first value:
#     lr4 = -256: dummy XMEM reads DATA_BASE-256 (before allocated region → zeros)
#     lr5 = -1:   dummy MULT reads r_cyclic[511] = 0  → contributes 0 to accumulator
#   After that dummy cycle lr4=0, lr5=0, and real k=0 data/weight is used from cycle 2.
#
#   tg=0: set lr4=-256 → first real XMEM addr = DATA_BASE + 0       = D[k=0, tg=0]
#   tg=1: set lr4=-128 → first real XMEM addr = DATA_BASE + 128     = D[k=0, tg=1]
#          (−256 + 128 = −128 because tg=1 starts 128B into each channel block)
#
# Memory layout:
#   DATA:    144 channels × 256 B = 36 864 B  (0x00000..0x08FFF)
#   WEIGHTS: 432 rows × 256 B    = 110 592 B (0x10000..0x2AFFF)
#   OUTPUT:  432×2 rows × 512 B  = 442 368 B (0x30000..0x9BFFF)

    set                 lr0 0;;
    set                 lr1 128;;
    set                 lr6 143;;
    set                 lr7 0;;
    set                 lr8 0;;
    set                 lr9 0;;
    set                 lr10 432;;

j_loop:
    ldr_cyclic_mult_reg lr8 cr1 lr0;;    # r_cyclic[0..127]   = W[j, 0..127]
    ldr_cyclic_mult_reg lr8 cr2 lr1;;    # r_cyclic[128..255] = W[j, 128..143] + zeros

    # -- token group 0 -------------------------------------------------------
    reset_acc;;
    set                 lr4 -256;;
    set                 lr5 -1;;

k_loop_tg0:                                  # K+1 iterations: 1 dummy startup + K real (k=0..143)
    ldr_mult_reg        mem_bypass lr4 cr0;  # XMEM : mem_bypass = D[k, tg=0]  (128 bytes, cache bypassed)
    incr                lr4 256;             # LR_A : lr4 += 256  (stride to next channel block)
    incr                lr5 1;               # LR_B : lr5 += 1    (cyclic index: −1 → 0 → … → 143)
    mult.ev             mem_bypass lr5 lr0 lr0; # MULT: 128 PE outputs = r_cyclic[lr5_old] × mem_bypass
    acc;                                     # ACC  : accumulator[0..127] += mult result  (FP32)
    blt                 lr5 lr6 k_loop_tg0;; # COND : branch while lr5 < 143; exit when lr5 == 143

    str_acc_reg         lr7 cr3;;            # store 512 B accumulator → OUTPUT[j, tg=0]
    incr                lr7 512;;            # advance output ptr by one row (512 B = 128 FP32 words)

    # -- token group 1 -------------------------------------------------------
    reset_acc;;                              # clear accumulator for tg=1 pass (weights stay in r_cyclic)
    set                 lr4 -128;;           # startup: tg=1 data starts at +128 within each channel block
    set                 lr5 -1;;             # reset cyclic index for second pass

k_loop_tg1:                                  # identical structure; iterates over same k, different tg
    ldr_mult_reg        mem_bypass lr4 cr0;  # XMEM : mem_bypass = D[k, tg=1]
    incr                lr4 256;             # LR_A : lr4 += 256
    incr                lr5 1;               # LR_B : lr5 += 1
    mult.ev             mem_bypass lr5 lr0 lr0; # MULT: r_cyclic[lr5_old] × mem_bypass
    acc;                                     # ACC  : accumulator += mult result
    blt                 lr5 lr6 k_loop_tg1;; # COND : branch while lr5 < 143

    str_acc_reg         lr7 cr3;;            # store 512 B accumulator → OUTPUT[j, tg=1]
    incr                lr7 512;;            # advance output ptr

    incr                lr8 256;             # advance weight offset to W[j+1, :]  (W_STRIDE = 256 B)
    incr                lr9 1;;              # increment j counter
    blt                 lr9 lr10 j_loop;;    # loop while j < 432

end:
    bkpt;;
