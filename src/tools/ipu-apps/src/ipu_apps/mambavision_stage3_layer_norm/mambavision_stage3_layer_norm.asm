# mambavision_stage3_layer_norm.asm
#
# Layer Norm over MambaVision Stage 3 tokens, wide-vector FP32 mode.
#
#   x:     196 tokens x 320 channels  (token-major, contiguous per token)
#   gamma: 320                        (shared across all tokens)
#   beta:  320                        (shared across all tokens)
#   y:     196 x 320  =  gamma * (x - mean) * invstd(var+eps) + beta
#
# Channels are tiled 128 + 128 + 64 (320 total). Runs entirely in
# wide-vector FP32 mode (IpuState(wide_vector_debug=True, FP32,
# wide_vector_quantize_output=False)) -- no INT8 clamping anywhere.
#
# Per token: three read-passes over x --
#   Pass 1: mean      = sum(x * (1/N)) over all 320 channels
#   Pass 2: var       = sum((x-mean)^2) * (1/N) over all 320 channels
#           invstd    = rsqrt(var + eps)
#   Pass 3: y         = gamma * (x-mean) * invstd + beta, tile by tile
#
# CR map (all set in Python setup()):
#   cr0  = ZERO / INPUT_BASE  (both 0 -- CR0 is hardware-reserved, permanently 0)
#   cr1  = LITERAL_ONE        (CR1 is hardware-reserved, permanently 1)
#   cr2  = OUTPUT_BASE
#   cr3  = PARAMS_BASE      (gamma at +0, beta at +1280 bytes)
#   cr4  = CONST_SCRATCH_BASE (consts at +0: [1.0, 1/320, eps, ...zeros];
#                              scratch: mean+invstd at +512/+516,
#                              xminusmean at +1024, var/tilesums at +1536)
#   cr5  = TOKEN_ROW_STRIDE (1280 = 320*4 bytes)
#   cr6  = TOKEN_LOOP_END   (250880 = 196*1280)
#   cr7  = TILE_BYTES       (512 = 128*4 bytes)
#   cr8  = ONE_CONST_IDX    (128, combined R0++R1 index of R1[0]=1.0)
#   cr9  = RECIPN_IDX       (129, combined index of R1[1]=1/320)
#   cr10 = EPS_IDX          (130, combined index of R1[2]=eps)
#   cr11 = DSTRUCT_VALID1   (dstructure valid_elements=1)
#   cr12 = DSTRUCT_VALID64  (dstructure valid_elements=64)
#   cr13 = FOUR             (4, literal small constant)
#   cr15 = DSTRUCT_VALID128 (dstructure valid_elements=128)

    SET                 lr0 cr0 ;;      # lr0 = 0, token byte offset (loop var)
    SET                 lr1 cr6 ;;      # lr1 = TOKEN_LOOP_END
    SET                 lr4 cr0 ;;      # lr4 = 0 (zero constant, reused as index/offset)
    SET                 lr5 cr8 ;;      # lr5 = 128 (one_idx)
    SET                 lr6 cr9 ;;      # lr6 = 129 (recipn_idx)
    SET                 lr7 cr10 ;;     # lr7 = 130 (eps_idx)
    SET                 lr8 cr1 ;;     # lr8 = 1 (selects R0[1] once mean/invstd reloaded)
    SET                 lr10 cr7 ;;     # lr10 = 512 (mean scratch offset)
    ADD                 lr11 lr10 cr7 ;;  # lr11 = 1024 (xminusmean scratch offset)
    ADD                 lr12 lr11 cr7 ;;  # lr12 = 1536 (var scratch offset)
    SET                 lr15 cr13 ;;    # lr15 = 4
    ADD                 lr13 lr10 lr15 ;; # lr13 = 516 (invstd scratch offset, adjacent to mean)
    SET                 lr14 cr5 ;;     # lr14 = 1280 (beta sub-offset within PARAMS_BASE)

    # Load the small constants buffer [1.0, 1/320, eps, 0, ...] into R1 once.
    LDR_MULT_REG        r1 lr4 cr4 ;;

token_loop:

    # ================= PASS 1: mean = sum(x * 1/320) =================
    SET                 lr2 cr0 ;;      # lr2 = 0 (tile offset)

    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr6 0 lr4 cr15 ;;
    AGG.SUM.FIRST       lr4 cr15 ;;
    ADD                 lr2 lr2 cr7 ;;

    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr6 0 lr4 cr15 ;;
    AGG.SUM             lr4 cr15 ;;
    ADD                 lr2 lr2 cr7 ;;

    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr6 0 lr4 cr12 ;;
    AGG.SUM             lr4 cr12 ;;

    ACTIVATE.QUANTIZE   identity cr11 ;;
    STR_POST_AAQ_REG    lr10 cr4 ;;
    LDR_MULT_REG        r0 lr10 cr4 ;;   # R0[0] = mean

    # ================= PASS 2: var = sum((x-mean)^2) / 320 =================
    # NOTE: ACC.ADD.FIRST/ACC.SUB touch ALL 128 R_ACC lanes unconditionally
    # (no valid_elements limit), so they cannot coexist with a running
    # cross-tile AGG.SUM accumulator in the same slot -- each tile's own
    # sum-of-squares is computed independently and stashed in its own
    # scratch slot, then combined in one clean final reduction.
    SET                 lr2 cr0 ;;

    # -- tile 0 --
    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr15 ;;
    ACC.ADD.FIRST ;;
    MULT.EE             lr4 cr1 0 lr4 cr15 ;;
    ACC.SUB ;;
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr11 cr4 lr4 ;;
    MULT.RC.VS          lr4 0 lr4 cr15 ;;
    AGG.SUM.FIRST       lr4 cr15 ;;
    ACTIVATE.QUANTIZE   identity cr11 ;;
    STR_POST_AAQ_REG    lr12 cr4 ;;      # tile0 sumsq -> scratch+1536
    ADD                 lr2 lr2 cr7 ;;

    # -- tile 1 --
    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr15 ;;
    ACC.ADD.FIRST ;;
    MULT.EE             lr4 cr1 0 lr4 cr15 ;;
    ACC.SUB ;;
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr11 cr4 lr4 ;;
    MULT.RC.VS          lr4 0 lr4 cr15 ;;
    AGG.SUM.FIRST       lr4 cr15 ;;
    ACTIVATE.QUANTIZE   identity cr11 ;;
    ADD                 lr15 lr12 cr13 ;;  # lr15 = 1536+4 = 1540
    STR_POST_AAQ_REG    lr15 cr4 ;;      # tile1 sumsq -> scratch+1540
    ADD                 lr2 lr2 cr7 ;;

    # -- tile 2 --
    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr12 ;;
    ACC.ADD.FIRST ;;
    MULT.EE             lr4 cr1 0 lr4 cr12 ;;
    ACC.SUB ;;
    ACTIVATE.QUANTIZE   identity cr12 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr11 cr4 lr4 ;;
    MULT.RC.VS          lr4 0 lr4 cr12 ;;
    AGG.SUM.FIRST       lr4 cr12 ;;
    ACTIVATE.QUANTIZE   identity cr11 ;;
    ADD                 lr15 lr12 cr13 ;;
    ADD                 lr15 lr15 cr13 ;;  # lr15 = 1536+8 = 1544
    STR_POST_AAQ_REG    lr15 cr4 ;;      # tile2 sumsq -> scratch+1544

    # combine: reload the 3 adjacent partial sums, scale by 1/320, sum -> var
    LDR_CYCLIC_MULT_REG lr12 cr4 lr4 ;;
    MULT.RC.VE          lr4 lr6 0 lr4 cr12 ;;
    AGG.SUM.FIRST       lr4 cr12 ;;
    MULT.EE             lr7 cr1 0 lr4 cr11 ;;
    ACC.ADD ;;
    ACTIVATE.QUANTIZE   rsqrt cr11 ;;
    STR_POST_AAQ_REG    lr13 cr4 ;;

    # reload mean (still at scratch+512) and invstd (scratch+516) together
    LDR_MULT_REG        r0 lr10 cr4 ;;   # R0[0] = mean, R0[1] = invstd

    # ================= PASS 3: y = gamma*(x-mean)*invstd + beta =================
    SET                 lr2 cr0 ;;

    LDR_MULT_REG        r0 lr10 cr4 ;;
    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr15 ;;
    ACC.ADD.FIRST ;;
    MULT.EE             lr4 cr1 0 lr4 cr15 ;;
    ACC.SUB ;;
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr11 cr4 lr4 ;;
    MULT.RC.VE          lr4 lr8 0 lr4 cr15 ;;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_MULT_REG        r0 lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr2 cr3 lr4 ;;
    MULT.RC.VV          lr4 r0 0 lr4 cr15 ;;
    ACC.ADD.FIRST ;;
    ADD                 lr9 lr2 lr14 ;;
    LDR_CYCLIC_MULT_REG lr9 cr3 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr15 ;;
    ACC.ADD ;;
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr3 cr2 ;;
    ADD                 lr2 lr2 cr7 ;;

    LDR_MULT_REG        r0 lr10 cr4 ;;
    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr15 ;;
    ACC.ADD.FIRST ;;
    MULT.EE             lr4 cr1 0 lr4 cr15 ;;
    ACC.SUB ;;
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr11 cr4 lr4 ;;
    MULT.RC.VE          lr4 lr8 0 lr4 cr15 ;;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_MULT_REG        r0 lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr2 cr3 lr4 ;;
    MULT.RC.VV          lr4 r0 0 lr4 cr15 ;;
    ACC.ADD.FIRST ;;
    ADD                 lr9 lr2 lr14 ;;
    LDR_CYCLIC_MULT_REG lr9 cr3 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr15 ;;
    ACC.ADD ;;
    ACTIVATE.QUANTIZE   identity cr15 ;;
    STR_POST_AAQ_REG    lr3 cr2 ;;
    ADD                 lr2 lr2 cr7 ;;

    LDR_MULT_REG        r0 lr10 cr4 ;;
    ADD                 lr3 lr0 lr2 ;;
    LDR_CYCLIC_MULT_REG lr3 cr0 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr12 ;;
    ACC.ADD.FIRST ;;
    MULT.EE             lr4 cr1 0 lr4 cr12 ;;
    ACC.SUB ;;
    ACTIVATE.QUANTIZE   identity cr12 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr11 cr4 lr4 ;;
    MULT.RC.VE          lr4 lr8 0 lr4 cr12 ;;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE   identity cr12 ;;
    STR_POST_AAQ_REG    lr11 cr4 ;;
    LDR_MULT_REG        r0 lr11 cr4 ;;
    LDR_CYCLIC_MULT_REG lr2 cr3 lr4 ;;
    MULT.RC.VV          lr4 r0 0 lr4 cr12 ;;
    ACC.ADD.FIRST ;;
    ADD                 lr9 lr2 lr14 ;;
    LDR_CYCLIC_MULT_REG lr9 cr3 lr4 ;;
    MULT.RC.VE          lr4 lr5 0 lr4 cr12 ;;
    ACC.ADD ;;
    ACTIVATE.QUANTIZE   identity cr12 ;;
    STR_POST_AAQ_REG    lr3 cr2 ;;

    BREAK ;;

    ADD                 lr0 lr0 cr5 ;;
    BLT                 lr0 lr1 token_loop ;;

end:
    BKPT ;;
