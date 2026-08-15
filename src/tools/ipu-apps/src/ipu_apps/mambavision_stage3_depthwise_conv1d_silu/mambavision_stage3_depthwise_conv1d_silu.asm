# mambavision_stage3_depthwise_conv1d_silu.asm
#
# Depthwise Conv1D (kernel size 3, non-causal) + SiLU, wide-vector INT32 mode.
# Same kernel run once per branch (x, z) with different weight/input files.
#
#   x:      196 tokens x 160 channels  (token-major, contiguous per token,
#           padded with 1 zero row before + 1 zero row after -> 198 rows)
#   taps:   3 x 160  (tap_minus1, tap_zero, tap_plus1 -- one weight per
#           channel per tap position, stored back-to-back, 640B each)
#   y:      196 x 160 = SiLU(tap_minus1*x[t-1] + tap_zero*x[t] + tap_plus1*x[t+1])
#
# Channels tiled 128 + 32 (160 total). Lanes = channels (not tokens) --
# each channel gets its own tap weight via MULT.RC.VV (lane-wise), so this
# reads/writes token-major data directly, no transpose needed anywhere.
#
# Boundary handling: the input buffer itself has a zero-padded row before
# token 0 and after token 195, so reading "token t-1"/"token t+1" never
# needs a conditional branch -- it just lands on real zero data at the ends.
#
# CR map (all set in Python setup()):
#   cr0  = ZERO / INPUT_BASE   (both 0 -- CR0 hardware-reserved, permanently 0)
#   cr2  = OUTPUT_BASE
#   cr3  = TAPS_BASE           (tap_minus1 at +0, tap_zero at +640, tap_plus1 at +1280)
#   cr4  = ROW_BYTES           (640 = 160*4 bytes)
#   cr5  = LOOP_START          (640 -- first padded "center" row offset, = token 0)
#   cr6  = LOOP_END            (126080 = 640*197)
#   cr7  = TILE1_SUBOFFSET     (512 = 128*4 bytes)
#   cr8  = TAPZERO_SUBOFFSET   (640, offset from TAPS_BASE to tap_zero's own base)
#   cr9  = TAPPLUS1_SUBOFFSET  (1280, offset from TAPS_BASE to tap_plus1's own base)
#   cr10 = DSTRUCT_VALID32     (dstructure valid_elements=32, tile1)
#   cr15 = DSTRUCT_VALID128    (dstructure valid_elements=128, tile0)

    SET                 lr0 cr5 ;;      # lr0 = 640, padded "center" row offset (token 0)
    SET                 lr1 cr6 ;;      # lr1 = loop end
    SET                 lr2 cr0 ;;      # lr2 = 0 (zero, reused as LDR_CYCLIC index)

token_loop:
    SUB                 lr3 lr0 cr4 ;;  # lr3 = center - 640 = prev row offset (also = output row offset)
    ADD                 lr4 lr0 cr4 ;;  # lr4 = center + 640 = next row offset

    # ================= tile 0: channels 0-127, valid=128 =================
    SET                 lr5 cr0 ;;      # lr5 = 0 (tile_suboffset)

    ADD                 lr6 lr3 lr5 ;;
    LDR_CYCLIC_MULT_REG lr6 cr0 lr2 ;;  # R_CYCLIC = x[t-1, tile0]
    LDR_MULT_REG        r0 lr5 cr3 ;;   # R0 = tap_minus1[tile0]
    MULT.RC.VV          lr2 r0 0 lr2 cr15 ;;
    ACC.ADD.FIRST ;;

    ADD                 lr6 lr0 lr5 ;;
    LDR_CYCLIC_MULT_REG lr6 cr0 lr2 ;;  # R_CYCLIC = x[t, tile0]
    ADD                 lr7 lr5 cr8 ;;
    LDR_MULT_REG        r0 lr7 cr3 ;;   # R0 = tap_zero[tile0]
    MULT.RC.VV          lr2 r0 0 lr2 cr15 ;;
    ACC.ADD ;;

    ADD                 lr6 lr4 lr5 ;;
    LDR_CYCLIC_MULT_REG lr6 cr0 lr2 ;;  # R_CYCLIC = x[t+1, tile0]
    ADD                 lr7 lr5 cr9 ;;
    LDR_MULT_REG        r0 lr7 cr3 ;;   # R0 = tap_plus1[tile0]
    MULT.RC.VV          lr2 r0 0 lr2 cr15 ;;
    ACC.ADD ;;

    ACTIVATE.QUANTIZE   silu cr15 ;;
    ADD                 lr6 lr3 lr5 ;;
    STR_POST_AAQ_REG    lr6 cr2 ;;

    # ================= tile 1: channels 128-159, valid=32 =================
    SET                 lr5 cr7 ;;      # lr5 = 512 (tile_suboffset)

    ADD                 lr6 lr3 lr5 ;;
    LDR_CYCLIC_MULT_REG lr6 cr0 lr2 ;;  # R_CYCLIC = x[t-1, tile1]
    LDR_MULT_REG        r0 lr5 cr3 ;;   # R0 = tap_minus1[tile1]
    MULT.RC.VV          lr2 r0 0 lr2 cr10 ;;
    ACC.ADD.FIRST ;;

    ADD                 lr6 lr0 lr5 ;;
    LDR_CYCLIC_MULT_REG lr6 cr0 lr2 ;;  # R_CYCLIC = x[t, tile1]
    ADD                 lr7 lr5 cr8 ;;
    LDR_MULT_REG        r0 lr7 cr3 ;;   # R0 = tap_zero[tile1]
    MULT.RC.VV          lr2 r0 0 lr2 cr10 ;;
    ACC.ADD ;;

    ADD                 lr6 lr4 lr5 ;;
    LDR_CYCLIC_MULT_REG lr6 cr0 lr2 ;;  # R_CYCLIC = x[t+1, tile1]
    ADD                 lr7 lr5 cr9 ;;
    LDR_MULT_REG        r0 lr7 cr3 ;;   # R0 = tap_plus1[tile1]
    MULT.RC.VV          lr2 r0 0 lr2 cr10 ;;
    ACC.ADD ;;

    ACTIVATE.QUANTIZE   silu cr10 ;;
    ADD                 lr6 lr3 lr5 ;;
    STR_POST_AAQ_REG    lr6 cr2 ;;

    BREAK ;;

    ADD                 lr0 lr0 cr4 ;;
    BLT                 lr0 lr1 token_loop ;;

end:
    BKPT ;;
