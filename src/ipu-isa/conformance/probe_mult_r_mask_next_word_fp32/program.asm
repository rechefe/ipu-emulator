# probe_mult_r_mask_next_word_fp32: timing probe mult_r_mask_next_word in FP32
LD.MASK LR0 CR2 ;;
MULT.RC.VS LR0 0 LR0 CR15 ;;
HALT ;;
