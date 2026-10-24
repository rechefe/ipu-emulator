# insn_blt_s_nx8: BLT.S in NX8
BLT.S LR1 CR7 skip ;;
INC LR5 1 ;;
skip:
BLT.S CR7 LR1 end ;;
INC LR5 2 ;;
end:
HALT ;;
