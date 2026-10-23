# insn_bge_s_nx8: BGE.S in NX8
BGE.S CR7 LR1 skip ;;
INC LR5 1 ;;
skip:
BGE.S LR1 CR7 end ;;
INC LR5 2 ;;
end:
HALT ;;
