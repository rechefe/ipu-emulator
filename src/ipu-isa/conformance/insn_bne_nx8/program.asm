# insn_bne_nx8: BNE in NX8
BNE LR0 CR0 skip ;;
INC LR5 1 ;;
skip:
BNE LR0 LR1 end ;;
INC LR5 2 ;;
end:
HALT ;;
