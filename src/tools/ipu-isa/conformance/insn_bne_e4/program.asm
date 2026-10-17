# insn_bne_e4: BNE in E4
BNE LR0 CR0 skip ;;
INC LR5 1 ;;
skip:
BNE LR0 LR1 end ;;
INC LR5 2 ;;
end:
HALT ;;
