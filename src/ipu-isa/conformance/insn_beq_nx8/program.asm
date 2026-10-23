# insn_beq_nx8: BEQ in NX8
BEQ LR0 LR1 skip ;;
INC LR5 1 ;;
skip:
BEQ LR0 CR0 end ;;
INC LR5 2 ;;
end:
HALT ;;
