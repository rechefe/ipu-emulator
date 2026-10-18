# insn_beq_int8: BEQ in INT8
BEQ LR0 LR1 skip ;;
INC LR5 1 ;;
skip:
BEQ LR0 CR0 end ;;
INC LR5 2 ;;
end:
HALT ;;
