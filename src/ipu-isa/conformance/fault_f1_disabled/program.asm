# fault_f1_disabled: F-1 disabled: the slot runs as NOP and the run goes on
# program.bin edits the assembled words: Give word 0's LR sub-slot 0 the first opcode no LR instruction has (F-1).
INC LR0 1 ;;
INC LR4 1 ;;
HALT ;;
