# fault_f1_reserved_opcode: F-1: an undefined LR opcode
# program.bin edits the assembled words: Give word 0's LR sub-slot 0 the first opcode no LR instruction has (F-1).
INC LR0 1 ;;
HALT ;;
