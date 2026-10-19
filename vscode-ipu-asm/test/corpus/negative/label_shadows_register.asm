// Divergence class: a label that shadows a register name. In label position
// lr0 is a label, in a register operand the register, in a Label operand the
// label again.
lr0:
    SET lr0 cr0 ;;
    BEQ lr0 lr1 lr0 ;;
    B lr0 ;;
