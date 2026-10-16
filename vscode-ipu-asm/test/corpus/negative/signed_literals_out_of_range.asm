// Divergence class: signed hex, binary and octal literals, out of range. The
// value is read exactly (int(value, 0) in the assembler, BigInt in the editor),
// so both reject a literal past the operand's range however it is spelled.
    ADDBI lrd0 -0x1_0000_0000 ;;
    BKPT ;;
