// Divergence class: signed hex, binary and octal literals.
// The assembler reads a number with Python's int(value, 0), which takes a sign
// before a 0x / 0b / 0o prefix and single underscores between digits; a JS
// Number() returned NaN for "-0x10". The verdict comes from the assembler
// (test/verdicts.js, check_verdicts.py), not from this comment.
    ADDBI lrd0 -0x10 ;;
    ADDBI lrd2 +0b101 ;;
    ADDBI lrd4 -0o17 ;;
    ADDBI lrd6 0x_7f ;;
    BKPT ;;
