// Edge cases for the TextMate agreement test.
//
// The six app kernels are all written in one house style, so they exercise a
// narrow slice of the grammar. Every construct below is legal and assembles,
// but is absent from those files — and each one corresponds to a bug that a
// hand-written grammar got wrong before this test existed.
//
// This file must keep assembling: it is built by //vscode-ipu-asm and any
// change that breaks it is a real grammar regression.

# The other comment form the grammar %ignores.

start:
    SET lr0 cr0 ;;

// A label may follow `;;` on the same line: compounds are separated by `;;`,
// not by newlines, so anchoring labels to start-of-line misses this one.
    BKPT;; mid_line: BKPT;;

// A label may be spelled like a mnemonic or a register. These are ordinary
// identifiers in label position; the keyword terminals must not claim them.
add:
    SET lr1 cr1 ;;
lr0:
    SET lr2 cr2 ;;

// Mnemonics are case-insensitive, and dotted names must not be split: a rule
// that matched ACC.ADD first would leave `.FIRST` dangling.
    acc.add.first ;;
    ACC.ADD ;;
    MULT.RC.VE lr15 lr5 0 lr15 cr15 ;
    acc.sub ;;

// Numbers in every base the assembler's int(value, 0) accepts, plus the signed
// form a relative branch target uses.
    ADDBI lrd0 255 ;;
    ADDBI lrd2 0x1f ;;
    ADDBI lrd4 0b1010 ;;
    ADDBI lrd6 0o17 ;;
    BEQ cr0 cr0 +1 ;;

// Identifiers that merely START with a mnemonic or register name must lex as
// one token, not as the keyword plus a remainder. Both the definition and the
// reference matter: the reference is an operand, which is where a keyword rule
// missing its word boundary would show up.
    BNE lr0 lr1 bne_target ;;
    BEQ lr0 lr1 lr0_scratch ;;

bne_target:
    BKPT ;;
lr0_scratch:
    BKPT ;;
