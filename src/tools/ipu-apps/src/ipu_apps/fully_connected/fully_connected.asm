    SET                 lr0 cr6 ;;
    SET                 lr1 cr7 ;;
    SET                 lr2 cr8 ;;

input_loop:
    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr4 cr9 ;;
    SET                 lr5 cr10 ;;
    SET                 lr6 cr11 ;;
    SET                 lr15 cr12 ;;

    LDR_CYCLIC_MULT_REG lr4 cr13 lr15;
    ADD                 lr4 lr4 cr3;
    ADD                 lr5 lr5 cr4;
    MULT.VE.CYCLIC      lr15 0 lr15 lr5;
    ACC.FIRST;
    BEQ                 lr5 lr6 done_acc;;

element_loop:
    LDR_CYCLIC_MULT_REG lr4 cr13 lr15;
    ADD                 lr4 lr4 cr3;
    ADD                 lr5 lr5 cr4;
    MULT.VE.CYCLIC      lr15 0 lr15 lr5;
    ACC;
    BNE                 lr5 lr6 element_loop;;

done_acc:
    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 cr5;
    ADD                 lr0 lr0 cr3;;

    BREAK;;

    BLT                 lr0 lr1 input_loop;;

end:
    BKPT;;
