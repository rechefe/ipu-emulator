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
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD.FIRST;;
    BNE                 lr5 lr6 element_loop;;
    B                   after_element_loop;;

element_loop:
    LDR_CYCLIC_MULT_REG lr4 cr13 lr15;
    ADD                 lr4 lr4 cr3;
    ADD                 lr5 lr5 cr4;
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD;;
    BNE                 lr5 lr6 element_loop;;

after_element_loop:
    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 cr5;
    ADD                 lr0 lr0 cr3;;

    BREAK;;

    BLT                 lr0 lr1 input_loop;;

end:
    BKPT;;
