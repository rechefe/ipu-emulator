    SET                 lr0 0 ;;
    SET                 lr1 1280 ;;
    SET                 lr2 0 ;;

input_loop:
    RESET_ACC;;

    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr4 -128;;
    SET                 lr5 -1;;
    SET                 lr6 127;;
    SET                 lr15 0;;


element_loop:
    LDR_CYCLIC_MULT_REG lr4 cr1 lr15;
    ADD                 lr4 lr4 cr3;
    ADD                 lr5 lr5 cr4;
    MULT.VE.CYCLIC      lr15 0 lr15 lr5;
    ACC;
    BLT                 lr5 lr6 element_loop;;

    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 cr5;
    ADD                 lr0 lr0 cr3;;

    BREAK;;

    BLT                 lr0 lr1 input_loop;;

end:
    BKPT;;
