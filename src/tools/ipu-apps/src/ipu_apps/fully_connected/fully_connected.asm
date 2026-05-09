    set                 lr0 0 ;;
    set                 lr1 1280 ;;
    set                 lr2 0 ;;

input_loop:
    reset_acc;;

    ldr_mult_reg        r0 lr0 cr0;;

    set                 lr4 -128;;
    set                 lr5 -1;;
    set                 lr6 127;;
    set                 lr15 0;;


element_loop:
    ldr_cyclic_mult_reg lr4 cr1 lr15;
    add                 lr4 lr4 cr3;
    add                 lr5 lr5 cr4;
    mult.ve.cyclic      lr15 0 lr15 lr5;
    acc;
    blt                 lr5 lr6 element_loop;;

    str_acc_reg         lr7 cr2;;
    add                 lr7 lr7 cr5;
    add                 lr0 lr0 cr3;;

    break;;

    blt                 lr0 lr1 input_loop;;

end:
    bkpt;;
