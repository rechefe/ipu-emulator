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
    incr                lr4 128;
    incr                lr5 1;
    mult.ve             lr15 lr15 lr15 lr5;
    acc;
    blt                 lr5 lr6 element_loop;;

    str_acc_reg         lr7 cr2;;
    incr                lr7 256;
    incr                lr0 128;;

    break;;

    blt                 lr0 lr1 input_loop;;

end:
    bkpt;;
