    set                 lr0 0 ;;
    set                 lr1 1280 ;;
    set                 lr2 0 ;;

input_loop:
    reset_acc;;

    ldr_cyclic_mult_reg lr0 cr0 lr15;;

    set                 lr4 0;;
    set                 lr5 0;;
    set                 lr6 127;;

element_loop:
    ldr_mult_reg        mem_bypass lr4 cr1;
    incr                lr4 128;
    incr                lr5 1;
    mult.ev             mem_bypass lr15 lr15 lr15 lr5;
    acc;
    blt                 lr5 lr6 element_loop;;

    str_acc_reg         lr7 cr2;
    incr                lr7 256;
    incr                lr0 128;;

    blt                 lr0 lr1 input_loop;;

end:
    bkpt;;
