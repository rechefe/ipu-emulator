    SET                 lr0 cr6 ;;
    SET                 lr1 cr7 ;;
    SET                 lr2 cr8 ;;

input_loop:
    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr5 cr10 ;;
    SET                 lr6 cr11 ;;
    SET                 lr15 cr12 ;;

    {#- MULT reads r_cyclic from the start-of-cycle snapshot (issue #157), so
        the chunk consumed by MULT.RC.VE must be loaded a cycle earlier than
        it's used: prime element 0's load here, then each loop body loads the
        NEXT element's chunk while multiplying the chunk loaded last cycle.
        lr5 (R0 scalar-select index, read LIVE by MULT) must equal the CURRENT
        element on the cycle MULT runs; lr14 (load offset, also read LIVE, by
        LDR) must already hold the NEXT element's offset on that same cycle.
        LR sub-slots run before LOAD/MULT within a word, so lr14's increment
        for element e+1 must land in the word BEFORE the load that consumes
        it -- it cannot share a word with that load (the load would see the
        already-bumped value and skip an element). lr5 shares a word with the
        MULT that consumes it, same as the original code, so MULT sees the
        element lr5 was just advanced to. -#}
    SET                 lr14 cr9 ;;                         {#- lr14 = -128 -#}
    ADD                 lr14 lr14 cr3 ;;                    {#- lr14 = element 0's offset (0) -#}
    LDR_CYCLIC_MULT_REG lr14 cr13 lr15 ;;                   {#- load row 0 (lr14 unchanged this word) -#}
    ADD                 lr14 lr14 cr3 ;
    ADD                 lr5  lr5  cr4 ;;                     {#- lr14 -> element 1's offset; lr5 -> 0 -#}

    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG lr14 cr13 lr15;
    BNE                 lr5 lr6 element_loop_pre;;
    B                   after_element_loop;;

element_loop_pre:
    ADD                 lr14 lr14 cr3;
    ADD                 lr5 lr5 cr4;;                        {#- lr14 -> next load offset; lr5 -> element just loaded -#}

element_loop:
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD;
    LDR_CYCLIC_MULT_REG lr14 cr13 lr15;
    BNE                 lr5 lr6 element_loop_pre;;

after_element_loop:
    STR_ACC_REG         lr7 cr2;;
    ADD                 lr7 lr7 cr5;
    ADD                 lr0 lr0 cr3;;

    BREAK;;

    BLT                 lr0 lr1 input_loop;;

end:
    BKPT;;
