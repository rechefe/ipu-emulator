{#- ==========================================================================
    fully_connected.asm -- a fully connected layer: y[s, :] = w @ x[s, :]

    10 samples of 128 inputs, 64 outputs; the shapes are fixed by the loops
    (kernel.toml's samples, in_features, out_features). The same program runs
    in INT8, INT32 (the emulator-only wide-integer format: INT8 data in 32-bit
    elements) and E4/E5; only the CR values differ, because a
    row is 128 bytes in an 8-bit format and 512 bytes in INT32.

    Data layout (rows; the regions sit at the old harness's byte addresses,
    0x0, 0x20000 and 0x40000, in every format):
      cr.x   x, one row per sample (128 inputs)
      cr.w   w transposed: row i holds input i's weight for each of the 64
             outputs, lanes 64..127 zero (kernel.toml's pack = "transposed")
      cr.y   y, one accumulator vector per sample. ST.ACC writes the whole
             512-byte accumulator whatever the format, so samples are
             cr.y_stride rows apart: 4 rows in an 8-bit format, 1 in INT32
             (one INT32 output vector alone takes 2 narrow rows; a 2-row
             stride would make the stores overlap). Only the first 64
             accumulator lanes are outputs. The 64 outputs are padded to 128
             lanes and no narrower width is declared, so every multiply
             counts all 128 lanes.

    Loops: for each sample (LR0 = 0 .. samples - 1, LR1 = samples), load the
    sample into R0, then for each input element e (LR5) multiply weight row
    e (in R_CYCLIC) by element e of R0 and accumulate (MULT.RC.VE), then
    store the accumulator at LR7 and step LR7 by cr.y_stride.

    Pipelining. MULT reads R_CYCLIC from the start-of-cycle snapshot, so the
    chunk MULT.RC.VE consumes must be loaded a cycle before it is used: the
    prologue loads element 0's row, then each loop word loads the NEXT
    element's row while multiplying the row loaded last cycle. LR5 (R0's
    scalar-select index, read live by MULT) must equal the CURRENT element in
    the cycle MULT runs; LR14 (the load offset, also read live, by the load)
    must already hold the NEXT element's offset in that same cycle. LR
    sub-slots run before LOAD and MULT within a word, so LR14's step for
    element e + 1 must be in the word BEFORE the load that uses it, never in
    the same word (the load would see the stepped value and skip an
    element). LR5 shares a word with the MULT that reads it, so MULT sees
    the element LR5 was just stepped to. Both start one step early
    (cr.weight_start = cr.element_start = -1): ADD runs before the load and
    MULT in a word, and the 32-bit wrap of the first ADD gives 0.

    Registers (kernel.toml's [abi]): cr.row_step = cr.element_step = 1,
    cr.first_sample = cr.lr2_start = cr.cyclic_index = 0, cr.samples = 10,
    cr.last_element = 127 (the element loop's BNE exit). CR1 = 1 is
    read-only; CR15 holds the default dstructure (128 valid elements).

    ST.ACC (the A16_STR_ACC_REG alias, once per sample) is emulator-only: a
    run warns DebugOnlyInstructionWarning.
========================================================================== -#}
    SET                 LR0 {{ cr.first_sample }} ;;
    SET                 LR1 {{ cr.samples }} ;;
    SET                 LR2 {{ cr.lr2_start }} ;;

input_loop:
    LD.MULT        R0 LR0 {{ cr.x }};;

    SET                 LR5 {{ cr.element_start }} ;;
    SET                 LR6 {{ cr.last_element }} ;;
    SET                 LR15 {{ cr.cyclic_index }} ;;

    SET                 LR14 {{ cr.weight_start }} ;;               {#- LR14 = -1 -#}
    ADD                 LR14 LR14 {{ cr.row_step }} ;;              {#- LR14 = element 0's offset (0) -#}
    LD.MULT.CYCLIC LR14 {{ cr.w }} LR15 ;;                          {#- load row 0 (LR14 unchanged this word) -#}
    ADD                 LR14 LR14 {{ cr.row_step }} ;
    ADD                 LR5  LR5  {{ cr.element_step }} ;;          {#- LR14 -> element 1's offset; LR5 -> 0 -#}

    MULT.RC.VE          LR15 LR5 0 LR15 CR15;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC LR14 {{ cr.w }} LR15;
    BNE                 LR5 LR6 element_loop_pre;;
    B                   after_element_loop;;

element_loop_pre:
    ADD                 LR14 LR14 {{ cr.row_step }};
    ADD                 LR5 LR5 {{ cr.element_step }};;             {#- LR14 -> next load offset; LR5 -> element just loaded -#}

element_loop:
    MULT.RC.VE          LR15 LR5 0 LR15 CR15;
    ACC.ADD;
    LD.MULT.CYCLIC LR14 {{ cr.w }} LR15;
    BNE                 LR5 LR6 element_loop_pre;;

after_element_loop:
    ST.ACC         LR7 {{ cr.y }};;
    ADD                 LR7 LR7 {{ cr.y_stride }};
    ADD                 LR0 LR0 {{ cr.row_step }};;

    BREAK;;

    BLT.S                 LR0 LR1 input_loop;;

end:
    HALT;;
