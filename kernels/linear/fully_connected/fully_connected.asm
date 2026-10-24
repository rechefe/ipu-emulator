{#- ==========================================================================
    fully_connected.asm -- a fully connected layer: y[s, :] = w @ x[s, :]

    10 samples of 128 inputs, 64 outputs; the shapes are fixed by the loops
    (kernel.toml's samples, in_features, out_features). It runs in NX8
    (ADR 0007), in each variant the case names (E4, E5, dynamic): the MULT
    expands x and w with the QF half cr.qf.L and each row's metadata, R_ACC
    accumulates in FP32, and the store quantizes each sample's outputs back to
    that QF.

    Data layout (rows; the regions sit at the old harness's byte addresses,
    0x0, 0x20000 and 0x40000):
      cr.x   x, one row per sample (128 inputs)
      cr.w   w transposed: row i holds input i's weight for each of the 64
             outputs, lanes 64..127 zero (kernel.toml's pack = "transposed")
      cr.y   y, one row per sample (cr.y_stride = 1): ACTIVATE.QUANTIZE
             identity stores 128 elements and the row's metadata. Only the
             first 64 lanes are outputs; the zero lanes 64..127 do not move
             the row's scale. The 64 outputs are padded to 128 lanes and no
             narrower width is declared, so every multiply counts all 128
             lanes.

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
    read-only; CR15.L holds the default DS (ds(128): every lane); cr.qf.L
    is the case's QF (kernel.toml's case_qf).
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

    MULT.RC.VE          LR15 LR5 0 LR15 CR15.L {{ cr.qf }}.L;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC LR14 {{ cr.w }} LR15;
    BNE                 LR5 LR6 element_loop_pre;;
    B                   after_element_loop;;

element_loop_pre:
    ADD                 LR14 LR14 {{ cr.row_step }};
    ADD                 LR5 LR5 {{ cr.element_step }};;             {#- LR14 -> next load offset; LR5 -> element just loaded -#}

element_loop:
    MULT.RC.VE          LR15 LR5 0 LR15 CR15.L {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC LR14 {{ cr.w }} LR15;
    BNE                 LR5 LR6 element_loop_pre;;

after_element_loop:
    ACTIVATE.QUANTIZE identity CR15.L {{ cr.qf }}.L {{ cr.y }} LR7;;
    ADD                 LR7 LR7 {{ cr.y_stride }};
    ADD                 LR0 LR0 {{ cr.row_step }};;

    BREAK;;

    BLT.S                 LR0 LR1 input_loop;;

end:
    HALT;;
