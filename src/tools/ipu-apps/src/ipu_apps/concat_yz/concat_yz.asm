{#-
================================================================================
 Concat the y and z branches -- MambaVisionMixer
================================================================================

     y = torch.cat([y, z], dim=1)

 y is the output of the selective scan, z is the gated branch that skipped
 the scan; both are in CHANNEL VIEW, (B, half, L). Concatenating on dim=1
 stacks their channel lines, so in XMEM this is two row copies into one
 destination buffer:

     input   y  : half lines of L elements, line c at SRCY_BASE + c*LPC
     input   z  : half lines of L elements, line c at SRCZ_BASE + c*LPC
     output  yz : d_inner lines of L elements at DST_BASE

     rows [0, HALF_ROWS)            <- y
     rows [HALF_ROWS, 2*HALF_ROWS)  <- z

     LPC = ceil(L/128),  HALF_ROWS = half * LPC

 This is the exact inverse of split_xz, and the same caveat applies: if the
 allocator places y and z back to back in one table the concat is free. It
 is written out here so the cost is measurable and so the kernel chain can
 run end to end on independent buffers.

 The copy idiom is the same "multiply by one" documented in split_xz.asm:
 MULT.RC.VE's scalar operand is an LcrIdx, so naming CR1 (permanently 1)
 takes 1 as the scalar directly -- no constants row and no R0 load.

 Only structural difference from split_xz: here the *source* buffer changes
 while the destination pointer keeps counting, so LR4 is rewound and
 R_CYCLIC re-primed between the halves -- two words of hand-off, not one.

--------------------------------------------------------------------------------
 Register map
--------------------------------------------------------------------------------
 CR1  1 (locked)  -- the multiply-by-one scalar
 CR2  SRCY_BASE_ROW                          CR15  dstructure (valid_elements=128)
 CR3  DST_BASE_ROW
 CR4  SRCZ_BASE_ROW
 CR5  HALF_ROWS = half * ceil(L/128)

 LR4  running source row (reset between the two halves)
 LR5  loop bound (HALF_ROWS)      LR7   destination row (never reset)
 LR6  loop counter                LR15  constant 0
-#}

{#- ---------------------------------------------------------------- prologue #}
    SET                 lr15 cr0 ;
    SET                 lr4 cr0 ;
    SET                 lr5 cr5 ;;
    SET                 lr6 cr0 ;
    SET                 lr7 cr0 ;;

    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;              {#- prime with y row 0 -#}

{#- ------------------------------------ y -> destination rows [0, HALF_ROWS) #}
y_loop:
    MULT.RC.VE          lr15 cr1 0 lr15 cr15 ;
    ACC.ADD.FIRST ;
    ADD                 lr4 lr4 cr1 ;
    ADD                 lr6 lr6 cr1 ;;

    ACTIVATE.QUANTIZE   identity cr15 ;
    LDR_CYCLIC_MULT_REG lr4 cr2 lr15 ;;

    STR_POST_AAQ_REG    lr7 cr3 ;;

    ADD                 lr7 lr7 cr1 ;
    BLT                 lr6 lr5 y_loop ;;

{#- Switch source buffers: rewind the source pointer and prime R_CYCLIC with
    z row 0. LR7 keeps counting, so z lands directly after y. -#}
    SET                 lr4 cr0 ;
    SET                 lr6 cr0 ;;
    LDR_CYCLIC_MULT_REG lr4 cr4 lr15 ;;

{#- ----------------------- z -> destination rows [HALF_ROWS, 2*HALF_ROWS) #}
z_loop:
    MULT.RC.VE          lr15 cr1 0 lr15 cr15 ;
    ACC.ADD.FIRST ;
    ADD                 lr4 lr4 cr1 ;
    ADD                 lr6 lr6 cr1 ;;

    ACTIVATE.QUANTIZE   identity cr15 ;
    LDR_CYCLIC_MULT_REG lr4 cr4 lr15 ;;

    STR_POST_AAQ_REG    lr7 cr3 ;;

    ADD                 lr7 lr7 cr1 ;
    BLT                 lr6 lr5 z_loop ;;

end:
    BKPT ;;
