{#- ==========================================================================
    maxpool2d_stride2_tail.asm -- compact-input 2x2 stride-2 FP32 max-pool

    This variant is selected when the final output XMEM row needs only one
    input XMEM row. CR12 is the number of complete output XMEM rows before that
    final row. Each matrix row occupies exactly ceil(W/128) XMEM rows.

    The +1 taps read one unused position past each loaded R_CYCLIC slot. That
    position is always removed by ACC.STRIDE, so no following input XMEM row is
    loaded and no guard XMEM row is required.

    CR2=input base, CR3=output base, CR4=scratch base, CR5=input row stride,
    CR6=2*input row stride, CR8=output height, CR9=channels, CR10=128,
    CR11=input plane stride, CR12=complete output XMEM rows before the tail.
========================================================================== -#}

{%- set lr_zero  = "lr0"  -%}
{%- set lr_c     = "lr1"  -%}
{%- set lr_out   = "lr2"  -%}
{%- set lr_cbase = "lr3"  -%}
{%- set lr_oy    = "lr4"  -%}
{%- set lr_rowb  = "lr5"  -%}
{%- set lr_ot    = "lr6"  -%}
{%- set lr_addr  = "lr7"  -%}
{%- set lr_s1    = "lr8"  -%}
{%- set lr_s2    = "lr9"  -%}
{%- set lr_baddr = "lr10" -%}
{%- set lr_one   = "lr11" -%}
{%- set lr_s2p1  = "lr12" -%}
{%- set lr_half  = "lr13" -%}
{%- set lr_itile = "lr14" -%}

    SET {{lr_zero}} cr0 ;
    SET {{lr_c}}    cr0 ;
    SET {{lr_out}}  cr0 ;;
    SET {{lr_cbase}} cr0 ;
    SET {{lr_s1}}    cr10 ;
    SET {{lr_one}}   cr1 ;;
    ADD {{lr_s2}} {{lr_s1}} {{lr_s1}} ;;
    ADD {{lr_s2p1}} {{lr_s2}} cr1 ;
    ADD {{lr_half}} {{lr_one}} cr1 ;;

chan_loop:
    SET {{lr_oy}} cr0 ;
    ADD {{lr_rowb}} {{lr_cbase}} cr0 ;;

row_loop:
    SET {{lr_ot}} cr0 ;
    ADD {{lr_itile}} {{lr_rowb}} cr0 ;;
    BGE {{lr_ot}} cr12 tail_tile ;;

full_tile_loop:
{#- First 64 output positions: input XMEM row 2*ot. -#}
    ADD {{lr_addr}} {{lr_itile}} cr0 ;
    ADD {{lr_baddr}} {{lr_itile}} cr5 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;
    LDR_CYCLIC_MULT_REG {{lr_baddr}} cr2 {{lr_s2}} ;;
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX.FIRST ;;
    MULT.RC.VE {{lr_one}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2p1}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_zero}} cr4 ;;

{#- Second 64 output positions: input XMEM row 2*ot+1. -#}
    ADD {{lr_addr}} {{lr_itile}} cr1 ;
    ADD {{lr_baddr}} {{lr_itile}} cr5 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;
    INC {{lr_baddr}} 1 ;
    LDR_CYCLIC_MULT_REG {{lr_baddr}} cr2 {{lr_s2}} ;;
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX.FIRST ;;
    MULT.RC.VE {{lr_one}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2p1}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_one}} cr4 ;;

    LDR_CYCLIC_MULT_REG {{lr_zero}} cr4 {{lr_zero}} ;;
    LDR_CYCLIC_MULT_REG {{lr_one}} cr4 {{lr_s1}} ;;
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ;
    ACC.STRIDE 64 on off {{lr_zero}} ;;
    MULT.RC.VE {{lr_s1}} cr1 0 {{lr_zero}} cr15 ;
    ACC.STRIDE 64 on off {{lr_half}} ;;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;
    ADD {{lr_out}} {{lr_out}} cr1 ;
    ADD {{lr_ot}} {{lr_ot}} cr1 ;
    INC {{lr_itile}} 2 ;;
    BLT {{lr_ot}} cr12 full_tile_loop ;;

tail_tile:
{#- Final <=64 output positions: only input XMEM row 2*CR12 exists. -#}
    ADD {{lr_addr}} {{lr_itile}} cr0 ;
    ADD {{lr_baddr}} {{lr_itile}} cr5 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;
    LDR_CYCLIC_MULT_REG {{lr_baddr}} cr2 {{lr_s2}} ;;
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX.FIRST ;;
    MULT.RC.VE {{lr_one}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2p1}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_zero}} cr4 ;;

    LDR_CYCLIC_MULT_REG {{lr_zero}} cr4 {{lr_zero}} ;;
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ;
    ACC.STRIDE 64 on off {{lr_zero}} ;;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_out}} cr3 ;;
    ADD {{lr_out}} {{lr_out}} cr1 ;;

    ADD {{lr_rowb}} {{lr_rowb}} cr6 ;
    ADD {{lr_oy}} {{lr_oy}} cr1 ;;
    BLT {{lr_oy}} cr8 row_loop ;;

    ADD {{lr_cbase}} {{lr_cbase}} cr11 ;
    ADD {{lr_c}} {{lr_c}} cr1 ;;
    BLT {{lr_c}} cr9 chan_loop ;;

end:
    BKPT ;;
