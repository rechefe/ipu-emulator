{#- ==========================================================================
    maxpool2d_stride2_tail.asm -- compact-input 2x2 stride-2 FP32 max-pool

    This variant is selected when the final output XMEM row needs only one
    input XMEM row. CR12 is the number of complete output XMEM rows before that
    final row. Each matrix row occupies exactly ceil(W/128) XMEM rows.

    The +1 taps read one unused position past each loaded R_CYCLIC slot. That
    position is always removed by ACC.STRIDE, so no following input XMEM row is
    loaded and no guard XMEM row is required.

    The taps, the scratch round trip and the ACC.STRIDE decimation are
    maxpool2d_stride2's (its header explains them); only the last output XMEM
    row of each output row differs. The layout is the same: channel-major,
    then spatial-row-major, ceil(W/128) XMEM rows per spatial row, real
    columns first and -FLT_MAX past W (../pack.py, halo_tiled with no halo);
    the output is ceil((W//2)/128) XMEM rows per output row; two scratch rows.

    SELECTION: exactly one of the two kernels takes a width. The final output
    XMEM row has one input XMEM row behind it when
    ceil(W/128) < 2 * ceil((W//2)/128), that is when 2 <= W mod 256 <= 128.
    kernel.toml's derived value tail_only refuses every other width, and
    maxpool2d_stride2's full_pairs refuses these.

    ABI (kernel.toml [abi]; CR0/CR1 are READ-ONLY hardware constants):
      CR2  = INPUT_BASE (rows)        CR3  = OUTPUT_BASE (rows)
      CR4  = SCRATCH_BASE (rows; 2 rows)
      CR5  = IN_ROW_STRIDE = ceil(W/128)
      CR6  = 2 * IN_ROW_STRIDE        CR7  = OUT_TILES_PER_ROW (not read)
      CR8  = OUT_HEIGHT = H // 2      CR9  = CHANNELS
      CR10 = 128                      CR11 = IN_PLANE_STRIDE = H * IN_ROW_STRIDE
      CR12 = complete output XMEM rows before the tail = OUT_TILES_PER_ROW - 1
      CR15.L = the default DS (ds(128)); CR14 = the QF
========================================================================== -#}

{%- set lr_zero  = "LR0"  -%}
{%- set lr_c     = "LR1"  -%}
{%- set lr_out   = "LR2"  -%}
{%- set lr_cbase = "LR3"  -%}
{%- set lr_oy    = "LR4"  -%}
{%- set lr_rowb  = "LR5"  -%}
{%- set lr_ot    = "LR6"  -%}
{%- set lr_addr  = "LR7"  -%}
{%- set lr_s1    = "LR8"  -%}
{%- set lr_s2    = "LR9"  -%}
{%- set lr_baddr = "LR10" -%}
{%- set lr_one   = "LR11" -%}
{%- set lr_s2p1  = "LR12" -%}
{%- set lr_half  = "LR13" -%}
{%- set lr_itile = "LR14" -%}

    SET {{lr_zero}} CR0 ;
    SET {{lr_c}}    CR0 ;
    SET {{lr_out}}  CR0 ;;
    SET {{lr_cbase}} CR0 ;
    SET {{lr_s1}}    CR10 ;
    SET {{lr_one}}   CR1 ;;
    ADD {{lr_s2}} {{lr_s1}} {{lr_s1}} ;;
    ADD {{lr_s2p1}} {{lr_s2}} CR1 ;
    ADD {{lr_half}} {{lr_one}} CR1 ;;

chan_loop:
    SET {{lr_oy}} CR0 ;
    ADD {{lr_rowb}} {{lr_cbase}} CR0 ;;

row_loop:
    SET {{lr_ot}} CR0 ;
    ADD {{lr_itile}} {{lr_rowb}} CR0 ;;
    BGE.S {{lr_ot}} CR12 tail_tile ;;

full_tile_loop:
{#- First 64 output positions: input XMEM row 2*ot. -#}
    ADD {{lr_addr}} {{lr_itile}} CR0 ;
    ADD {{lr_baddr}} {{lr_itile}} CR5 ;
    LD.MULT.CYCLIC {{lr_addr}} CR2 {{lr_zero}} ;;
    LD.MULT.CYCLIC {{lr_baddr}} CR2 {{lr_s2}} ;;
    MULT.RC.VE {{lr_zero}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX.FIRST ;;
    MULT.RC.VE {{lr_one}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2p1}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;
    ACTIVATE.EXACT identity CR15.L CR4 {{lr_zero}} ;;

{#- Second 64 output positions: input XMEM row 2*ot+1. -#}
    ADD {{lr_addr}} {{lr_itile}} CR1 ;
    ADD {{lr_baddr}} {{lr_itile}} CR5 ;
    LD.MULT.CYCLIC {{lr_addr}} CR2 {{lr_zero}} ;;
    INC {{lr_baddr}} 1 ;
    LD.MULT.CYCLIC {{lr_baddr}} CR2 {{lr_s2}} ;;
    MULT.RC.VE {{lr_zero}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX.FIRST ;;
    MULT.RC.VE {{lr_one}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2p1}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;
    ACTIVATE.EXACT identity CR15.L CR4 {{lr_one}} ;;

    LD.MULT.CYCLIC {{lr_zero}} CR4 {{lr_zero}} ;;
    LD.MULT.CYCLIC {{lr_one}} CR4 {{lr_s1}} ;;
    MULT.RC.VE {{lr_zero}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ;
    ACC.STRIDE 64 on off {{lr_zero}} ;;
    MULT.RC.VE {{lr_s1}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ;
    ACC.STRIDE 64 on off {{lr_half}} ;;
    ACTIVATE.EXACT identity CR15.L CR3 {{lr_out}} ;;
    ADD {{lr_out}} {{lr_out}} CR1 ;
    ADD {{lr_ot}} {{lr_ot}} CR1 ;
    INC {{lr_itile}} 2 ;;
    BLT.S {{lr_ot}} CR12 full_tile_loop ;;

tail_tile:
{#- Final <=64 output positions: only input XMEM row 2*CR12 exists. -#}
    ADD {{lr_addr}} {{lr_itile}} CR0 ;
    ADD {{lr_baddr}} {{lr_itile}} CR5 ;
    LD.MULT.CYCLIC {{lr_addr}} CR2 {{lr_zero}} ;;
    LD.MULT.CYCLIC {{lr_baddr}} CR2 {{lr_s2}} ;;
    MULT.RC.VE {{lr_zero}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX.FIRST ;;
    MULT.RC.VE {{lr_one}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;;
    MULT.RC.VE {{lr_s2p1}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;
    ACTIVATE.EXACT identity CR15.L CR4 {{lr_zero}} ;;

    LD.MULT.CYCLIC {{lr_zero}} CR4 {{lr_zero}} ;;
    MULT.RC.VE {{lr_zero}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ;
    ACC.STRIDE 64 on off {{lr_zero}} ;;
    ACTIVATE.EXACT identity CR15.L CR3 {{lr_out}} ;;
    ADD {{lr_out}} {{lr_out}} CR1 ;;

    ADD {{lr_rowb}} {{lr_rowb}} CR6 ;
    ADD {{lr_oy}} {{lr_oy}} CR1 ;;
    BLT.S {{lr_oy}} CR8 row_loop ;;

    ADD {{lr_cbase}} {{lr_cbase}} CR11 ;
    ADD {{lr_c}} {{lr_c}} CR1 ;;
    BLT.S {{lr_c}} CR9 chan_loop ;;

end:
    HALT ;;
