{#
  Integration-test kernel: copies rows, like the identity kernel.
  CR2 = input row, CR3 = output row, CR4 = row count, CR15.L = DS, CR14.L = QF.
#}
{%- set lr_row = "lr1" -%}
{%- set lr_zero = "lr0" -%}

    SET {{lr_zero}} cr0 ;
    SET {{lr_row}} cr0 ;;

row_loop:
    LD.MULT r0 {{lr_row}} cr2 ;;
    MULT.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15.l cr14.l ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity cr15.l cr14.l cr3 {{lr_row}} ;;
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    BLT.S {{lr_row}} cr4 row_loop ;;
    HALT ;;
