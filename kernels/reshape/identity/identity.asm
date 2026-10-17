{#- ==========================================================================
    identity.asm -- minimal FP32 wide-vector application example

    Copies an ROWS x 128 FP32 matrix from CR2 to CR3, one XMEM row at a time.

    CR0 and CR1 are read-only hardware constants. In FP32 wide-vector mode,
    CR1 supplies the scalar 1.0, so MULT.VE is an identity operation.

      CR2 = input base row
      CR3 = output base row
      CR4 = ROWS
      CR15.valid_elements = 128

    XMEM operands are row numbers. One FP32 wide-vector row is 512 bytes.
========================================================================== -#}

{%- set lr_zero = "lr0" -%}
{%- set lr_row  = "lr1" -%}

    SET {{lr_zero}} CR0 ;
    SET {{lr_row}} CR0 ;;

row_loop:
    LD.MULT R0 {{lr_row}} CR2 ;;
    MULT.VE {{lr_zero}} CR1 0 {{lr_zero}} CR15 ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15 ;
    ST.AAQ {{lr_row}} CR3 ;;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    BLT.S {{lr_row}} CR4 row_loop ;;
    HALT ;;
