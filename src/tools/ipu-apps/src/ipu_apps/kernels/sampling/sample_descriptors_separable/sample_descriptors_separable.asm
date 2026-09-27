{#- Stock-coordinate dense interpolation, separable FP32, HWC with C=256.

    For each coarse row i:
      T[a,j,c] = sum_dy Ky[i,dy,a] D[i+dy-1,j,c]
      O[8i+a,8j+b,c] = sum_dx Kx[j,dx,b] T[a,j+dx-1,c].
    The coordinates are identical to the direct stock kernel. FP32 operation
    grouping differs: Y interpolation is rounded before X interpolation.
    No L2 normalization. Custom nonseparable 3x3 filters use the direct kernel.

    MEMORY (all addresses are 128-FP32/512-byte XMEM rows):
      Input: (H+2,W+2,256), one zero halo cell on every side.
      Ky: H rows, Kx: W rows. Each packs [tap,phase] in lanes 0..23;
          remaining lanes are zero. The full-size tables occupy 70 KiB.
      Scratch T: (8,W+2,256); only interior columns are written. Its zero
          horizontal halos persist across coarse rows. Full size: 656 KiB.
      Output: (8*band_height,8*W,256), contiguous HWC.

    CR0=0 CR1=1, hardware constants
    CR2=input base at band's top halo row; CR3=output base
    CR4=Ky base at first band cell; CR5=Kx base; CR6=scratch base
    CR7=2*(W+2), padded spatial row stride for input AND scratch
    CR8=W; CR9=band coarse height; CR10=8; CR11=2 channel tiles
    CR12=128 cyclic slot step; CR13=2*W; CR15=128-lane FP32 dstructure

    LR0=zero/slot0; LR1=coarse row/Ky offset; LR2=input row's first real tile
    LR3=phase a; LR4=vertical tile or horizontal coarse column
    LR5=scratch store address or horizontal left-neighbor address
    LR6=walking tap address; LR7=weight scalar index; LR8=output cell origin
    LR9=output store address; LR10=phase b; LR11=channel half
    LR13=cyclic slot1 (128); LR14=cyclic slot2 (256).

    Vertical pass streams three input rows while a single Ky row stays in R0.
    Horizontal pass keeps three neighbor vectors resident in R_CYCLIC across
    all eight b phases; one Kx row is loaded per coarse column, not per tap.
    MULT data reads use snapshots, LR indexes are live; BLT reads snapshot LRs.
-#}
    SET lr0 cr0 ; SET lr1 cr0 ; SET lr2 cr11 ;;
    SET lr8 cr0 ; SET lr13 cr12 ;;
    ADD lr14 lr13 lr13 ;;
coarse_row_loop:
    LDR_MULT_REG r0 lr1 cr4 ; SET lr3 cr0 ; SET lr5 cr11 ;;
vertical_phase_loop:
    SET lr4 cr0 ;;
vertical_tile_loop:
    ADD lr6 lr2 lr4 ; ADD lr7 lr3 cr0 ;
    LDR_CYCLIC_MULT_REG lr6 cr2 lr0 ;;
    ADD lr6 lr6 cr7 ; LDR_CYCLIC_MULT_REG lr6 cr2 lr0 ;
    MULT.RC.VE lr0 lr7 0 lr0 cr15 ; ACC.ADD.FIRST ;;
    ADD lr6 lr6 cr7 ; ADD lr7 lr7 cr10 ;
    LDR_CYCLIC_MULT_REG lr6 cr2 lr0 ;
    MULT.RC.VE lr0 lr7 0 lr0 cr15 ; ACC.ADD ;;
    ADD lr7 lr7 cr10 ; ADD lr4 lr4 cr1 ;
    MULT.RC.VE lr0 lr7 0 lr0 cr15 ; ACC.ADD ;
    ACTIVATE.QUANTIZE identity cr15 ; STR_POST_AAQ_REG lr5 cr6 ;;
    ADD lr5 lr5 cr1 ; BLT lr4 cr13 vertical_tile_loop ;;
    ADD lr3 lr3 cr1 ; INC lr5 4 ;;
    BLT lr3 cr10 vertical_phase_loop ;;

    SET lr3 cr0 ; SET lr5 cr0 ;;
horizontal_row_loop:
    SET lr4 cr0 ;;
horizontal_cell_loop:
    LDR_MULT_REG r0 lr4 cr5 ; SET lr11 cr0 ;;
horizontal_tile_loop:
    ADD lr6 lr5 lr11 ; ADD lr9 lr8 lr11 ; SET lr10 cr0 ;;
    LDR_CYCLIC_MULT_REG lr6 cr6 lr0 ;;
    ADD lr6 lr6 cr11 ; LDR_CYCLIC_MULT_REG lr6 cr6 lr13 ;;
    ADD lr6 lr6 cr11 ; LDR_CYCLIC_MULT_REG lr6 cr6 lr14 ;;
horizontal_phase_loop:
    ADD lr7 lr10 cr0 ;
    MULT.RC.VE lr0 lr7 0 lr0 cr15 ; ACC.ADD.FIRST ;;
    ADD lr7 lr7 cr10 ;
    MULT.RC.VE lr13 lr7 0 lr0 cr15 ; ACC.ADD ;;
    ADD lr7 lr7 cr10 ; ADD lr10 lr10 cr1 ;
    MULT.RC.VE lr14 lr7 0 lr0 cr15 ; ACC.ADD ;
    ACTIVATE.QUANTIZE identity cr15 ; STR_POST_AAQ_REG lr9 cr3 ;;
    ADD lr9 lr9 cr11 ; BLT lr10 cr10 horizontal_phase_loop ;;
    ADD lr11 lr11 cr1 ;;
    BLT lr11 cr11 horizontal_tile_loop ;;
    ADD lr4 lr4 cr1 ; ADD lr5 lr5 cr11 ; INC lr8 16 ;;
    BLT lr4 cr8 horizontal_cell_loop ;;
    ADD lr3 lr3 cr1 ; INC lr5 4 ;;
    BLT lr3 cr10 horizontal_row_loop ;;
    ADD lr1 lr1 cr1 ; ADD lr2 lr2 cr7 ;;
    BLT lr1 cr9 coarse_row_loop ;;
    BKPT ;;
