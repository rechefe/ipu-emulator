{#- Stock-coordinate dense interpolation, separable FP32, HWC with C=256.

    For each coarse row i:
      T[a,j,c] = sum_dy Ky[i,dy,a] D[i+dy-1,j,c]
      O[8i+a,8j+b,c] = sum_dx Kx[j,dx,b] T[a,j+dx-1,c].
    The coordinates are identical to the direct stock kernel. FP32 operation
    grouping differs: Y interpolation is rounded before X interpolation.
    No L2 normalization. Custom nonseparable 3x3 filters use the direct kernel.

    MEMORY (all addresses are 128-FP32/512-byte XMEM rows):
      Input: (H+2,W+2,256), one zero halo cell on every side
          (../pack.py, hwc_halo).
      Ky: H rows, Kx: W rows. Each packs [tap,phase] in lanes 0..23;
          remaining lanes are zero (../tables.py). The full-size tables
          occupy 70 KiB.
      Scratch T: (8,W+2,256); only interior columns are written. Its zero
          horizontal halos persist across coarse rows. Full size: 656 KiB.
      Output: (8*band_height,8*W,256), contiguous HWC.

    ABI (kernel.toml [abi]); CR0=0 CR1=1, hardware constants
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
    MULT data reads use snapshots, LR indexes are live; BLT.S reads snapshot LRs.
-#}
    SET LR0 CR0 ; SET LR1 CR0 ; SET LR2 CR11 ;;
    SET LR8 CR0 ; SET LR13 CR12 ;;
    ADD LR14 LR13 LR13 ;;
coarse_row_loop:
    LD.MULT R0 LR1 CR4 ; SET LR3 CR0 ; SET LR5 CR11 ;;
vertical_phase_loop:
    SET LR4 CR0 ;;
vertical_tile_loop:
    ADD LR6 LR2 LR4 ; ADD LR7 LR3 CR0 ;
    LD.MULT.CYCLIC LR6 CR2 LR0 ;;
    ADD LR6 LR6 CR7 ; LD.MULT.CYCLIC LR6 CR2 LR0 ;
    MULT.RC.VE LR0 LR7 0 LR0 CR15 ; ACC.ADD.FIRST ;;
    ADD LR6 LR6 CR7 ; ADD LR7 LR7 CR10 ;
    LD.MULT.CYCLIC LR6 CR2 LR0 ;
    MULT.RC.VE LR0 LR7 0 LR0 CR15 ; ACC.ADD ;;
    ADD LR7 LR7 CR10 ; ADD LR4 LR4 CR1 ;
    MULT.RC.VE LR0 LR7 0 LR0 CR15 ; ACC.ADD ;
    ACTIVATE.QUANTIZE identity CR15 ; ST.AAQ LR5 CR6 ;;
    ADD LR5 LR5 CR1 ; BLT.S LR4 CR13 vertical_tile_loop ;;
    ADD LR3 LR3 CR1 ; INC LR5 4 ;;
    BLT.S LR3 CR10 vertical_phase_loop ;;

    SET LR3 CR0 ; SET LR5 CR0 ;;
horizontal_row_loop:
    SET LR4 CR0 ;;
horizontal_cell_loop:
    LD.MULT R0 LR4 CR5 ; SET LR11 CR0 ;;
horizontal_tile_loop:
    ADD LR6 LR5 LR11 ; ADD LR9 LR8 LR11 ; SET LR10 CR0 ;;
    LD.MULT.CYCLIC LR6 CR6 LR0 ;;
    ADD LR6 LR6 CR11 ; LD.MULT.CYCLIC LR6 CR6 LR13 ;;
    ADD LR6 LR6 CR11 ; LD.MULT.CYCLIC LR6 CR6 LR14 ;;
horizontal_phase_loop:
    ADD LR7 LR10 CR0 ;
    MULT.RC.VE LR0 LR7 0 LR0 CR15 ; ACC.ADD.FIRST ;;
    ADD LR7 LR7 CR10 ;
    MULT.RC.VE LR13 LR7 0 LR0 CR15 ; ACC.ADD ;;
    ADD LR7 LR7 CR10 ; ADD LR10 LR10 CR1 ;
    MULT.RC.VE LR14 LR7 0 LR0 CR15 ; ACC.ADD ;
    ACTIVATE.QUANTIZE identity CR15 ; ST.AAQ LR9 CR3 ;;
    ADD LR9 LR9 CR11 ; BLT.S LR10 CR10 horizontal_phase_loop ;;
    ADD LR11 LR11 CR1 ;;
    BLT.S LR11 CR11 horizontal_tile_loop ;;
    ADD LR4 LR4 CR1 ; ADD LR5 LR5 CR11 ; INC LR8 16 ;;
    BLT.S LR4 CR8 horizontal_cell_loop ;;
    ADD LR3 LR3 CR1 ; INC LR5 4 ;;
    BLT.S LR3 CR10 horizontal_row_loop ;;
    ADD LR1 LR1 CR1 ; ADD LR2 LR2 CR7 ;;
    BLT.S LR1 CR9 coarse_row_loop ;;
    HALT ;;
