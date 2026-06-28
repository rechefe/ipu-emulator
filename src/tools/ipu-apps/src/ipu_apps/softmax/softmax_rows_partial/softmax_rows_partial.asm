{#- ==========================================================================
    softmax_rows_partial.asm -- packed row-softmax for N<=128, P=128/ps rows/chunk

    P logical rows share one 128-lane chunk. C_VEC=log2(e) resident in R0
    (broadcast); x-chunk in r_cyclic. Per partition p, MULT.RC.VV rc_idx=p*ps*4
    reads r_cyclic[p*ps + i]*R0[i] = c*x[p][i] into mult_res lanes 0..N-1; masked
    AGG/ACTIVATE (CR15.valid_elements=N) act on those N lanes only.

    Addressing: cyclic/mult loads take (offset_lr, base_cr, index_lr); base CRs
    are fixed, the chunk byte offset walks in an LR. The partition slide rc_idx is
    a SEPARATE intra-register offset (0, ps*4, ...), not an XMEM address.

    Layout: input/output PACKED; numerators UNPACKED (1 chunk/row); maxvec=c*xmax
    and rvec=1/sum are 128-lane scalar vectors (slot per row).

    CR map (CR0=0, CR1=1 read-only):
      CR2=OUT_BASE  CR3=CVEC  CR4=NUM_BASE  CR5=MAXVEC  CR6=RVEC
      CR7=512 chunk stride  CR8=1  CR9=padded_rows  CR10=INPUT_BASE
      CR11=0xFF(-1.0)  CR12=ps*4 partition stride  CR13=num_chunks  CR14=P
========================================================================== -#}

{%- set lr_slide  = "lr0" -%}  {#- intra-register partition slide (0, ps*4, ...) -#}
{%- set lr_row    = "lr1" -%}  {#- global logical-row index = AGG dest slot -#}
{%- set lr_cyc    = "lr2" -%}  {#- constant 0 (cyclic index) -#}
{%- set lr_chunk  = "lr3" -%}  {#- chunk counter -#}
{%- set lr_p      = "lr4" -%}  {#- partition counter (0..P-1) -#}
{%- set lr_coff   = "lr5" -%}  {#- chunk byte offset (chunk * 512) -#}
{%- set lr_num    = "lr6" -%}  {#- numerator chunk byte offset (per row) -#}
{%- set lr_maxid  = "lr7" -%}  {#- R1 byte index for maxvec[row] select (128 + row) -#}
{%- set lr_rslide = "lr8" -%}  {#- Pass-4 repack slide = (512 - p*ps*4) mod 512 -#}
{%- set lr_c512   = "lr9" -%}  {#- holds 512 (for SUB src_a; SUB needs LR src_a) -#}

{#- ===================================================================== -#}
{#- PASS 1 -- maxvec[row] = c*xmax over each partition's N lanes.          -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}} cr0 ;;
    LDR_MULT_REG r0 {{lr_cyc}} cr3 ;;          {#- R0 = C_VEC (broadcast) -#}
    SET {{lr_row}} cr0 ;
    SET {{lr_chunk}} cr0 ;
    SET {{lr_coff}} cr0 ;;

p1_chunk:
    LDR_CYCLIC_MULT_REG {{lr_coff}} cr10 {{lr_cyc}} ;;   {#- r_cyclic = x chunk -#}
    SET {{lr_p}} cr0 ;
    SET {{lr_slide}} cr0 ;;

p1_part:
    MULT.RC.VV    {{lr_slide}} r0 0 {{lr_cyc}} ;          {#- c*x[p] at lanes 0..N-1 -#}
    AGG.MAX.FIRST {{lr_row}} 0 ;;                          {#- masked -> maxvec[row] -#}
    ADD {{lr_slide}} {{lr_slide}} cr12 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;
    ADD {{lr_p}} {{lr_p}} cr1 ;;
    BLT {{lr_p}} cr14 p1_part ;;
    ADD {{lr_coff}} {{lr_coff}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    BLT {{lr_chunk}} cr13 p1_chunk ;;

    ACTIVATE identity 1 ;;
    STR_POST_AAQ_REG {{lr_cyc}} cr5 ;;                     {#- MAXVEC <- maxvec -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[row] = 2^(c*x[p] - maxvec[row]).  Unpacked, 1 chunk/row. -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}} cr0 ;;
    LDR_MULT_REG r0 {{lr_cyc}} cr3 ;;          {#- R0 = C_VEC -#}
    LDR_MULT_REG r1 {{lr_cyc}} cr5 ;;          {#- R1 = maxvec (elem row = maxvec[row]) -#}
    SET {{lr_row}} cr0 ;
    SET {{lr_chunk}} cr0 ;
    SET {{lr_coff}} cr0 ;;
    SET {{lr_num}} cr0 ;
    SET {{lr_maxid}} cr8 ;;                      {#- R1 byte index = 128 (CR8) + row -#}

p2_chunk:
    LDR_CYCLIC_MULT_REG {{lr_coff}} cr10 {{lr_cyc}} ;;
    SET {{lr_p}} cr0 ;
    SET {{lr_slide}} cr0 ;;

p2_part:
    MULT.RC.VV {{lr_slide}} r0 0 {{lr_cyc}} ;            {#- c*x[p] -#}
    ACC.FIRST ;;
    MULT.EE {{lr_maxid}} cr11 0 {{lr_cyc}} ;             {#- maxvec[row]*(-1) -#}
    ACC ;;
    ACTIVATE exp2 0 ;;                                    {#- masked -> num[row] lanes 0..N-1 -#}
    STR_POST_AAQ_REG {{lr_num}} cr4 ;;                    {#- NUM[row] (unpacked) -#}
    ADD {{lr_slide}} {{lr_slide}} cr12 ;
    ADD {{lr_num}} {{lr_num}} cr7 ;
    ADD {{lr_maxid}} {{lr_maxid}} cr1 ;;
    ADD {{lr_row}} {{lr_row}} cr1 ;
    ADD {{lr_p}} {{lr_p}} cr1 ;;
    BLT {{lr_p}} cr14 p2_part ;;
    ADD {{lr_coff}} {{lr_coff}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    BLT {{lr_chunk}} cr13 p2_chunk ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[row] = SUM num[row]; rvec = 1/sumvec.  Per row.       -#}
{#- ===================================================================== -#}
    SET {{lr_num}} cr0 ;
    SET {{lr_row}} cr0 ;
    SET {{lr_cyc}} cr0 ;;

p3_row:
    LDR_CYCLIC_MULT_REG {{lr_num}} cr4 {{lr_cyc}} ;;      {#- r_cyclic = num[row] (snapshot: visible NEXT cycle) -#}
    MULT.RC.VE    {{lr_cyc}} cr1 0 {{lr_cyc}} ;           {#- *1.0 -#}
    AGG.SUM.FIRST {{lr_row}} 0 ;;                          {#- masked -> sumvec[row] -#}
    ADD {{lr_num}} {{lr_num}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    BLT {{lr_row}} cr9 p3_row ;;

    ACTIVATE reciprocal 1 ;;
    STR_POST_AAQ_REG {{lr_cyc}} cr6 ;;                     {#- RVEC <- 1/sum -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[row] = num[row]*rvec[row], RE-PACKED to chunk lanes.     -#}
{#-   per chunk: per partition p: r_cyclic=num[row]; MULT.RC.VE with        -#}
{#-   rc_idx = lr_rslide = (128 - p*ps)*4 mod 512 places product at lanes   -#}
{#-   p*ps; ACC accumulates into chunk r_acc. Full-width ACTIVATE drains.   -#}
{#-   lr_rslide: p=0 -> 0; p>=1 -> 512 - p*ps*4 (descending by ps*4).       -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}} cr0 ;;
    LDR_MULT_REG r0 {{lr_cyc}} cr6 ;;          {#- R0 = rvec (elem row = rvec[row]) -#}
    SET {{lr_row}} cr0 ;
    SET {{lr_chunk}} cr0 ;
    SET {{lr_coff}} cr0 ;;                      {#- output chunk offset -#}
    SET {{lr_num}} cr0 ;
    SET {{lr_c512}} cr7 ;;                       {#- lr_c512 = 512 (SUB src_a) -#}

{#- lr_slide ascends 0, ps*4, ... ; reverse slide rslide = (512 - lr_slide) % 512 -#}
{#- places num (r_cyclic lanes 0..N-1) at mult_res lanes p*ps. p=0: 512%512=0. -#}
p4_chunk:
    {#- p=0: ACC.FIRST clears the chunk r_acc -#}
    LDR_CYCLIC_MULT_REG {{lr_num}} cr4 {{lr_cyc}} ;
    SET {{lr_slide}} cr0 ;;
    SUB {{lr_rslide}} {{lr_c512}} {{lr_slide}} ;;                  {#- rslide = 512 - 0 = 512 (->0) -#}
    MULT.RC.VE {{lr_rslide}} {{lr_row}} 0 {{lr_cyc}} ;
    ACC.FIRST ;;
    SET {{lr_p}} cr1 ;
    ADD {{lr_num}} {{lr_num}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    ADD {{lr_slide}} {{lr_slide}} cr12 ;;                  {#- p=1 ascending slide = ps*4 -#}
    BGE {{lr_p}} cr14 p4_drain ;;                          {#- guard: P=1 -> skip p4_part (no p>=1) -#}

p4_part:
    LDR_CYCLIC_MULT_REG {{lr_num}} cr4 {{lr_cyc}} ;       {#- r_cyclic = num[row] -#}
    SUB {{lr_rslide}} {{lr_c512}} {{lr_slide}} ;;                  {#- rslide = 512 - p*ps*4 -#}
    {#- MULT reads lr_row LIVE; keep its increment OUT of this word (LR runs before MULT). -#}
    MULT.RC.VE {{lr_rslide}} {{lr_row}} 0 {{lr_cyc}} ;    {#- num*rvec[row] placed at p*ps -#}
    ACC ;;
    ADD {{lr_slide}} {{lr_slide}} cr12 ;
    ADD {{lr_num}} {{lr_num}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    ADD {{lr_p}} {{lr_p}} cr1 ;;
    BLT {{lr_p}} cr14 p4_part ;;

p4_drain:
    ACTIVATE identity 1 ;;
    STR_POST_AAQ_REG {{lr_coff}} cr2 ;;                    {#- packed output chunk -#}
    ADD {{lr_coff}} {{lr_coff}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    BLT {{lr_chunk}} cr13 p4_chunk ;;

end:
    BKPT ;;
