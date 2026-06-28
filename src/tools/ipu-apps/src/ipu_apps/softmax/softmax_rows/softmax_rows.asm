{#- ==========================================================================
    softmax_rows.asm -- numerically stable row-softmax, FP32 wide-vector mode

    softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)
    so that 2^(c*d) == e^d and the IPU's native exp2 activation applies directly.

    All scaling is on the FP32 vector path; CR scalars are integer-only in wide
    mode, so c=log2(e) is a resident vector (C_VEC) while the integer scalars
    1.0 (CR1) and -1.0 (CR11=0xFF) ride in CRs.

    ROW GROUPING (exact -- no padding):
      maxvec[r] / rvec[r] pack one scalar per row into a single 128-lane vector,
      so the chip can hold the per-row bookkeeping for at most 128 rows at once.
      All four passes therefore run once per GROUP of up to 128 rows. Each group
      is processed completely (Pass 1..4) before the next, because maxvec/rvec
      are single shared vectors that the next group overwrites.

      The group size is computed EXACTLY, not padded: at the top of each group
        rows_this_group = min(128, total_rows - rows_done)
      so the last (partial) group runs only as many rows as actually remain. A
      7-row input runs exactly 7 rows; a 130-row input runs 128 then 2. The four
      passes all branch against lr_bound (this group's exact size), not a fixed
      constant. The loop ends when rows_done reaches total_rows.

      lr_gbase = rows_done * 512 is the group's byte base into the input / num /
      output regions; each pass initialises its working offset from it, so row r
      (0..bound-1) of the current group lives at BASE + lr_gbase + r*512.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0  (zero source)       CR1  = 1   (-> 1.0 scalar, Pass 3 identity)
      CR2  = OUTPUT_BASE            CR3  = CVEC_ADDR        CR4  = NUM_BASE
      CR5  = MAXVEC_ADDR            CR6  = RVEC_ADDR        CR7  = ROW_BYTES (512)
      CR8  = 1 (row incr)           CR9  = GROUP_CAP (128)  CR10 = INPUT_BASE
      CR11 = 0xFF (-> -1.0 scalar)  CR12 = 128 (R1 byte base for maxvec select)
      CR13 = TOTAL_ROWS

    NOTE: input logits live at CR10; the literal 0 is CR0.

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    r_cyclic loads use index 0 in wide mode (full 512B load).
========================================================================== -#}

{%- set lr_off       = "lr0" -%}  {#- working input/num/out byte offset (steps by 512) -#}
{%- set lr_row       = "lr1" -%}  {#- per-group row index r = AGG dest slot / element select -#}
{%- set lr_cyc       = "lr2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_wr        = "lr3" -%}  {#- working write offset for the NUM region -#}
{%- set lr_max_idx   = "lr4" -%}  {#- byte index 128+r into R1 = maxvec[r] -#}
{%- set lr_done      = "lr5" -%}  {#- rows completed so far (group byte base / 512) -#}
{%- set lr_gbase     = "lr6" -%}  {#- group byte base = rows_done * 512 -#}
{%- set lr_bound     = "lr7" -%}  {#- this group's exact row count = min(128, remaining) -#}
{%- set lr_total     = "lr8" -%}  {#- total_rows (LR copy; SUB needs an LR src_a) -#}

    SET {{lr_done}}  cr0 ;
    SET {{lr_gbase}} cr0 ;
    SET {{lr_cyc}}   cr0 ;;
    SET {{lr_total}} cr13 ;;                             {#- lr_total = total_rows (for SUB src_a) -#}

group_loop:
{#- ---- exact group size: lr_bound = min(128, total_rows - rows_done) ------ -#}
    SUB {{lr_bound}} {{lr_total}} {{lr_done}} ;;         {#- bound = remaining = total - done (>=1) -#}
    BLT {{lr_bound}} cr9 group_size_set ;;               {#- if remaining < 128, keep it ... -#}
    SET {{lr_bound}} cr9 ;;                              {#- ... else cap bound = 128 -#}
group_size_set:

{#- ===================================================================== -#}
{#- PASS 1 -- maxvec[r] = max_j (c * x[r, j]).  No row written.            -#}
{#- ===================================================================== -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}} ;;   {#- r_cyclic = C_VEC -#}
    ADD {{lr_off}} {{lr_gbase}} cr0 ;                   {#- working offset = group base (SET takes CR only) -#}
    SET {{lr_row}} cr0 ;;

pass1_loop:
    LDR_MULT_REG  r0 {{lr_off}} cr10 ;;                 {#- R0 = x[r] (snapshot: visible NEXT cycle, issue #157) -#}
    MULT.RC.VV    {{lr_cyc}} r0 0 {{lr_cyc}} ;          {#- mult_res = c*x[r] -#}
    AGG.MAX.FIRST {{lr_row}} 1 ;;                       {#- r_acc[r] = max lane -#}
    ADD {{lr_off}} {{lr_off}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr8 ;;
    BLT {{lr_row}} {{lr_bound}} pass1_loop ;;

    ACTIVATE identity 1 ;;
    STR_POST_AAQ_REG {{lr_cyc}} cr5 ;;                  {#- MAXVEC_ADDR <- maxvec -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[r] = 2^(c*x[r] - maxvec[r]).  Writes NUM region.         -#}
{#-   k:   R0=x[r]; R1=maxvec row; mult_res=c*x[r]; ACC.FIRST              -#}
{#-   k+1: mult_res = maxvec[r] * (-1) broadcast; ACC -> r_acc - maxvec[r] -#}
{#-   k+2: ACTIVATE exp2 (reads r_acc snapshot); store num[r]              -#}
{#- ===================================================================== -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}} ;;   {#- r_cyclic = C_VEC -#}
    LDR_MULT_REG r1 {{lr_cyc}} cr5 ;;                   {#- R1 = maxvec row (resident; all rows) -#}
    ADD {{lr_off}} {{lr_gbase}} cr0 ;
    ADD {{lr_wr}}  {{lr_gbase}} cr0 ;
    SET {{lr_row}} cr0 ;;
    SET {{lr_max_idx}} cr12 ;;                          {#- lr_max_idx = 128 (R1[0] = maxvec[0]) -#}

pass2_loop:
    LDR_MULT_REG r0 {{lr_off}} cr10 ;;                   {#- R0 = x[r] (snapshot: visible NEXT cycle, issue #157) -#}
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} ;           {#- mult_res = c*x[r] -#}
    ACC.FIRST ;;                                         {#- r_acc = c*x[r] -#}

    MULT.EE {{lr_max_idx}} cr11 0 {{lr_cyc}} ;          {#- mult_res = maxvec[r]*(-1) -#}
    ACC ;;                                               {#- r_acc = c*x[r] - maxvec[r] -#}

    ACTIVATE exp2 1 ;;
    STR_POST_AAQ_REG {{lr_wr}} cr4 ;;                    {#- NUM[r] = 2^(...) -#}

    ADD {{lr_off}} {{lr_off}} cr7 ;
    ADD {{lr_wr}} {{lr_wr}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr8 ;;
    ADD {{lr_max_idx}} {{lr_max_idx}} cr8 ;;             {#- next maxvec element -#}
    BLT {{lr_row}} {{lr_bound}} pass2_loop ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[r] = SUM_j num[r,j]; rvec = 1/sumvec.  Staged.        -#}
{#-   r_cyclic = num[r]; MULT.RC.VE x CR1(=1.0); AGG.SUM.FIRST into r_acc[r]-#}
{#- ===================================================================== -#}
    ADD {{lr_off}} {{lr_gbase}} cr0 ;
    SET {{lr_row}} cr0 ;;

pass3_loop:
    LDR_CYCLIC_MULT_REG {{lr_off}} cr4 {{lr_cyc}} ;;    {#- r_cyclic = num[r] (snapshot: visible NEXT cycle) -#}
    MULT.RC.VE    {{lr_cyc}} cr1 0 {{lr_cyc}} ;         {#- mult_res = num[r]*1.0 -#}
    AGG.SUM.FIRST {{lr_row}} 1 ;;                        {#- r_acc[r] = SUM num[r] -#}
    ADD {{lr_off}} {{lr_off}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr8 ;;
    BLT {{lr_row}} {{lr_bound}} pass3_loop ;;

    ACTIVATE reciprocal 1 ;;
    STR_POST_AAQ_REG {{lr_cyc}} cr6 ;;                   {#- RVEC_ADDR <- 1/sum -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[r] = num[r] * rvec[r].  Writes OUTPUT region.            -#}
{#-   r_cyclic = num[r]; R0 = rvec row; broadcast rvec[r] via LR-element.  -#}
{#- ===================================================================== -#}
    LDR_MULT_REG r0 {{lr_cyc}} cr6 ;;                    {#- R0 = rvec row (resident; elem r = rvec[r]) -#}
    ADD {{lr_off}} {{lr_gbase}} cr0 ;
    SET {{lr_row}} cr0 ;;

pass4_loop:
    LDR_CYCLIC_MULT_REG {{lr_off}} cr4 {{lr_cyc}} ;;    {#- r_cyclic = num[r] (snapshot: visible NEXT cycle) -#}
    MULT.RC.VE   {{lr_cyc}} {{lr_row}} 0 {{lr_cyc}} ;   {#- mult_res = num[r]*rvec[r] -#}
    ACC.FIRST ;;
    ACTIVATE identity 1 ;;
    STR_POST_AAQ_REG {{lr_off}} cr2 ;;                   {#- OUT[r] = softmax row -#}

    ADD {{lr_off}} {{lr_off}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr8 ;;
    BLT {{lr_row}} {{lr_bound}} pass4_loop ;;

{#- ---- advance: rows_done += bound; loop while done < total -------------- -#}
{#- After Pass 4, lr_off = gbase + bound*512 (it stepped once per row), which -#}
{#- is exactly the next group's byte base -- no multiply needed (the LR slot  -#}
{#- has no multiply). Copy it into lr_gbase. -#}
    ADD {{lr_done}} {{lr_done}} {{lr_bound}} ;            {#- rows_done += this group -#}
    ADD {{lr_gbase}} {{lr_off}} cr0 ;;                   {#- next gbase = end of this group -#}
    BLT {{lr_done}} {{lr_total}} group_loop ;;

end:
    BKPT ;;
