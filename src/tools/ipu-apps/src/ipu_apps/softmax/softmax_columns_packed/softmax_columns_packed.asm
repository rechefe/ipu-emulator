{#- ==========================================================================
    softmax_columns_packed.asm -- packed sub-128 column-softmax, FP32 wide mode

    softmax(x[r,col]) = 2^(c*(x[r,col]-cmax[col])) / SUM_r 2^(c*(x[r,col]-cmax[col]))
    c = log2(e), so 2^(c*d) == e^d and the IPU's native exp2 applies directly.

    ONE matrix, narrow rows (real width <= 64) packed rpv = 128/W_pad rows per
    128-lane vector (W_pad in {16,32,64}; lanes [g*W_pad : g*W_pad+W_pad) hold
    matrix row that maps to group g). Softmax is per column, reduced DOWN all
    rows -- and a column's members span BOTH lane-groups (within a vector) AND
    vectors. So each reduce has two stages:

      (1) DOWN-VECTOR partial: running ACC.MAX / ACC.ADD over num_vectors gives,
          per lane g*W+col, the reduce over the rows that landed in group g.
      (2) CROSS-GROUP fold: a cyclic-shift all-reduce (shift W, 2W, 4W ... < 128
          lanes, with ACC.MAX / ACC.ADD) folds the rpv groups together AND
          broadcasts the full per-column result back across every group's lanes
          (rpv is a power of two, so the cyclic all-reduce lands the same value
          in all rpv lanes of a column). After the fold, cmax / rvec are full
          128-lane vectors lane-aligned for the Pass 2/4 trips.

    The cyclic shift is the rc_idx byte offset of MULT.RC.* into r_cyclic (wraps
    at 512 B): rc_idx = shift_lanes*4 reads r_cyclic[i] = data[(i+shift)%128].
    The fold is a runtime loop (shift doubles until 512 B), so one code path
    serves rpv in {2,4,8}.

    Issue #157: MULT reads Ra/r_cyclic DATA from the snapshot (visible NEXT
    cycle), so loads sit one VLIW word before the MULT that consumes them.

    Pad lanes (intra-group width padding + the last vector's missing rows) are
    zeroed in the output by folding a resident KEEP-mask (1.0 real / 0.0 pad)
    into rvec once before Pass 4 (rvec <- rvec*keep), since the MULT hardware
    mask is inert in wide FP32 mode.

    CR map (CR0/CR1 are READ-ONLY constants):
      CR0=0   CR1=1(=1.0/incr)   CR2=OUT  CR3=CVEC  CR4=NUM  CR5=CMAX  CR6=RVEC
      CR7=512(chunk stride)  CR8=KEEP_ADDR  CR10=INPUT  CR11=NUM_VECTORS
      CR12=SCRATCH_ADDR  CR13=W*4 (initial fold shift bytes)  CR14=512 (fold stop)
      CR15=dstructure valid_elements=128

    ";;" ends a VLIW word, ";" separates sub-instructions; LR has 3 sub-slots.
========================================================================== -#}

{%- set lr_ioff   = "lr0" -%}  {#- working input/num/out byte offset (steps by 512) -#}
{%- set lr_vec    = "lr1" -%}  {#- vector index v (the reduced row dimension) -#}
{%- set lr_cyc    = "lr2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_woff   = "lr3" -%}  {#- working write offset (NUM / OUT), steps by 512 -#}
{%- set lr_shift  = "lr4" -%}  {#- fold shift in bytes (W*4, doubling) -#}

{#- ===================================================================== -#}
{#- PASS 1 -- cmax[col] = max over all rows of (c * x).                     -#}
{#-   (1) down-vector running ACC.MAX; (2) cross-group shift-fold.          -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}}  cr0 ;
    SET {{lr_ioff}} cr0 ;
    SET {{lr_vec}}  cr1 ;;                                  {#- v=0 peeled; loop from v=1 -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}} ;;       {#- r_cyclic = C_VEC -#}

    LDR_MULT_REG r0 {{lr_ioff}} cr10 ;;                     {#- R0 = x[0] (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;
    ADD {{lr_ioff}} {{lr_ioff}} cr7 ;;
    ACC.MAX.FIRST ;;

p1_vec:
    LDR_MULT_REG r0 {{lr_ioff}} cr10 ;;
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;
    ADD {{lr_ioff}} {{lr_ioff}} cr7 ;
    ADD {{lr_vec}}  {{lr_vec}}  cr1 ;;
    ACC.MAX ;;
    BLT {{lr_vec}} cr11 p1_vec ;;

{#- ---- cross-group fold: shift = W*4; while shift<512: max(r_acc, r_acc<<shift) -#}
    SET {{lr_shift}} cr13 ;;                                {#- shift = W*4 -#}
p1_fold:
    ACTIVATE.QUANTIZE identity cr15 ;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr12 ;;                     {#- scratch <- current partial -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr12 {{lr_cyc}} ;;       {#- r_cyclic = partial (full 512B) -#}
    MULT.RC.VE   {{lr_shift}} cr1 0 {{lr_cyc}} cr15 ;       {#- mult_res = partial cyclically shifted by `shift` -#}
    ACC.MAX ;;                                              {#- r_acc = max(partial, partial<<shift) -#}
    ADD {{lr_shift}} {{lr_shift}} {{lr_shift}} ;;           {#- shift *= 2 -#}
    BLT {{lr_shift}} cr14 p1_fold ;;

    ACTIVATE.QUANTIZE identity cr15 ;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr5 ;;                      {#- CMAX <- folded per-column max -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[v,col] = 2^(c*x[v,col] - cmax[col]).  Writes NUM region.  -#}
{#- ===================================================================== -#}
    SET {{lr_ioff}} cr0 ;
    SET {{lr_woff}} cr0 ;
    SET {{lr_vec}}  cr0 ;;

p2_vec:
    LDR_MULT_REG r0 {{lr_ioff}} cr10 ;;
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr3 {{lr_cyc}} ;;        {#- r_cyclic = C_VEC (reload each vec) -#}
    MULT.RC.VV   {{lr_cyc}} r0 0 {{lr_cyc}} cr15 ;          {#- mult_res = c*x[v] -#}
    acc.add.first ;;

    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr5 {{lr_cyc}} ;;        {#- r_cyclic = cmax -#}
    MULT.RC.VE   {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;         {#- mult_res = cmax*1.0 -#}
    acc.sub ;                                               {#- r_acc = c*x[v] - cmax -#}
    ACTIVATE.QUANTIZE exp2 cr15 ;                            {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_woff}} cr4 ;;

    ADD {{lr_ioff}} {{lr_ioff}} cr7 ;
    ADD {{lr_woff}} {{lr_woff}} cr7 ;
    ADD {{lr_vec}}  {{lr_vec}}  cr1 ;;
    BLT {{lr_vec}} cr11 p2_vec ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sum[col] = SUM over all rows of num.                          -#}
{#-   (1) down-vector running ACC.ADD; (2) cross-group shift-fold; recip.   -#}
{#- ===================================================================== -#}
    SET {{lr_ioff}} cr0 ;
    SET {{lr_vec}}  cr1 ;;                                  {#- v=0 peeled -#}

    LDR_CYCLIC_MULT_REG {{lr_ioff}} cr4 {{lr_cyc}} ;;       {#- r_cyclic = num[0] -#}
    MULT.RC.VE   {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;
    ADD {{lr_ioff}} {{lr_ioff}} cr7 ;;
    acc.add.first ;;

p3_vec:
    LDR_CYCLIC_MULT_REG {{lr_ioff}} cr4 {{lr_cyc}} ;;
    MULT.RC.VE   {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;
    ADD {{lr_ioff}} {{lr_ioff}} cr7 ;
    ADD {{lr_vec}}  {{lr_vec}}  cr1 ;;
    acc.add ;;
    BLT {{lr_vec}} cr11 p3_vec ;;

{#- ---- cross-group fold (sum) ----------------------------------------- -#}
    SET {{lr_shift}} cr13 ;;
p3_fold:
    ACTIVATE.QUANTIZE identity cr15 ;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr12 ;;
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr12 {{lr_cyc}} ;;       {#- r_cyclic = partial (full 512B) -#}
    MULT.RC.VE   {{lr_shift}} cr1 0 {{lr_cyc}} cr15 ;       {#- mult_res = partial cyclically shifted by `shift` -#}
    acc.add ;;                                              {#- r_acc += partial<<shift -#}
    ADD {{lr_shift}} {{lr_shift}} {{lr_shift}} ;;
    BLT {{lr_shift}} cr14 p3_fold ;;

    ACTIVATE.QUANTIZE reciprocal cr15 ;                     {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr6 ;;                      {#- RVEC <- 1/sum (folded) -#}

{#- ---- fold the keep-mask into rvec once: rvec <- rvec * keep ---------- -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr8 {{lr_cyc}} ;;        {#- r_cyclic = keep-mask -#}
    LDR_MULT_REG r1 {{lr_cyc}} cr6 ;;                       {#- R1 = rvec -#}
    MULT.RC.VV   {{lr_cyc}} r1 0 {{lr_cyc}} cr15 ;          {#- mult_res = keep * rvec -#}
    acc.add.first ;
    ACTIVATE.QUANTIZE identity cr15 ;                        {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr6 ;;                      {#- RVEC <- rvec * keep -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[v,col] = num[v,col] * (rvec*keep)[col].                   -#}
{#- ===================================================================== -#}
    LDR_MULT_REG r1 {{lr_cyc}} cr6 ;;                       {#- R1 = rvec*keep (resident) -#}
    SET {{lr_ioff}} cr0 ;
    SET {{lr_vec}}  cr0 ;;

p4_vec:
    LDR_CYCLIC_MULT_REG {{lr_ioff}} cr4 {{lr_cyc}} ;;       {#- r_cyclic = num[v] -#}
    MULT.RC.VV   {{lr_cyc}} r1 0 {{lr_cyc}} cr15 ;          {#- mult_res = num[v] * (rvec*keep) -#}
    acc.add.first ;
    ACTIVATE.QUANTIZE identity cr15 ;                        {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_ioff}} cr2 ;;

    ADD {{lr_ioff}} {{lr_ioff}} cr7 ;
    ADD {{lr_vec}}  {{lr_vec}}  cr1 ;;
    BLT {{lr_vec}} cr11 p4_vec ;;

end:
    BKPT ;;
