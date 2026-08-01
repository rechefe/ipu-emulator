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
      (2) CROSS-GROUP fold: a linear rc_idx walk (0, W, 2W, ... (rpv-1)*W, with
          ACC.MAX / ACC.ADD) folds the rpv groups together AND broadcasts the
          full per-column result back across every group's lanes (see the
          CROSS-GROUP FOLD section below for the full derivation). After the
          fold, cmax / rvec are full 128-lane vectors lane-aligned for the
          Pass 2/4 trips.

    CROSS-GROUP FOLD (rewritten -- see issue #182/PR #196 background): rc_idx
    is an ELEMENT index into r_cyclic's 512-element ring (matches
    LDR_CYCLIC_MULT_REG's index), and only elements [0:128) of that ring hold
    real data after a SINGLE LDR_CYCLIC_MULT_REG -- wide mode does NOT wrap a
    loaded row at 128 elements the way the old code assumed; the ring only
    wraps at its full 512-element span. So a fold built by shifting rc_idx up
    to 128 (let alone doubling past it) reads uninitialised ring memory for
    every rc_idx > 0, not a wrapped copy of the real data.

    Fix: DUPLICATE the down-vector partial into TWO adjacent ring slots
    (LDR_CYCLIC_MULT_REG at index=0 AND index=128, same source data), so
    elements [0:256) are all real (the partial repeated twice back-to-back).
    Then a plain rpv-step linear walk rc_idx = 0, W, 2W, ..., (rpv-1)*W
    (W=group_width) -- max rc_idx+128 = (rpv-1)*W+128 <= 256 since rpv*W=128
    -- reads only ever land inside [0:256), i.e. always real data, no masking
    needed at all. Each step's read pairs group p (real, at rc_idx=p*W) with
    whatever already-folded value sits at r_acc; running ACC.MAX/ACC.ADD
    converges every lane to the true combined value after rpv steps (verified
    numerically for rpv in {2,4,8} before implementing -- see the harness
    docstring for the derivation).

    Issue #157: MULT reads Ra/r_cyclic DATA from the snapshot (visible NEXT
    cycle), so loads sit one VLIW word before the MULT that consumes them.

    Pad lanes (intra-group width padding + the last vector's missing rows) are
    zeroed in the output by folding a resident KEEP-mask (1.0 real / 0.0 pad,
    a plain data vector multiplied in via MULT.RC.VV) into rvec once before
    Pass 4 (rvec <- rvec*keep).

    CR map (CR0/CR1 are READ-ONLY constants):
      CR0=0   CR1=1(=1.0/incr)   CR2=OUT  CR3=CVEC  CR4=NUM  CR5=CMAX  CR6=RVEC
      CR7=1 (row stride)  CR8=free  CR9=LANES(128) (element index for the
      duplicate LDR_CYCLIC_MULT_REG's index operand)  CR10=INPUT row
      CR11=NUM_VECTORS  CR12=SCRATCH_ADDR row  CR13=W (fold rc_idx step,
      elements)  CR14=rpv (fold step count)  CR15=dstructure
      valid_elements=128 (default; every lane live throughout)

    ";;" ends a VLIW word, ";" separates sub-instructions; LR has 3 sub-slots.
========================================================================== -#}

{%- set lr_ioff   = "lr0" -%}  {#- working input/num/out byte offset (steps by 512) -#}
{%- set lr_vec    = "lr1" -%}  {#- vector index v (the reduced row dimension) -#}
{%- set lr_cyc    = "lr2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_woff   = "lr3" -%}  {#- working write offset (NUM / OUT), steps by 512 -#}
{%- set lr_shift  = "lr4" -%}  {#- fold rc_idx (element offset, walks 0, W, 2W, ...) -#}
{%- set lr_p      = "lr5" -%}  {#- fold step counter (1..rpv-1) -#}
{%- set lr_maskrow = "lr6" -%}  {#- keep-mask row (CVEC row + 1), Pass 3 only -#}
{%- set lr_lanes  = "lr7" -%}  {#- LANES(128): index for the fold's duplicate LDR_CYCLIC_MULT_REG -#}

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

{#- ---- cross-group fold (max): duplicate the partial into slots 0+1, then rpv -#}
{#- linear steps rc_idx = 0, W, 2W, ..., (rpv-1)*W -- all real, no masking. -#}
    ACTIVATE.QUANTIZE identity cr15 ;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr12 ;;                     {#- scratch <- down-vector partial -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr12 {{lr_cyc}} ;;       {#- r_cyclic slot0 = partial -#}
    SET {{lr_lanes}} cr9 ;;
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr12 {{lr_lanes}} ;;     {#- r_cyclic slot1 = SAME partial (index=LANES) -#}

    SET {{lr_shift}} cr0 ;;                                 {#- shift = 0 (p=0) -#}
    MULT.RC.VE   {{lr_shift}} cr1 0 {{lr_cyc}} cr15 ;
    ACC.MAX.FIRST ;;

    SET {{lr_p}}     cr1 ;;                                 {#- p = 1 -#}
p1_fold:
    ADD {{lr_shift}} {{lr_shift}} cr13 ;;                   {#- shift += W -#}
    MULT.RC.VE   {{lr_shift}} cr1 0 {{lr_cyc}} cr15 ;
    ACC.MAX ;;
    ADD {{lr_p}}     {{lr_p}}     cr1 ;;
    BLT {{lr_p}} cr14 p1_fold ;;                            {#- loop while p < rpv -#}

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

{#- ---- cross-group fold (sum): same duplicate-slot linear walk as Pass 1. ---- -#}
    ACTIVATE.QUANTIZE identity cr15 ;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr12 ;;                     {#- scratch <- down-vector partial -#}
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr12 {{lr_cyc}} ;;       {#- r_cyclic slot0 = partial -#}
    SET {{lr_lanes}} cr9 ;;
    LDR_CYCLIC_MULT_REG {{lr_cyc}} cr12 {{lr_lanes}} ;;     {#- r_cyclic slot1 = SAME partial (index=LANES) -#}

    SET {{lr_shift}} cr0 ;;                                 {#- shift = 0 (p=0) -#}
    MULT.RC.VE   {{lr_shift}} cr1 0 {{lr_cyc}} cr15 ;
    acc.add.first ;;

    SET {{lr_p}}     cr1 ;;                                 {#- p = 1 -#}
p3_fold:
    ADD {{lr_shift}} {{lr_shift}} cr13 ;;                   {#- shift += W -#}
    MULT.RC.VE   {{lr_shift}} cr1 0 {{lr_cyc}} cr15 ;
    acc.add ;;
    ADD {{lr_p}}     {{lr_p}}     cr1 ;;
    BLT {{lr_p}} cr14 p3_fold ;;                            {#- loop while p < rpv -#}

    ACTIVATE.QUANTIZE reciprocal cr15 ;                     {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr6 ;;                      {#- RVEC <- 1/sum (folded) -#}

{#- ---- fold the keep-mask into rvec once: rvec <- rvec * keep ---------- -#}
    SET {{lr_maskrow}} cr3 ;;
    ADD {{lr_maskrow}} {{lr_maskrow}} cr1 ;;                {#- keep-mask row = CVEC row + 1 -#}
    LDR_CYCLIC_MULT_REG {{lr_maskrow}} cr0 {{lr_cyc}} ;;    {#- r_cyclic = keep-mask -#}
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
