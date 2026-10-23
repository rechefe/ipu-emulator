{#- ==========================================================================
    softmax_columns_packed.asm -- packed sub-128 column-softmax, FP32

    softmax(x[r,col]) = 2^(c*(x[r,col]-cmax[col])) / SUM_r 2^(c*(x[r,col]-cmax[col]))
    c = log2(e), so 2^(c*d) == e^d and the IPU's native exp2 applies directly.

    y[:, col] = softmax(x[:, col]) for a (rows, width) matrix with width <= 64,
    packed rpv = 128/W rows per 128-element vector (W = partition_size(width)
    in {16, 32, 64}; elements [g*W : g*W+W) hold the matrix row that maps to
    group g). The harness packs and unpacks x and y that way
    (ipu_kernels.softmax.pack col_slots_negpad), so the files keep the plain
    (rows, width) layout. Softmax is per column, reduced DOWN all
    rows -- and a column's members span BOTH element-groups (within a vector) AND
    vectors. So each reduce has two stages:

      (1) DOWN-VECTOR partial: running ACC.MAX / ACC.ADD over num_vectors gives,
          per element g*W+col, the reduce over the rows that landed in group g.
      (2) CROSS-GROUP fold: a linear rc_idx walk (0, W, 2W, ... (rpv-1)*W, with
          ACC.MAX / ACC.ADD) folds the rpv groups together AND broadcasts the
          full per-column result back across every group's elements (see the
          CROSS-GROUP FOLD section below for the full derivation). After the
          fold, cmax / rvec are full 128-element vectors element-aligned for the
          Pass 2/4 trips.

    CROSS-GROUP FOLD (rewritten -- see issue #182/PR #196 background): rc_idx
    is an ELEMENT index into R_CYCLIC's 512-element ring (matches
    LD.MULT.CYCLIC's index), and only elements [0:128) of that ring hold
    real data after a SINGLE LD.MULT.CYCLIC -- FP32 does NOT wrap a
    loaded row at 128 elements the way the old code assumed; the ring only
    wraps at its full 512-element span. So a fold built by shifting rc_idx up
    to 128 (let alone doubling past it) reads uninitialised ring memory for
    every rc_idx > 0, not a wrapped copy of the real data.

    Fix: DUPLICATE the down-vector partial into TWO adjacent ring slots
    (LD.MULT.CYCLIC at index=0 AND index=128, same source data), so
    elements [0:256) are all real (the partial repeated twice back-to-back).
    Then a plain rpv-step linear walk rc_idx = 0, W, 2W, ..., (rpv-1)*W
    (W=group_width) -- max rc_idx+128 = (rpv-1)*W+128 <= 256 since rpv*W=128
    -- reads only ever land inside [0:256), i.e. always real data, no masking
    needed at all. Each step's read pairs group p (real, at rc_idx=p*W) with
    whatever already-folded value sits at R_ACC; running ACC.MAX/ACC.ADD
    converges every element to the true combined value after rpv steps (verified
    numerically for rpv in {2,4,8} before implementing).

    Issue #157: MULT reads Ra/R_CYCLIC DATA at cycle start (visible NEXT
    cycle), so loads sit one instruction word before the MULT that consumes them.

    Pad elements (intra-group width padding + the last vector's missing rows)
    hold NEG_PAD = -1e30 on input: it loses every max and its exp2 underflows to
    0 in the sum, so the fold ignores missing-row groups and width padding for
    free. They are zeroed in the output by folding a resident KEEP-mask (1.0 real / 0.0 pad,
    a plain data vector multiplied in via MULT.RC.VV) into rvec once before
    Pass 4 (rvec <- rvec*keep).

    Registers. All XMEM operands are ROW numbers; one row is 128 FP32 lanes. The
    harness places the regions and sets the CRs from kernel.toml ([abi]):
      cr.x      x, the input logits         cr.y     y, the output
      cr.num    num, the numerators         cr.log2e C_VEC: log2(e) in every lane;
                                                     the KEEP row at cr.log2e + 1
      cr.max    cmax row                    cr.rvec  rvec row
      cr.fold   the fold's scratch row
      cr.row_stride  = 1 (one row per vector)
      cr.dup_index   = 128: the element index of the duplicate LD.MULT.CYCLIC
      cr.device_rows = the vector count (the vector loop bound)
      cr.gw          = W: the fold's rc_idx step, in elements
      cr.rpv         = rpv: the fold's step count
      CR0 = 0 and CR1 = 1 (-> the 1.0 scalar, and the +1 step) are read-only;
      CR15 holds the default dstructure (128 valid elements: every element live
      throughout).

    ";;" ends a instruction word, ";" separates sub-instructions; LR has 3 sub-slots.
========================================================================== -#}

{%- set lr_ioff   = "LR0" -%}  {#- working x / num / y row offset (steps by 1) -#}
{%- set lr_vec    = "LR1" -%}  {#- vector index v (the reduced row dimension) -#}
{%- set lr_cyc    = "LR2" -%}  {#- cyclic index (always 0) -#}
{%- set lr_woff   = "LR3" -%}  {#- working write offset (num / y), steps by 1 -#}
{%- set lr_shift  = "LR4" -%}  {#- fold rc_idx (element offset, walks 0, W, 2W, ...) -#}
{%- set lr_p      = "LR5" -%}  {#- fold step counter (1..rpv-1) -#}
{%- set lr_maskrow = "LR6" -%}  {#- keep-mask row (CVEC row + 1), Pass 3 only -#}
{%- set lr_lanes  = "LR7" -%}  {#- LANES(128): index for the fold's duplicate LD.MULT.CYCLIC -#}

{#- ===================================================================== -#}
{#- PASS 1 -- cmax[col] = max over all rows of (c * x).                     -#}
{#-   (1) down-vector running ACC.MAX; (2) cross-group shift-fold.          -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}}  CR0 ;
    SET {{lr_ioff}} CR0 ;
    SET {{lr_vec}}  CR1 ;;                                  {#- v=0 peeled; loop from v=1 -#}
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.log2e }} {{lr_cyc}} ;;       {#- R_CYCLIC = C_VEC -#}

    LD.MULT R0 {{lr_ioff}} {{ cr.x }} ;;                     {#- R0 = x[0] (visible next cycle) -#}
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} CR15 ;
    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride }} ;;
    ACC.MAX.FIRST ;;

p1_vec:
    LD.MULT R0 {{lr_ioff}} {{ cr.x }} ;;
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} CR15 ;
    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride }} ;
    ADD {{lr_vec}}  {{lr_vec}}  CR1 ;;
    ACC.MAX ;;
    BLT.S {{lr_vec}} {{ cr.device_rows }} p1_vec ;;

{#- ---- cross-group fold (max): duplicate the partial into slots 0+1, then rpv -#}
{#- linear steps rc_idx = 0, W, 2W, ..., (rpv-1)*W -- all real, no masking. -#}
    ACTIVATE.QUANTIZE identity CR15 ;                       {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_cyc}} {{ cr.fold }} ;;                     {#- scratch <- down-vector partial -#}
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.fold }} {{lr_cyc}} ;;       {#- R_CYCLIC slot0 = partial -#}
    SET {{lr_lanes}} {{ cr.dup_index }} ;;
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.fold }} {{lr_lanes}} ;;     {#- R_CYCLIC slot1 = SAME partial (index=LANES) -#}

    SET {{lr_shift}} CR0 ;;                                 {#- shift = 0 (p=0) -#}
    MULT.RC.VE   {{lr_shift}} CR1 0 {{lr_cyc}} CR15 ;
    ACC.MAX.FIRST ;;

    SET {{lr_p}}     CR1 ;;                                 {#- p = 1 -#}
p1_fold:
    ADD {{lr_shift}} {{lr_shift}} {{ cr.gw }} ;;                   {#- shift += W -#}
    MULT.RC.VE   {{lr_shift}} CR1 0 {{lr_cyc}} CR15 ;
    ACC.MAX ;;
    ADD {{lr_p}}     {{lr_p}}     CR1 ;;
    BLT.S {{lr_p}} {{ cr.rpv }} p1_fold ;;                            {#- loop while p < rpv -#}

    ACTIVATE.QUANTIZE identity CR15 ;                       {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_cyc}} {{ cr.max }} ;;                      {#- CMAX <- folded per-column max -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[v,col] = 2^(c*x[v,col] - cmax[col]).  Writes NUM region.  -#}
{#- ===================================================================== -#}
    SET {{lr_ioff}} CR0 ;
    SET {{lr_woff}} CR0 ;
    SET {{lr_vec}}  CR0 ;;

p2_vec:
    LD.MULT R0 {{lr_ioff}} {{ cr.x }} ;;
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.log2e }} {{lr_cyc}} ;;        {#- R_CYCLIC = C_VEC (reload each vec) -#}
    MULT.RC.VV   {{lr_cyc}} R0 0 {{lr_cyc}} CR15 ;          {#- MULT_RES = c*x[v] -#}
    ACC.ADD.FIRST ;;

    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.max }} {{lr_cyc}} ;;        {#- R_CYCLIC = cmax -#}
    MULT.RC.VE   {{lr_cyc}} CR1 0 {{lr_cyc}} CR15 ;         {#- MULT_RES = cmax*1.0 -#}
    ACC.SUB ;                                               {#- R_ACC = c*x[v] - cmax -#}
    ACTIVATE.QUANTIZE exp2 CR15 ;                            {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_woff}} {{ cr.num }} ;;

    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride }} ;
    ADD {{lr_woff}} {{lr_woff}} {{ cr.row_stride }} ;
    ADD {{lr_vec}}  {{lr_vec}}  CR1 ;;
    BLT.S {{lr_vec}} {{ cr.device_rows }} p2_vec ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sum[col] = SUM over all rows of num.                          -#}
{#-   (1) down-vector running ACC.ADD; (2) cross-group shift-fold; recip.   -#}
{#- ===================================================================== -#}
    SET {{lr_ioff}} CR0 ;
    SET {{lr_vec}}  CR1 ;;                                  {#- v=0 peeled -#}

    LD.MULT.CYCLIC {{lr_ioff}} {{ cr.num }} {{lr_cyc}} ;;       {#- R_CYCLIC = num[0] -#}
    MULT.RC.VE   {{lr_cyc}} CR1 0 {{lr_cyc}} CR15 ;
    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride }} ;;
    ACC.ADD.FIRST ;;

p3_vec:
    LD.MULT.CYCLIC {{lr_ioff}} {{ cr.num }} {{lr_cyc}} ;;
    MULT.RC.VE   {{lr_cyc}} CR1 0 {{lr_cyc}} CR15 ;
    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride }} ;
    ADD {{lr_vec}}  {{lr_vec}}  CR1 ;;
    ACC.ADD ;;
    BLT.S {{lr_vec}} {{ cr.device_rows }} p3_vec ;;

{#- ---- cross-group fold (sum): same duplicate-slot linear walk as Pass 1. ---- -#}
    ACTIVATE.QUANTIZE identity CR15 ;                       {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_cyc}} {{ cr.fold }} ;;                     {#- scratch <- down-vector partial -#}
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.fold }} {{lr_cyc}} ;;       {#- R_CYCLIC slot0 = partial -#}
    SET {{lr_lanes}} {{ cr.dup_index }} ;;
    LD.MULT.CYCLIC {{lr_cyc}} {{ cr.fold }} {{lr_lanes}} ;;     {#- R_CYCLIC slot1 = SAME partial (index=LANES) -#}

    SET {{lr_shift}} CR0 ;;                                 {#- shift = 0 (p=0) -#}
    MULT.RC.VE   {{lr_shift}} CR1 0 {{lr_cyc}} CR15 ;
    ACC.ADD.FIRST ;;

    SET {{lr_p}}     CR1 ;;                                 {#- p = 1 -#}
p3_fold:
    ADD {{lr_shift}} {{lr_shift}} {{ cr.gw }} ;;                   {#- shift += W -#}
    MULT.RC.VE   {{lr_shift}} CR1 0 {{lr_cyc}} CR15 ;
    ACC.ADD ;;
    ADD {{lr_p}}     {{lr_p}}     CR1 ;;
    BLT.S {{lr_p}} {{ cr.rpv }} p3_fold ;;                            {#- loop while p < rpv -#}

    ACTIVATE.QUANTIZE reciprocal CR15 ;                     {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_cyc}} {{ cr.rvec }} ;;                      {#- RVEC <- 1/sum (folded) -#}

{#- ---- fold the keep-mask into rvec once: rvec <- rvec * keep ---------- -#}
    SET {{lr_maskrow}} {{ cr.log2e }} ;;
    ADD {{lr_maskrow}} {{lr_maskrow}} CR1 ;;                {#- keep-mask row = CVEC row + 1 -#}
    LD.MULT.CYCLIC {{lr_maskrow}} CR0 {{lr_cyc}} ;;    {#- R_CYCLIC = keep-mask -#}
    LD.MULT R1 {{lr_cyc}} {{ cr.rvec }} ;;                       {#- R1 = rvec -#}
    MULT.RC.VV   {{lr_cyc}} R1 0 {{lr_cyc}} CR15 ;          {#- MULT_RES = keep * rvec -#}
    ACC.ADD.FIRST ;
    ACTIVATE.QUANTIZE identity CR15 ;                        {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_cyc}} {{ cr.rvec }} ;;                      {#- RVEC <- rvec * keep -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[v,col] = num[v,col] * (rvec*keep)[col].                   -#}
{#- ===================================================================== -#}
    LD.MULT R1 {{lr_cyc}} {{ cr.rvec }} ;;                       {#- R1 = rvec*keep (resident) -#}
    SET {{lr_ioff}} CR0 ;
    SET {{lr_vec}}  CR0 ;;

p4_vec:
    LD.MULT.CYCLIC {{lr_ioff}} {{ cr.num }} {{lr_cyc}} ;;       {#- R_CYCLIC = num[v] -#}
    MULT.RC.VV   {{lr_cyc}} R1 0 {{lr_cyc}} CR15 ;          {#- MULT_RES = num[v] * (rvec*keep) -#}
    ACC.ADD.FIRST ;
    ACTIVATE.QUANTIZE identity CR15 ;                        {#- reads R_ACC forwarded -#}
    ST.AAQ {{lr_ioff}} {{ cr.y }} ;;

    ADD {{lr_ioff}} {{lr_ioff}} {{ cr.row_stride }} ;
    ADD {{lr_vec}}  {{lr_vec}}  CR1 ;;
    BLT.S {{lr_vec}} {{ cr.device_rows }} p4_vec ;;

end:
    HALT ;;
