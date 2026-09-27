{#- ==========================================================================
    score_threshold.asm -- keypoint selection gate + soft survivor count, FP32

      selected[i] = relu(s[i] - tau)                    (exact set {s > tau})
      count       = SUM_i sigmoid(T * (s[i] - tau))     (approximate |{s > tau}|)

    SuperPoint's keypoint selection. The gate is exact: a surviving score keeps
    a positive, shifted value and a rejected one becomes 0. The count is what
    the HOST bisects on to hit a target k -- it re-launches with a new tau,
    reads the count, and narrows.

    WHAT THIS DOES NOT DO: produce the ranked list of the k largest INDICES.
    Exact ranked top-k needs a per-element compare and index extraction, and
    the ISA has neither a vector compare nor lane-index extraction. The host
    selects the final k from the thresholded scores.

    TWO RESIDENT THRESHOLD VECTORS, NOT ONE:
      CR scalars are integer-only in wide-vector mode, so tau rides in a
      128-element XMEM row. The count needs T*(s - tau), and scaling R_ACC by T
      would cost an XMEM round trip -- so the harness supplies a SECOND resident
      vector holding T*tau, and the count path computes T*s (a vector-vector
      multiply against the resident T vector in R0) and subtracts it. Both
      subtractions are ACC.SUB; neither needs the negate-and-add the older
      kernel used.

    THE PADDING LANES ARE SUPPRESSED, NOT IGNORED:
      Elements past the real count fill the last row, and sigmoid(0 - tau) is
      NOT zero -- padding would inflate the count. The harness fills them with
      tau - 800/T, so T*(pad - tau) = -800 exactly and sigmoid underflows to a
      true zero. relu of the same value is zero too, so the padding costs
      nothing in either output.

    PASS STRUCTURE:
      Pass 1   per row: the gate and the staged sigmoid plane   (8 words/row)
      Pass 2   ACC.ADD down the staged plane -> 128 partial sums (3 words/row)
      Drain    stage the partials, reload, AGG.SUM -> one scalar

      The count cannot accumulate during pass 1: AGG.SUM writes ONE R_ACC slot,
      and the next row's ACC.ADD.FIRST overwrites all 128 -- including that
      slot. Staging the sigmoid plane and reducing it afterwards is what keeps
      the running total out of R_ACC's way.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0 (-> the 0.0 scalar that clears R_ACC)   CR1 = 1
      CR2  = SCORES_BASE   (rows)     CR3  = SELECTED_BASE (rows)
      CR4  = STAGED_BASE   (rows)     -- the sigmoid plane
      CR5  = COUNT_ROW     (rows)     -- partial sums, then the scalar total
      CR6  = TAU_ROW       (rows)     CR7  = TTAU_ROW (rows; holds T*tau)
      CR8  = TVEC_ROW      (rows)     -- 128 copies of T, loaded into R0
      CR9  = ROWS                     CR10 = 128
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND, so a subtract and the ACTIVATE that reads its result share one word,
    while R_CYCLIC needs a word of separation after its load.
========================================================================== -#}

{%- set lr_zero = "lr0" -%}  {#- 0: mask_shift and the R_CYCLIC slot-0 index -#}
{%- set lr_addr = "lr1" -%}  {#- row offset; the three big regions share one layout -#}
{%- set lr_r    = "lr2" -%}  {#- row counter -#}
{%- set lr_tau  = "lr3" -%}  {#- 128: R_CYCLIC slot-1 index, the resident tau -#}
{%- set lr_ttau = "lr4" -%}  {#- 256: R_CYCLIC slot-2 index, the resident T*tau -#}

    SET {{lr_zero}} cr0 ;
    SET {{lr_addr}} cr0 ;
    SET {{lr_r}}    cr0 ;;
    SET {{lr_tau}}  cr10 ;;                              {#- 128 -#}
    ADD {{lr_ttau}} {{lr_tau}} {{lr_tau}} ;;             {#- 256 -#}
    LDR_CYCLIC_MULT_REG {{lr_zero}} cr6 {{lr_tau}} ;;    {#- slot1 <- tau, resident -#}
    LDR_CYCLIC_MULT_REG {{lr_zero}} cr7 {{lr_ttau}} ;;   {#- slot2 <- T*tau, resident -#}
    LDR_MULT_REG r0 {{lr_zero}} cr8 ;;                   {#- R0 <- T, resident -#}

{#- ---- PASS 1: the gate, and the staged sigmoid plane -------------------- -#}
pass1_loop:
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot0 <- s[row] -#}
    ADD {{lr_r}} {{lr_r}} cr1 ;;                         {#- slot0 visible NEXT word -#}
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ; ACC.ADD.FIRST ;;   {#- R_ACC = s -#}
    MULT.RC.VE {{lr_tau}} cr1 0 {{lr_zero}} cr15 ; ACC.SUB ;
    ACTIVATE.QUANTIZE relu cr15 ;
    STR_POST_AAQ_REG {{lr_addr}} cr3 ;;                  {#- SELECTED = relu(s - tau) -#}
    MULT.RC.VV {{lr_zero}} r0 0 {{lr_zero}} cr15 ; ACC.ADD.FIRST ;;    {#- R_ACC = T*s -#}
    MULT.RC.VE {{lr_ttau}} cr1 0 {{lr_zero}} cr15 ; ACC.SUB ;
    ACTIVATE.QUANTIZE sigmoid cr15 ;
    STR_POST_AAQ_REG {{lr_addr}} cr4 ;;                  {#- STAGED = sigmoid(T*(s - tau)) -#}
    ADD {{lr_addr}} {{lr_addr}} cr1 ;
    BLT {{lr_r}} cr9 pass1_loop ;;                       {#- post-increment: bound ROWS -#}

{#- ---- PASS 2: 128 running partial sums down the staged plane ------------ -#}
    MULT.RC.VE {{lr_zero}} cr0 0 {{lr_zero}} cr15 ; ACC.ADD.FIRST ;
    SET {{lr_addr}} cr0 ;
    SET {{lr_r}} cr0 ;;                                  {#- R_ACC = 0 -#}

pass2_loop:
    ADD {{lr_r}} {{lr_r}} cr1 ;
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr4 {{lr_zero}} ;;   {#- slot0 <- staged[row] -#}
    ADD {{lr_addr}} {{lr_addr}} cr1 ;
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ; ACC.ADD ;
    BLT {{lr_r}} cr9 pass2_loop ;;

{#- ---- DRAIN: collapse the 128 partials to one scalar -------------------- -#}
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_zero}} cr5 ;;                  {#- stage the partial vector -#}
    LDR_CYCLIC_MULT_REG {{lr_zero}} cr5 {{lr_zero}} ;;
    NOP ;;                                               {#- slot0 visible NEXT word -#}
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ; AGG.SUM.FIRST {{lr_zero}} cr15 ;;
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_zero}} cr5 ;;                  {#- COUNT[0] = the soft survivor count -#}

end:
    BKPT ;;
