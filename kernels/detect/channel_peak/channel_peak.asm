{#- ==========================================================================
    channel_peak.asm -- detector confidence + fixed-threshold gate, FP32 wide

      conf[n] = max over c of logits[c, n]
      keep[n] = relu(conf[n] - tau)

    SuperPoint's detector read-out with cells in lanes. The per-cell confidence
    is the maximum over the channel planes; the gate keeps the cells whose
    confidence clears tau, with a positive shifted score, and zeroes the rest.

    WHY A MAX OVER CHANNELS IS ARGMAX-EQUIVALENT TO THE SOFTMAX PATH:
      argmax(softmax(x)) == argmax(x), so which cell wins is unchanged by
      skipping the softmax. The VALUE is not a probability though -- it is the
      raw logit -- so this replaces the softmax only where the ranking matters.

    NO FAN-OUT, NO AGG:
      Every cell is an independent maximum and the datapath is 128 lanes wide,
      so one pass down the channel planes reduces 128 cells at once with
      ACC.MAX. The running maximum stays a full 128-element vector.

    THE FIRST PLANE IS PEELED:
      ACC.MAX.FIRST seeds R_ACC from plane 0 without needing a -inf vector, but
      the channel count is a run-time bound, so the first iteration cannot carry
      a different accumulate mode. Plane 0 is therefore issued before the loop,
      and a BGE.S skips the loop entirely when there is only one plane.

    THE THRESHOLD IS A RESIDENT VECTOR:
      tau is a per-run input vector, not a CR scalar (an FP32 CR scalar would
      have to hold tau's float32 bit pattern, Q49). It lives
      in R_CYCLIC slot 1 for the whole run and is subtracted with ACC.SUB. The
      older kernel negated it and added (AGG max value_cr(-1) then ACC.ADD_AAQ);
      ACC.SUB does it directly.

      R_ACC is not modified by ACTIVATE.EXACT, so the confidence is stored
      and then subtracted from in place -- the maximum is computed once.

    ABI (kernel.toml [abi]; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE      (rows)   CR3  = CONFIDENCE_BASE (rows: y[0])
      CR4  = KEEP_BASE       (rows: y[1])   CR5  = TAU_ROW   (rows)
      CR6  = TILES -- rows per channel plane, AND the tile-loop bound
      CR7  = CHANNELS
      CR10 = 128 (R_CYCLIC slot-1 index, where tau lives)
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> AAQ_EXACT ->
    COND, so ACC.SUB and the ACTIVATE that reads its result share one word,
    while R_CYCLIC needs a word of separation after its load (its CONTENTS are
    read from the start-of-word snapshot).
========================================================================== -#}

{%- set lr_zero = "LR0" -%}  {#- 0: mask_shift and the R_CYCLIC slot-0 index -#}
{%- set lr_t    = "LR1" -%}  {#- tile counter AND the tile's row offset -#}
{%- set lr_c    = "LR2" -%}  {#- channel counter -#}
{%- set lr_addr = "LR3" -%}  {#- walking input row address -#}
{%- set lr_tau  = "LR4" -%}  {#- 128: R_CYCLIC slot-1 index, the resident tau -#}

    SET {{lr_zero}} CR0 ;
    SET {{lr_t}}    CR0 ;
    SET {{lr_tau}}  CR10 ;;                              {#- 128 -#}
    LD.MULT.CYCLIC {{lr_zero}} CR5 {{lr_tau}} ;;    {#- slot1 <- tau, resident -#}

tile_loop:
    ADD {{lr_addr}} {{lr_t}} CR0 ;
    SET {{lr_c}} CR0 ;;                                  {#- plane 0 of this tile -#}
    LD.MULT.CYCLIC {{lr_addr}} CR2 {{lr_zero}} ;;   {#- slot0 <- x[0, t] -#}
    ADD {{lr_addr}} {{lr_addr}} CR6 ;
    ADD {{lr_c}} {{lr_c}} CR1 ;;
    MULT.RC.VE {{lr_zero}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX.FIRST ;;
    BGE.S {{lr_c}} CR7 peak_done ;;                        {#- one plane only: nothing to fold in -#}

chan_loop:
    LD.MULT.CYCLIC {{lr_addr}} CR2 {{lr_zero}} ;;   {#- slot0 <- x[c, t] -#}
    ADD {{lr_addr}} {{lr_addr}} CR6 ;
    ADD {{lr_c}} {{lr_c}} CR1 ;;
    MULT.RC.VE {{lr_zero}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.MAX ;
    BLT.S {{lr_c}} CR7 chan_loop ;;                        {#- post-increment: bound CHANNELS -#}

peak_done:
    ACTIVATE.EXACT identity CR15.L CR3 {{lr_t}} ;; {#- CONF[t] = max over planes -#}
    MULT.RC.VE {{lr_tau}} CR1 0 {{lr_zero}} CR15.L {{ cr.qf }}.L ; ACC.SUB ;
    ACTIVATE.EXACT relu CR15.L CR4 {{lr_t}} ;; {#- KEEP[t] = relu(conf - tau) -#}
    ADD {{lr_t}} {{lr_t}} CR1 ;;
    BLT.S {{lr_t}} CR6 tile_loop ;;

end:
    HALT ;;
