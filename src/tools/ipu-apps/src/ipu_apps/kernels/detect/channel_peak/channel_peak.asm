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
      and a BGE skips the loop entirely when there is only one plane.

    THE THRESHOLD IS A RESIDENT VECTOR:
      CR scalars are integer-only in wide-vector mode (MULT.*'s CR operand
      supplies its low byte), so a fractional tau cannot ride in a CR. It lives
      in R_CYCLIC slot 1 for the whole run and is subtracted with ACC.SUB. The
      older kernel negated it and added (AGG max value_cr(-1) then ACC.ADD_AAQ);
      ACC.SUB does it directly.

      R_ACC is not modified by ACTIVATE.QUANTIZE, so the confidence is stored
      and then subtracted from in place -- the maximum is computed once.

    CR map (set by the harness; CR0/CR1 are READ-ONLY hardware constants):
      CR0  = 0                        CR1  = 1  (-> 1.0 scalar; every +1)
      CR2  = INPUT_BASE      (rows)   CR3  = CONFIDENCE_BASE (rows)
      CR4  = KEEP_BASE       (rows)   CR5  = TAU_ROW         (rows)
      CR6  = TILES -- rows per channel plane, AND the tile-loop bound
      CR7  = CHANNELS
      CR10 = 128 (R_CYCLIC slot-1 index, where tau lives)
      CR15 = dstructure (valid_elements = 128)

    LR uses 3 sub-slots; ";;" ends a VLIW word, ";" separates sub-instructions.
    Slot order within a word is LR -> LOAD -> MULT -> ACC -> AAQ -> STORE ->
    COND, so ACC.SUB and the ACTIVATE that reads its result share one word,
    while R_CYCLIC needs a word of separation after its load (its CONTENTS are
    read from the start-of-word snapshot).
========================================================================== -#}

{%- set lr_zero = "lr0" -%}  {#- 0: mask_shift and the R_CYCLIC slot-0 index -#}
{%- set lr_t    = "lr1" -%}  {#- tile counter AND the tile's row offset -#}
{%- set lr_c    = "lr2" -%}  {#- channel counter -#}
{%- set lr_addr = "lr3" -%}  {#- walking input row address -#}
{%- set lr_tau  = "lr4" -%}  {#- 128: R_CYCLIC slot-1 index, the resident tau -#}

    SET {{lr_zero}} cr0 ;
    SET {{lr_t}}    cr0 ;
    SET {{lr_tau}}  cr10 ;;                              {#- 128 -#}
    LDR_CYCLIC_MULT_REG {{lr_zero}} cr5 {{lr_tau}} ;;    {#- slot1 <- tau, resident -#}

tile_loop:
    ADD {{lr_addr}} {{lr_t}} cr0 ;
    SET {{lr_c}} cr0 ;;                                  {#- plane 0 of this tile -#}
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot0 <- x[0, t] -#}
    ADD {{lr_addr}} {{lr_addr}} cr6 ;
    ADD {{lr_c}} {{lr_c}} cr1 ;;
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX.FIRST ;;
    BGE {{lr_c}} cr7 peak_done ;;                        {#- one plane only: nothing to fold in -#}

chan_loop:
    LDR_CYCLIC_MULT_REG {{lr_addr}} cr2 {{lr_zero}} ;;   {#- slot0 <- x[c, t] -#}
    ADD {{lr_addr}} {{lr_addr}} cr6 ;
    ADD {{lr_c}} {{lr_c}} cr1 ;;
    MULT.RC.VE {{lr_zero}} cr1 0 {{lr_zero}} cr15 ; ACC.MAX ;
    BLT {{lr_c}} cr7 chan_loop ;;                        {#- post-increment: bound CHANNELS -#}

peak_done:
    ACTIVATE.QUANTIZE identity cr15 ;
    STR_POST_AAQ_REG {{lr_t}} cr3 ;;                     {#- CONF[t] = max over planes -#}
    MULT.RC.VE {{lr_tau}} cr1 0 {{lr_zero}} cr15 ; ACC.SUB ;
    ACTIVATE.QUANTIZE relu cr15 ;
    STR_POST_AAQ_REG {{lr_t}} cr4 ;;                     {#- KEEP[t] = relu(conf - tau) -#}
    ADD {{lr_t}} {{lr_t}} cr1 ;;
    BLT {{lr_t}} cr6 tile_loop ;;

end:
    BKPT ;;
