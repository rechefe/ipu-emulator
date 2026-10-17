{#- conv_first_layer.asm -- relu(bias[f] + 3x3 stride-2 pad-1 FP32 conv): 3x256x256 -> 16x128x128, the network's first layer.

    out[f, r, c] = relu( bias[f]
                         + SUM_ch SUM_kr SUM_kc W[f,ch,kr,kc] * in[ch, 2r+kr, 2c+kc] )

    Hand-optimised for this one shape (no parameterisation): batch-norm scale
    is assumed folded into the weights, so only the per-filter bias remains
    separate, accumulated once before the conv taps; ReLU is applied at the
    end. FP32 element format: weights and activations are genuine floats, no
    INT8 quantization anywhere in this kernel.

    Design notes

    - MEMORY (rows of 128 elements; regions pinned by kernel.toml at the old
      harness's addresses, byte address / 512):
        x       row 0:    row-interleaved by channel: (row r, channel ch) at
                          element (r*3 + ch)*256 (pack row_interleaved), so one
                          channel-row is two rows and one spatial row of all
                          three channels is row_group = 6 rows.
        filters row 2048: one row per filter: element 0 = bias, elements
                          1 + ch*9 + (kr+1)*3 + (kc+1) = the 27 taps, the rest
                          zero. R0 holds one filter block.
        mask    row 2080: the 128-byte R_MASK blob (8 slots x 16 bytes, bit 1
                          keeps a lane): slot 0 all keep, slot 1 zeros lane 0,
                          slot 2 zeros lane 127. Only slot 0 is selected: the
                          edges are zeroed by mask_shift (below), so slots 1
                          and 2 are dead data.
        temp    rows 2082-2083: temp0 (slot-0 half) and temp1 (slot-1 half).
        y       row 2560: row-interleaved: (row r, filter f) at element
                          (r*16 + f)*128 (pack row_interleaved).

    - The 256-wide input strip trick (no seam): each channel-row is two
      128-lane R_CYCLIC slots (slot 0 = cols 0..127 at cyclic index 0, slot 1
      = cols 128..255 at index 128), loaded back to back, so the full row is
      one contiguous 256-element strip. A kc = -1/0/+1 tap is a cyclic read at
      center-1 / center / center+1: there is no half seam (output col 64's
      center is position 128, its left neighbour position 127, the last
      element of slot 0, read with no wrap). ACC.STRIDE decimates the
      256-wide strip to 128 stride-2 outputs.

    - Datapath per (output row, filter):
        slot 0: r_acc = bias; += taps over slot0 lanes; QUANTIZE identity -> temp0.
        slot 1: r_acc = bias; += taps over slot1 lanes; QUANTIZE identity -> temp1.
        reload temp0, identity-MULT, ACC.STRIDE 64 on off offset 0 -> r_acc[0..63].
        reload temp1, identity-MULT, ACC.STRIDE 64 on off offset 2 -> r_acc[64..127].
        ACTIVATE.QUANTIZE relu -> ST.AAQ at lr6 (+y).

    - Two loops:
        EDGE   (output row 0):      kr in {0,+1}    -> 4 loads/ch (row -1 out of bounds).
        MIDDLE (output rows 1..127): kr in {-1,0,+1} -> 6 loads/ch (row 255 in
                                    bounds, so the last output row is a middle row).
      The skipped kr=-1 taps in the edge body advance the kernel index by 3 so
      ch1/ch2 tap indices stay aligned.

    - Borders (true pad = 1): top row (out row 0): the kr=-1 input row is out of
      bounds and skipped; bottom row (out row 127): center 254, kr=+1 -> row
      255, in bounds, no special case; left (col 0) and right (col 255): the
      cyclic wrap would pull the opposite edge in, so mask_shift zeros lane 0
      (kc=-1 on slot 0) and lane 127 (kc=+1 on slot 1).

    - Registers ([abi]; rows unless noted): channel_stride = 2 (channel to
      channel within a row group), row_group = 6 (one input spatial row, 3
      channels), block_stride = 1 (the filter-block stride, also the output
      advance), row_advance = 12 (input advance per OUTPUT row, stride 2),
      out_rows = 128 (the output-row loop bound, not an XMEM value),
      filter_rows = 16 (filter-block limit), slot1 = 128 (the R_CYCLIC slot-1
      ELEMENT index; kept apart from block_stride because the two units
      diverge under row addressing). CR15 is the default dstructure
      (valid_elements = 128); CR2 is not used.
    - LR: lr0=0 lr1=1 lr2=+1 lr3=-1 lr4=kern idx lr5=center row-group base
      lr6=out ptr lr7=filter block base lr8=row counter lr9,lr10,lr14=scratch
      lr11=128 lr12=scratch.
-#}
{#- ============================ MACROS ============================ -#}
{#- Load channel-row (ch, kr) into both slots.  Center base = lr5. -#}
{%- macro load_ch_kr(ch, kr) %}
{%-   if kr == -1 %}
    SUB                 LR9 LR5 {{ cr.row_group }};;
{%-   elif kr == 0 %}
    ADD                 LR9 LR5 LR0;;
{%-   else %}
    ADD                 LR9 LR5 {{ cr.row_group }};;
{%-   endif %}
{%-   if ch >= 1 %}
    ADD                 LR9 LR9 {{ cr.channel_stride }};;
{%-   endif %}
{%-   if ch == 2 %}
    ADD                 LR9 LR9 {{ cr.channel_stride }};;
{%-   endif %}
    LD.MULT.CYCLIC LR9 {{ cr.x }} LR0;
    ADD                 LR10 LR9 {{ cr.block_stride }};;
    LD.MULT.CYCLIC LR10 {{ cr.x }} LR11;;
{%- endmacro -%}
{#- Emit the 3 kc taps of one (slot, ch, kr).
    lr4 CONVENTION: on entry lr4 holds (this kr's kc=-1 tap index) - 1, i.e. it
    points at the LAST index the previous group consumed.  Each of the 3 tap
    words carries its own `INC lr4 1` in the SAME word as its MULT.

    Why that is correct, and why the obvious alternative is NOT:
    MULT.RC.VE's `src` operand is read **LIVE**, not from the snapshot
    (`_mult_resolve_lcr_scalar`, ipu.py:768 -- "the LR holds an INDEX into Ra ->
    read it LIVE"), and the LR slot dispatches BEFORE the MULT slot within a
    cycle (Cycle.step).  So an `INC lr4` sharing a word with a MULT is
    visible to that MULT: the MULT reads the ALREADY-INCREMENTED value.

    That makes the pre-increment form used here exact -- each word bumps lr4 and
    then consumes the weight it just moved to.  It also means a
    trailing standalone `INC lr4 1;;` could NOT simply be merged into
    the third tap's word: that would have made the third MULT see L+3 instead of
    L+2, silently reading the NEXT group's first weight.  (Verified empirically:
    a MULT with a same-word INC reads weight[idx+1], not weight[idx].)

    Folding the increments this way removes one standalone VLIW word per 3-tap
    group -- 18 words per middle filter body, 12 per edge body.

    Horizontal edges are zeroed by mask_shift alone (mask slot 0 = all keep,
    CR15 partition P1 -> shift +1 zeros lane 0 = col 0; shift -1 zeros lane 127
    = col 255).  Seam taps (slot0 kc=+1 lane127=col128; slot1 kc=-1 lane0=col127)
    must NOT shift-zero -> shift 0. -#}
{%- macro kc_taps(slot) %}
{%-   if slot == 0 %}
    SUB                 LR14 LR0 CR1;                # rc_idx = -1 (slot0 kc=-1)
    INC                 LR4 1;
    MULT.RC.VE          LR14 LR4 0 LR2 CR15;         # left edge: shift +1 zeros lane0 (col0)
    ACC.ADD;;
    INC                 LR4 1;
    MULT.RC.VE          LR0 LR4 0 LR0 CR15;          # kc=0, no shift
    ACC.ADD;;
    INC                 LR4 1;
    MULT.RC.VE          LR1 LR4 0 LR0 CR15;          # slot0 kc=+1 (rc_idx=1): seam KEEP, no shift
    ACC.ADD;;
{%-   else %}
    SUB                 LR14 LR11 CR1;               # rc_idx = 127 (slot1 kc=-1 -> seam nbr col127)
    INC                 LR4 1;
    MULT.RC.VE          LR14 LR4 0 LR0 CR15;         # seam KEEP, no shift
    ACC.ADD;;
    INC                 LR4 1;
    MULT.RC.VE          LR11 LR4 0 LR0 CR15;         # rc_idx = 128 (slot1 kc=0)
    ACC.ADD;;
    ADD                 LR14 LR11 CR1;               # rc_idx = 129 (slot1 kc=+1)
    INC                 LR4 1;
    MULT.RC.VE          LR14 LR4 0 LR3 CR15;         # right edge: shift -1 zeros lane127 (col255)
    ACC.ADD;;
{%-   endif %}
{%- endmacro -%}
{#- Full filter body for a given kr list (edge=[0,1], middle=[-1,0,1]). -#}
{%- macro filter_body(krlist) %}
{%- for slot in [0, 1] %}
    # ---- SLOT {{slot}} ----
{%-   if slot == 0 %}
    LD.MULT        R0 LR7 {{ cr.filters }};;
{%-   endif %}
    # Bias tap: MULT.EE reads lr4 live too, so lr4 must be 0 (bias byte index)
    # in THIS word.  The following `SET lr4 cr0` re-arms it to 0 = "last index
    # consumed" under kc_taps' pre-increment convention, so the first tap's own
    # `INC lr4 1` lands on weight index 1.  Both fit in one word (2 of 3 LR
    # sub-slots): SET dispatches in the LR phase, and since both SETs write the
    # same value there is no ordering hazard between them.
    SET                 LR4 CR0;
    MULT.EE             LR4 CR1 0 LR0 CR15;
    ACC.ADD.FIRST;;
{%-   for ch in [0, 1, 2] %}
    # -- channel {{ch}} --
{%-     if krlist == [0, 1] %}
    INC                 LR4 3;;                      # skip kr=-1 taps (advance +3)
{%-     endif %}
{%-     for kr in krlist %}
{{ load_ch_kr(ch, kr) }}
{{ kc_taps(slot) }}
{%-     endfor %}
{%-   endfor %}
    # ACTIVATE.QUANTIZE reads r_acc LIVE and ST.AAQ reads
    # post_aaq_reg LIVE (store dispatches after aaq in the per-cycle order),
    # so they co-issue in one word — no separate store cycle needed.
{%-   if slot == 0 %}
    ACTIVATE.QUANTIZE   identity CR15;
    ST.AAQ    LR0 {{ cr.temp }};;
{%-   else %}
    ACTIVATE.QUANTIZE   identity CR15;
    # XMEM offset for temp1 = +1 ROW (temp0/temp1 are adjacent 128-byte rows);
    # lr11 holds 128 in its r_cyclic-ELEMENT role, so use lr1 (=1) here instead.
    ST.AAQ    LR1 {{ cr.temp }};;
{%-   endif %}
{%- endfor %}

    # ---- decimate + relu + store ----
    ADD                 LR12 LR1 CR1;;               # lr12 = 2 (offset%4=2 -> r_acc start 64)
    LD.MULT.CYCLIC LR0 {{ cr.temp }} LR0;;
    MULT.RC.VE          LR0 CR1 0 LR0 CR15;;
    ACC.STRIDE          64 on off LR0;;
    LD.MULT.CYCLIC LR1 {{ cr.temp }} LR0;;   # XMEM offset +1 row (temp1); lr11 stays r_cyclic-ELEMENT-only
    MULT.RC.VE          LR0 CR1 0 LR0 CR15;;
    # acc.stride (finalizes r_acc), ACTIVATE.QUANTIZE (reads r_acc LIVE), and
    # ST.AAQ (reads post_aaq_reg LIVE; store dispatches after aaq)
    # co-issue in one word — no separate ACTIVATE/store cycles needed.
    ACC.STRIDE          64 on off LR12;
    ACTIVATE.QUANTIZE   relu CR15;
    ST.AAQ    LR6 {{ cr.y }};;
{%- endmacro -%}

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 LR0 CR0;
    LD.MASK   LR0 {{ cr.mask }};;

    ADD                 LR1 LR0 CR1;                 # +1 (kc=+1 rc_idx for slot0; const)
    ADD                 LR2 LR0 CR1;                 # +1  (mask_shift kc=-1)
    SUB                 LR3 LR0 CR1;;                # -1  (mask_shift kc=+1)

    SET                 LR11 {{ cr.slot1 }};;                  # 128 (slot1 rc ELEMENT index / temp1 off)

    SET                 LR6 CR0;                     # output pointer = 0
    SET                 LR8 CR0;                     # output row counter = 0
    SET                 LR5 CR0;;                    # center row-group base = 0 (row 0)

# ===========================================================================
# EDGE loop: output row 0 (kr in {0,+1})
# ===========================================================================

    SET                 LR7 CR0;;                    # filter block base = 0

row0_filter_loop:
{{ filter_body([0, 1]) }}
    ADD                 LR6 LR6 {{ cr.block_stride }};
    ADD                 LR7 LR7 {{ cr.block_stride }};;
    BLT.S                 LR7 {{ cr.filter_rows }} row0_filter_loop;;

    # advance to output row 1
    ADD                 LR5 LR5 {{ cr.row_advance }};                 # center row-group += row_advance (two input spatial rows)
    INC                 LR8 1;;

# ===========================================================================
# MIDDLE loop: output rows 1..127 (kr in {-1,0,+1})
# ===========================================================================

row_loop:
    SET                 LR7 CR0;;

filter_loop:
{{ filter_body([-1, 0, 1]) }}
    ADD                 LR6 LR6 {{ cr.block_stride }};
    ADD                 LR7 LR7 {{ cr.block_stride }};;
    BLT.S                 LR7 {{ cr.filter_rows }} filter_loop;;

    ADD                 LR5 LR5 {{ cr.row_advance }};
    INC                 LR8 1;;
    BLT.S                 LR8 {{ cr.out_rows }} row_loop;;

end:
    HALT;;
