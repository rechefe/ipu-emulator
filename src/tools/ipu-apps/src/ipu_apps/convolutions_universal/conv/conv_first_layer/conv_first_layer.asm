# First-Layer Conv: 256x256x3 -> 128x128x16, stride 2, pad 1, INT8 + BN bias + ReLU.
#
# Rendered with Jinja (no context): only literal for-loop / macro unrolling is
# used, so the harness cannot inject values -- all runtime parameters via CRs.
#
# Input (row-interleaved by channel): (row r, channel ch) at (r*3 + ch)*256.
#   One channel-row = 256 bytes = two R_CYCLIC slots loaded so the full row is one
#   contiguous 256-element strip: slot0 (cols 0..127) at cyclic index 0, slot1
#   (cols 128..255) at cyclic index 128.  A kc=-1/0/+1 tap is a cyclic read at
#   base+kc bytes; the slot0/slot1 boundary carries the true neighbour, so the
#   half seam (output col 64) needs no special handling.
#
# Kernel: 16 filters x 128-byte block.  byte 0 = BN bias; tap byte for
#   (ch,kr,kc) = 1 + ch*9 + (kr+1)*3 + (kc+1).  R0 holds one filter block.
#
# Output (row-interleaved, FINAL address): (row r, filter f) at (r*16 + f)*128.
#
# Datapath per (output row, filter):
#   slot 0: r_acc = bias; += taps over slot0 lanes; QUANTIZE identity -> temp0.
#   slot 1: r_acc = bias; += taps over slot1 lanes; QUANTIZE identity -> temp1.
#   reload temp0, identity-MULT, ACC.STRIDE 64 on off offset 0 -> r_acc[0..63].
#   reload temp1, identity-MULT, ACC.STRIDE 64 on off offset 2 -> r_acc[64..127].
#   ACTIVATE.QUANTIZE relu -> STR_POST_AAQ_REG at lr6 (+cr5).
#
# Two loops:
#   EDGE   (output row 0):     kr in {0,+1}  -> 4 loads/ch (row -1 out of bounds).
#   MIDDLE (output rows 1..127): kr in {-1,0,+1} -> 6 loads/ch (row 255 in bounds,
#                                so the last output row is a middle row).
#   The skipped kr=-1 taps in the edge body advance the kernel index by 3 so
#   ch1/ch2 tap bytes stay aligned.
#
# Borders (true pad=1):
#   left  (col 0):  kc=-1 slot0 -> mask slot 1 zeros lane 0.
#   right (col255): kc=+1 slot1 -> mask slot 2 zeros lane 127.
#
# Row-addressed ISA (mb/195): XMEM offset/base operands (ldr_mult_reg,
# ldr_cyclic_mult_reg's offset+base, str_post_aaq_reg) are ROW numbers;
# ldr_cyclic_mult_reg's `index` operand stays an ELEMENT index (0/128 slot
# boundary, unscaled). cr8 used to double as both (XMEM filter-block stride
# AND the literal r_cyclic slot1 index 128) -- split onto cr14 for the
# r_cyclic role, since the two units diverge under row addressing.
# CR (ROW-space; cr14 is the one literal ELEMENT-space const):
#   cr3 KERNEL(row) cr4 MASK(row) cr5 OUTPUT(row) cr6 2(rows) cr7 6(rows)
#   cr8 1(row, filter-block stride) cr9 12(rows) cr10 INPUT(row)
#   cr11 128(rows, loop bound, not XMEM) cr12 16(rows) cr13 TEMP(row)
#   cr14 128 (r_cyclic slot1 ELEMENT index, unscaled)  cr0=0 cr1=1 cr15=dstruct
# LR: lr0=0 lr1=1 lr2=+1 lr3=-1 lr4=kern idx lr5=center row-grp base(2r*768)
#     lr6=out ptr lr7=filter block base lr8=row counter lr9,lr10,lr14=scratch
#     lr11=128 lr12=scratch
{#- ============================ MACROS ============================ -#}
{#- Load channel-row (ch, kr) into both slots.  Center base = lr5. -#}
{%- macro load_ch_kr(ch, kr) %}
{%-   if kr == -1 %}
    sub                 lr9 lr5 cr7;;
{%-   elif kr == 0 %}
    add                 lr9 lr5 lr0;;
{%-   else %}
    add                 lr9 lr5 cr7;;
{%-   endif %}
{%-   if ch >= 1 %}
    add                 lr9 lr9 cr6;;
{%-   endif %}
{%-   if ch == 2 %}
    add                 lr9 lr9 cr6;;
{%-   endif %}
    ldr_cyclic_mult_reg lr9 cr10 lr0;
    add                 lr10 lr9 cr8;;
    ldr_cyclic_mult_reg lr10 cr10 lr11;;
{%- endmacro -%}
{#- Emit the 3 kc taps of one (slot, ch, kr).  lr4 already at this kr's kc=-1 tap. -#}
{%- macro kc_taps(slot) %}
{#- Horizontal edges are zeroed by mask_shift alone (mask slot 0 = all keep,
    CR15 partition P1 -> shift +1 zeros lane 0 = col 0; shift -1 zeros lane 127
    = col 255).  Seam taps (slot0 kc=+1 lane127=col128; slot1 kc=-1 lane0=col127)
    must NOT shift-zero -> shift 0. -#}
{%-   if slot == 0 %}
    SUB                 lr14 lr0 cr1;                # rc_idx = -1 (slot0 kc=-1)
    MULT.RC.VE          lr14 lr4 0 lr2 cr15;         # left edge: shift +1 zeros lane0 (col0)
    acc.add;;
    INC                 lr4 1;
    MULT.RC.VE          lr0 lr4 0 lr0 cr15;          # kc=0, no shift
    acc.add;;
    INC                 lr4 1;
    MULT.RC.VE          lr1 lr4 0 lr0 cr15;          # slot0 kc=+1 (rc_idx=1): seam KEEP, no shift
    acc.add;;
    INC                 lr4 1;;
{%-   else %}
    SUB                 lr14 lr11 cr1;               # rc_idx = 127 (slot1 kc=-1 -> seam nbr col127)
    MULT.RC.VE          lr14 lr4 0 lr0 cr15;         # seam KEEP, no shift
    acc.add;;
    INC                 lr4 1;
    MULT.RC.VE          lr11 lr4 0 lr0 cr15;         # rc_idx = 128 (slot1 kc=0)
    acc.add;;
    INC                 lr4 1;
    ADD                 lr14 lr11 cr1;               # rc_idx = 129 (slot1 kc=+1)
    MULT.RC.VE          lr14 lr4 0 lr3 cr15;         # right edge: shift -1 zeros lane127 (col255)
    acc.add;;
    INC                 lr4 1;;
{%-   endif %}
{%- endmacro -%}
{#- Full filter body for a given kr list (edge=[0,1], middle=[-1,0,1]). -#}
{%- macro filter_body(krlist) %}
{%- for slot in [0, 1] %}
    # ---- SLOT {{slot}} ----
{%-   if slot == 0 %}
    ldr_mult_reg        r0 lr7 cr3;;
{%-   endif %}
    SET                 lr4 cr0;
    MULT.EE             lr4 cr1 0 lr0 cr15;
    acc.add.first;;
    SET                 lr4 cr1;;
{%-   for ch in [0, 1, 2] %}
    # -- channel {{ch}} --
{%-     if krlist == [0, 1] %}
    INC                 lr4 3;;                      # skip kr=-1 taps (advance +3)
{%-     endif %}
{%-     for kr in krlist %}
{{ load_ch_kr(ch, kr) }}
{{ kc_taps(slot) }}
{%-     endfor %}
{%-   endfor %}
    # ACTIVATE.QUANTIZE reads r_acc LIVE and STR_POST_AAQ_REG reads
    # post_aaq_reg LIVE (store dispatches after aaq in the per-cycle order),
    # so they co-issue in one word — no separate store cycle needed.
{%-   if slot == 0 %}
    ACTIVATE.QUANTIZE   identity cr15;
    str_post_aaq_reg    lr0 cr13;;
{%-   else %}
    ACTIVATE.QUANTIZE   identity cr15;
    # XMEM offset for temp1 = +1 ROW (temp0/temp1 are adjacent 128-byte rows);
    # lr11 holds 128 in its r_cyclic-ELEMENT role, so use lr1 (=1) here instead.
    str_post_aaq_reg    lr1 cr13;;
{%-   endif %}
{%- endfor %}

    # ---- decimate + relu + store ----
    ADD                 lr12 lr1 cr1;;               # lr12 = 2 (offset%4=2 -> r_acc start 64)
    ldr_cyclic_mult_reg lr0 cr13 lr0;;
    MULT.RC.VE          lr0 cr1 0 lr0 cr15;;
    acc.stride          64 on off lr0;;
    ldr_cyclic_mult_reg lr1 cr13 lr0;;   # XMEM offset +1 row (temp1); lr11 stays r_cyclic-ELEMENT-only
    MULT.RC.VE          lr0 cr1 0 lr0 cr15;;
    # acc.stride (finalizes r_acc), ACTIVATE.QUANTIZE (reads r_acc LIVE), and
    # STR_POST_AAQ_REG (reads post_aaq_reg LIVE; store dispatches after aaq)
    # co-issue in one word — no separate ACTIVATE/store cycles needed.
    acc.stride          64 on off lr12;
    ACTIVATE.QUANTIZE   relu cr15;
    str_post_aaq_reg    lr6 cr5;;
{%- endmacro -%}

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 lr0 cr0;
    ldr_mult_mask_reg   lr0 cr4;;

    ADD                 lr1 lr0 cr1;                 # +1 (kc=+1 rc_idx for slot0; const)
    ADD                 lr2 lr0 cr1;                 # +1  (mask_shift kc=-1)
    SUB                 lr3 lr0 cr1;;                # -1  (mask_shift kc=+1)

    SET                 lr11 cr14;;                  # 128 (slot1 rc ELEMENT index / temp1 off)

    SET                 lr6 cr0;                     # output pointer = 0
    SET                 lr8 cr0;                     # output row counter = 0
    SET                 lr5 cr0;;                    # center row-group base = 0 (row 0)

# ===========================================================================
# EDGE loop: output row 0 (kr in {0,+1})
# ===========================================================================

    SET                 lr7 cr0;;                    # filter block base = 0

row0_filter_loop:
{{ filter_body([0, 1]) }}
    add                 lr6 lr6 cr8;
    add                 lr7 lr7 cr8;;
    blt                 lr7 cr12 row0_filter_loop;;

    # advance to output row 1
    add                 lr5 lr5 cr9;                 # center row-group += 1536
    INC                 lr8 1;;

# ===========================================================================
# MIDDLE loop: output rows 1..127 (kr in {-1,0,+1})
# ===========================================================================

row_loop:
    SET                 lr7 cr0;;

filter_loop:
{{ filter_body([-1, 0, 1]) }}
    add                 lr6 lr6 cr8;
    add                 lr7 lr7 cr8;;
    blt                 lr7 cr12 filter_loop;;

    add                 lr5 lr5 cr9;
    INC                 lr8 1;;
    blt                 lr8 cr11 row_loop;;

end:
    bkpt;;
