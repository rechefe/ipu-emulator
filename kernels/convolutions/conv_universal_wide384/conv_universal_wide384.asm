{#- conv_universal_wide384.asm -- y = conv2d(x, w, 3x3, stride 1, padding 1), FP32, for widths that are multiples of 128 and >= 384 (unoptimized).

    Wide (W >= 384) standard 3x3 convolution, stride 1.  Correctness-first,
    UNOPTIMIZED (no rotating-slot pipelining): reloads a fresh 3-slot R_CYCLIC
    strip for every (row, col-chunk, filter, channel, kr) combination.

    ``cpr`` (chunks-per-row = W // 128, kernel.toml's [derived] ``cpr``) and
    ``rows`` (the ``height`` parameter) are template values: the harness
    renders the program for each case from them, at ASSEMBLE time (rows must
    be known at assemble time so the row-border sections can be fully
    unrolled with compile-time mask choices -- see below; a fresh binary per
    (rows, cpr) pair, though rows changes the words only at 1 and 2). The
    top and bottom border rows are each unrolled once; the interior rows
    (1..rows-2, which share the same mask choices) are ONE runtime loop, since
    unrolling every row would overflow the 1024-word program memory for all
    but tiny rows. The column-chunk loop (cpr sections, 3 to 5) is fully
    unrolled, as its border masking varies by cc.

    MEMORY LAYOUT (per user spec): channel-interleaved PER SPATIAL ROW; each
    channel's spatial row spans `cpr` consecutive XMEM rows (128 lanes each):
      input  row(r, ic, cc) = (r * in_channels  + ic) * cpr + cc
      output row(r, f,  cc) = (r * out_channels + f)  * cpr + cc

    CORE TECHNIQUE (per-tap 3-slot strip -- generalizes conv_first_layer's
    2-slot 256-wide strip to an ARBITRARY column-chunk boundary without
    needing a whole spatial row resident in R_CYCLIC at once: 3 kr-rows *
    cpr*128 elements would overflow the 512-element ring once cpr>=2). For
    each (channel, kr) the asm loads exactly 3 128-lane chunks into R_CYCLIC
    slots at element index 0/128/256:
      slot A (idx 0)   = chunk (cc-1) of row r+kr   [dummy/clamped at cc==0]
      slot B (idx 128) = chunk  cc    of row r+kr    (the chunk being computed)
      slot C (idx 256) = chunk (cc+1) of row r+kr   [dummy/clamped at cc==cpr-1]
    The 3 kc taps then read fixed-offset 128-element windows within this
    384-element strip -- always the SAME rc_idx regardless of cc, since the
    strip is always built the same way relative to slot B:
      kc=-1: rc_idx = 127  (straddles slot A's last elem .. slot B's first 127)
      kc= 0: rc_idx = 128  (slot B exactly)
      kc=+1: rc_idx = 129  (slot B's elems 1..127 .. slot C's first elem)
    No cross-boundary wraparound trickery is needed since MULT.RC.VE reads
    are NOT slot-aligned (only LD.MULT.CYCLIC writes are); a 128-element
    read starting at rc_idx=127 legitimately straddles slot A/B.

    BORDER MASKING: chosen as an ASSEMBLE-TIME immediate per tap site
    (mask_offset is an immediate operand, not a runtime register) -- both the
    ROW loop (unrolled into 3 sections: row 0 / rows 1..rows-2 / row rows-1,
    mirroring conv_universal's g0/main/gN split) and the COLUMN-CHUNK loop
    (unrolled into cpr sections) make every (row-section, cc, kr, kc) tap
    site's validity known at assemble time, so no runtime border branching is
    needed. Two DIFFERENT kinds of border exist and need DIFFERENT mask
    granularity:
      - Vertical border (kr takes the row out of [0,rows)): the WHOLE
        neighbour row is out of bounds, so every lane of that tap is
        invalid -> mask slot 1 (ZERO all).
      - True horizontal image edge (cc==0's kc=-1, or cc==cpr-1's kc=+1):
        ONLY the single edge output column (lane 0 or lane 127 of that tap)
        actually lacks a neighbour -- every OTHER lane in the same tap is a
        legitimate in-chunk read (the rc_idx=127/129 straddle still covers
        lanes 1..127 / 0..126 correctly; see the per-tap 3-slot strip note
        above). Masking the WHOLE tap here would incorrectly zero 127 valid
        output columns -- this was the first bug found while debugging --
        so a single-LANE mask is required:
        slot 2 (ZERO lane 0 only) / slot 3 (ZERO lane 127 only).
      - Otherwise: mask slot 0 (KEEP all).

    CR registers (all bound in kernel.toml's [abi], plain host-computed
    constants -- no runtime LR multiplication tricks needed):
      CR0  = 0 (read-only const), CR1 = 1 (read-only const), CR15 = the default DS (LOW) and the QF (HIGH)
      {{ cr.x }}  = INPUT_BASE_ROW
      {{ cr.w }}  = KERNEL_BASE_ROW
      {{ cr.y }}  = OUTPUT_BASE_ROW
      {{ cr.mask }}  = MASK_BASE_ROW
      {{ cr.in_channels }}  = in_channels                  (channel-loop bound)
      {{ cr.row_limit }}  = OUTPUT_BASE_ROW + (rows-1)*out_channels*cpr  (out_row_base of the
                                             LAST row; interior-row runtime
                                             loop bound, compared directly
                                             against lr7)
      {{ cr.fpb }}  = FPB                           (14; kernel reload threshold)
      {{ cr.kernel_row_stride }}  = 1                             (rows; per-BLOCK reload advance --
                                             one dense FPB=14 block is exactly
                                             1 XMEM row, so `_reload`'s
                                             `add lr9 lr9 {{ cr.kernel_row_stride }}` always steps by
                                             1 regardless of blocks_per_filter)
      {{ cr.out_row_stride }} = out_channels * cpr            (row-to-row OUTPUT base stride, rows)
      {{ cr.in_row_stride }} = in_channels * cpr             (row-to-row INPUT base stride, rows)
      {{ cr.cpr }} = cpr                           (channel-to-channel INPUT stride
                                             within a row / filter-to-filter
                                             OUTPUT stride within a row, rows)
      {{ cr.total_kernel_rows }} = out_channels * blocks_per_filter  (total kernel-row extent;
                                             filter loop bound, compared
                                             against the kernel-base
                                             accumulator directly so no
                                             separate "filter counter f"
                                             register is needed)
      {{ cr.blocks_per_filter }} = blocks_per_filter             (rows; per-FILTER advance -- split
                                             off {{ cr.kernel_row_stride }} since the two roles
                                             diverge once in_channels > FPB:
                                             advancing to the NEXT FILTER'S
                                             first block must skip ALL of
                                             THIS filter's blocks, whereas
                                             `_reload` only steps by one block
                                             at a time)

    LR allocation:
      lr0  = 0 (const)                  lr1  = 1 (const)
      lr2  = 127 (const, rc_idx kc=-1)  lr3  = 128 (const, rc_idx kc=0 / slot B)
      lr4  = 129 (const, rc_idx kc=+1)  lr5  = 256 (const, slot C index)
      lr6  = row_base     (XMEM row of (r, ic=0, cc=0);   += {{ cr.in_row_stride }} each row)
      lr7  = out_row_base (XMEM row of (r, f=0, cc=0);    += {{ cr.out_row_stride }} each row)
      lr8  = kernel filter base (persistent, f*{{ cr.blocks_per_filter }}; += {{ cr.blocks_per_filter }} once per filter in
             row_section; UNTOUCHED by filter_body except being copied FROM)
      lr9  = kernel working offset for THIS cc (copied from lr8 at cc start;
             the reload path advances lr9 ONLY, never lr8, since each cc
             independently re-walks the same kernel-block sequence)
      lr10 = kernel element index (0..8 within a channel's 9-tap slot; seeded
             to -1 so each tap's word does INC+MULT.RC.VE together -- LR
             executes before MULT in the same cycle, so the mult sees the
             FRESH post-increment index)
      lr11 = ic (channel counter, 0..in_channels-1)
      lr12 = kernel channel-in-block counter (0..FPB-1, resets each block)
      lr13 = PURE SCRATCH inside filter_body (kr-row / slot-address
             arithmetic); holds no value that survives across cc iterations
      lr14 = out ptr for current (f, cc) within this row
      lr15 = ic_row_base (row_base + ic*cpr, threaded incrementally: +={{ cr.cpr }}
             each channel iteration; reset to lr6 at the start of each cc's
             channel loop)

      Per-tap slot-address scratch: lr13 holds the kr-row base for the
      current channel (= lr15 +/- {{ cr.in_row_stride }}); the per-slot (cc-1/cc/cc+1) absolute
      row is then computed via INC-load-DEC on lr13 itself, but split across
      SEPARATE VLIW words (never write the SAME lr twice in one word -- the
      ISA has no defined behaviour for that) so each word writes lr13 at
      most once.
-#}
{%- set CPR = cpr -%}{#- [derived] cpr = width / 128 -#}
{%- set ROWS = height -%}
{#- ======================= shared per-(row,cc) filter body ================ -#}
{%- macro filter_body(row_id, cc, kr_top_valid, kr_bot_valid) %}
{%-   set kc_neg_valid = (cc != 0) %}
{%-   set kc_pos_valid = (cc != CPR - 1) %}
{%-   set lbl = 'r' ~ row_id ~ '_cc' ~ cc %}
    # The kernel does NOT depend on cc (same weights for every column-chunk),
    # so each cc independently re-walks the SAME kernel-block sequence from
    # the start of THIS FILTER's blocks: lr8 (persistent, f*cr14, maintained by
    # row_section across the filter loop, untouched here) is copied into the
    # per-cc working register lr9; the RELOAD path below advances lr9 only,
    # never lr8, so the next cc still starts this filter's walk from block 0.
    ADD                 LR9 LR8 CR0;
    LD.MULT        R0 LR9 {{ cr.w }};;
    SET                 LR12 CR0;;                          # kernel channel-in-block ctr = 0

    SET                 LR10 CR0;
    MULT.EE             LR10 CR0 0 LR0 CR15.L CR15.H;             # 0 * R0[0], broadcast
    ACC.ADD.FIRST;;                                       # r_acc = 0 (seed; conv taps add on top)

    SET                 LR11 CR0;                          # ic = 0
    ADD                 LR15 LR6 CR0;;                      # ic_row_base = row_base (ic=0)

    DEC                 LR10 1;;                            # kernel elem idx seeded to -1: each
                                                              # tap's word does INC+MULT.RC.VE
                                                              # together (LR executes before MULT,
                                                              # so the mult sees the FRESH index).

{{ lbl }}_ch_loop:
    BLT.S                 LR12 {{ cr.fpb }} {{ lbl }}_ch_body;;
    B                   {{ lbl }}_reload;;

{{ lbl }}_reload:
    ADD                 LR9 LR9 {{ cr.kernel_row_stride }};
    LD.MULT        R0 LR9 {{ cr.w }};
    SET                 LR12 CR0;
    SET                 LR10 CR0;;
    DEC                 LR10 1;;                            # re-seed kernel elem idx to -1 for the new block
    B                   {{ lbl }}_ch_body;;

{{ lbl }}_ch_body:
{%-   for kr in [-1, 0, 1] %}
{%-     if kr == -1 %}
{%-       set kr_valid = kr_top_valid %}
{%-     elif kr == 1 %}
{%-       set kr_valid = kr_bot_valid %}
{%-     else %}
{%-       set kr_valid = true %}
{%-     endif %}
    # ---- kr = {{ kr }} ({{ 'valid' if kr_valid else 'BORDER (masked)' }}) ----
{%-     if kr == -1 and kr_valid %}
    SUB                 LR13 LR15 {{ cr.in_row_stride }};;
{%-     elif kr == 1 and kr_valid %}
    ADD                 LR13 LR15 {{ cr.in_row_stride }};;
{%-     else %}
    ADD                 LR13 LR15 CR0;;
{%-     endif %}
    # lr13 = XMEM row of (row r+kr, this channel, cc=0) -- reused here as pure
    # scratch (its "filter counter f" value lives in row_section and is not
    # touched again until AFTER the whole cc loop finishes). Add cc to reach
    # this column-chunk (cc constant, folded into literal adds below).

    # slot B: this col-chunk cc.  lr13 is bumped to the exact chunk row in its
    # OWN word first (INC), then read in a separate word (load), then
    # restored in a separate word (DEC) -- lr13 is never written twice in one
    # VLIW word.
{%-     if cc == 0 %}
    LD.MULT.CYCLIC LR13 {{ cr.x }} LR3;;
{%-     else %}
    INC                 LR13 {{ cc }};;
    LD.MULT.CYCLIC LR13 {{ cr.x }} LR3;;
    DEC                 LR13 {{ cc }};;
{%-     endif %}
    # slot A: col-chunk cc-1 (dummy at cc==0, that tap is masked anyway)
{%-     if cc == 0 %}
    LD.MULT.CYCLIC LR13 {{ cr.x }} LR0;;
{%-     else %}
    INC                 LR13 {{ cc - 1 }};;
    LD.MULT.CYCLIC LR13 {{ cr.x }} LR0;;
    DEC                 LR13 {{ cc - 1 }};;
{%-     endif %}
    # slot C: col-chunk cc+1 (dummy at cc==cpr-1, that tap is masked anyway)
{%-     if cc == CPR - 1 %}
    LD.MULT.CYCLIC LR13 {{ cr.x }} LR5;;
{%-     else %}
    INC                 LR13 {{ cc + 1 }};;
    LD.MULT.CYCLIC LR13 {{ cr.x }} LR5;;
    DEC                 LR13 {{ cc + 1 }};;
{%-     endif %}

{%-     for kc in [-1, 0, 1] %}
{%-       if kc == -1 %}
{%-         set kc_valid = kc_neg_valid %}
{%-         set rc_reg = 'lr2' %}
{%-       elif kc == 1 %}
{%-         set kc_valid = kc_pos_valid %}
{%-         set rc_reg = 'lr4' %}
{%-       else %}
{%-         set kc_valid = true %}
{%-         set rc_reg = 'lr3' %}
{%-       endif %}
{#-       Mask choice: kr invalid (vertical border) zeroes the WHOLE tap
          (slot 1) regardless of kc, since the whole neighbour ROW is out of
          bounds. kr valid but kc invalid means only the SINGLE true image
          edge LANE is out of bounds (lane 0 for kc=-1 at cc==0, lane 127 for
          kc=+1 at cc==CPR-1) -- every OTHER lane in that same tap is a
          legitimate in-chunk neighbour (see the per-tap 3-slot strip note
          above: rc_idx=127/129 still correctly addresses lanes 1..127 /
          0..126 within this chunk). Slots 2/3 zero only that one lane. -#}
{%-       if not kr_valid %}
{%-         set mask_slot = 1 %}
{%-       elif kc_valid %}
{%-         set mask_slot = 0 %}
{%-       elif kc == -1 %}
{%-         set mask_slot = 2 %}
{%-       else %}
{%-         set mask_slot = 3 %}
{%-       endif %}
    MULT.RC.VE          {{ rc_reg }} LR10 {{ mask_slot }} LR0 CR15.L CR15.H;
    INC                 LR10 1;
    ACC.ADD;;
{%-     endfor %}
{%-   endfor %}

    ADD                 LR15 LR15 {{ cr.cpr }};
    INC                 LR11 1;
    INC                 LR12 1;;
    BLT.S                 LR11 {{ cr.in_channels }} {{ lbl }}_ch_loop;;

    ACTIVATE.EXACT   identity CR15.L {{ cr.y }} LR14;;
    INC                 LR14 1;;
{%- endmacro -%}

{#- ============== per-row-section: cc loop wrapping filter_body ============ -#}
{#- NOTE: there is no dedicated "filter counter f" register -- lr13 is used
    purely as scratch inside filter_body (see its kr-row-address role), so
    the filter loop instead terminates by comparing the persistent kernel
    base lr8 (= f * {{ cr.blocks_per_filter }}) against {{ cr.total_kernel_rows }} (= out_channels * blocks_per_filter, the total
    kernel-row extent), which advances in lockstep with f and needs no
    dedicated counter register of its own. -#}
{%- macro row_section(row_id, kr_top_valid, kr_bot_valid) %}
    ADD                 LR14 LR7 CR0;;                      # out ptr = out_row_base (f=0, cc=0)

    SET                 LR8 CR0;;                            # kernel filter base = 0 (f=0)

r{{ row_id }}_filter_loop:
{%-   for cc in range(CPR) %}
{{ filter_body(row_id, cc, kr_top_valid, kr_bot_valid) }}
{%-   endfor %}

    ADD                 LR8 LR8 {{ cr.blocks_per_filter }};;
    BLT.S                 LR8 {{ cr.total_kernel_rows }} r{{ row_id }}_filter_loop;;
{%- endmacro -%}
{#- Interior-row body is IDENTICAL for every row 1..ROWS-2 (kr_top_valid=
    kr_bot_valid=true, i.e. no vertical border), so it is emitted ONCE as a
    genuine runtime loop, instead of being unrolled per row like row_section
    above. This keeps the assembled program within INST_MEM_SIZE (1024
    instructions) for larger `rows` -- only `cpr` (small, a handful of
    column-chunks) is unrolled at compile time; `rows` is NOT, since border
    masking only differs for the first/last row. The loop re-purposes lr7
    (out_row_base, already advanced by {{ cr.out_row_stride }} every row) as its own trip
    counter -- no dedicated counter register needed -- comparing it against
    {{ cr.row_limit }} (= out_row_base of the LAST row, set by the harness), so the loop
    naturally stops right before the bottom-border row. -#}
{%- macro interior_row_loop() %}
main_row_loop:
    ADD                 LR14 LR7 CR0;;                      # out ptr = out_row_base (f=0, cc=0)

    SET                 LR8 CR0;;                            # kernel filter base = 0 (f=0)

main_filter_loop:
{%-   for cc in range(CPR) %}
{{ filter_body('main', cc, true, true) }}
{%-   endfor %}

    ADD                 LR8 LR8 {{ cr.blocks_per_filter }};;
    BLT.S                 LR8 {{ cr.total_kernel_rows }} main_filter_loop;;

    ADD                 LR6 LR6 {{ cr.in_row_stride }};
    ADD                 LR7 LR7 {{ cr.out_row_stride }};;
    BLT.S                 LR7 {{ cr.row_limit }} main_row_loop;;
{%- endmacro -%}

# ===========================================================================
# Initialization
# ===========================================================================

    SET                 LR0 CR0;
    LD.MASK   LR0 {{ cr.mask }};;

    ADD                 LR1 LR0 CR1;;                      # 1

    INC                 LR2 127;;                           # 127 (rc_idx kc=-1)
    INC                 LR3 128;;                           # 128 (rc_idx kc=0 / slot B index)
    INC                 LR4 129;;                           # 129 (rc_idx kc=+1)
    INC                 LR5 128;;                           # 256 (slot C index), 2 words --
    INC                 LR5 128;;                           # INC/DEC immediates are limited to
                                                              # [0,255], and lr5 can't be written
                                                              # twice in one VLIW word.

    # row_base/out_row_base are RELATIVE offsets from cr2/cr4 (added explicitly
    # as `base` by every ld.mult.cyclic/activate.exact below) -- NOT
    # absolute row numbers -- so they start at 0, mirroring lr8 (kernel filter
    # base)'s relative-to-cr3 convention. Seeding them from cr2/cr4 directly
    # would double-count the base at every load/store site.
    SET                 LR6 CR0;                            # row_base = 0 (relative to cr2)
    SET                 LR7 CR0;;                            # out_row_base = 0 (relative to cr4)

{% if ROWS == 1 %}
{{ row_section(0, false, false) }}
{% else %}
# ---- row 0 (top border: kr=-1 masked) ----
{{ row_section(0, false, true) }}

    ADD                 LR6 LR6 {{ cr.in_row_stride }};
    ADD                 LR7 LR7 {{ cr.out_row_stride }};;

{%   if ROWS > 2 %}
# ---- interior rows 1 .. {{ ROWS - 2 }} (genuine runtime loop -- see
#      interior_row_loop's doc comment for why this is NOT unrolled per row) ----
{{ interior_row_loop() }}
{%   endif %}

# ---- row {{ ROWS - 1 }} (bottom border: kr=+1 masked) ----
{{ row_section(ROWS - 1, true, false) }}
{% endif %}

end:
    HALT;;
