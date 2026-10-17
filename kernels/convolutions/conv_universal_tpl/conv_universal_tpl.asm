{#- conv_universal_tpl.asm -- y = act(conv2d(x, w, stride 1, padding 1) + b): a 3x3 "same" convolution in FP32 wide vectors, with an optional folded bias and identity or ReLU.

    The template of kernels/convolutions/registry.toml's conv_universal
    (has_bias = false, apply_relu = false) and conv_universal_bn_activation
    (apply_relu = true; has_bias either: a query without one runs with a zero
    bias). Batch-norm is assumed folded into the weights and the bias, so no
    separate BN step runs. The program reads the query's own parameters:
    "seeded" (has_bias or apply_relu) selects the bias seed, apply_relu the
    activation. The combination has_bias without ReLU renders the seeded
    identity program, which no entry routes to yet (E07-20 adds it).

    Pipeline per output filter
    - R_ACC is seeded by MULT.EE + ACC.ADD.FIRST: R0[0] times CR1 (the bias
      element, seeded) or times CR0 (a zero seed).
    - += the 9 taps of every input channel (MULT.RC.VE from the cyclic ring).
    - ACTIVATE.QUANTIZE (relu when apply_relu, else identity) and ST.AAQ store
      128 elements per chunk, co-issued in one word (AAQ reads R_ACC live,
      ST.AAQ reads the AAQ result live).
    The variants differ in exactly three places, each in the 3 sections
    (chunk 0, chunks 1..N-2, chunk N-1): the seed register, the kernel-index
    preset (DEC to -1 unseeded, 0 seeded, so tap 1's INC lands on the first
    weight), and the activation.

    Filter super-blocks (pack "fpb28_filters", input filters [out, 9 * in + 1]: the weights, then the bias)
    - One 256-element super-block (2 XMEM rows: R0 half, R1 half) holds up
      to 28 input channels of one filter, 9 taps each, in tap order
      kr = -1, 0, +1 and kc = -1, 0, +1 within a row (row-major from the
      torch [out, in, 3, 3] weight, no reordering).
    - Unseeded: channel s at elements [9s, 9s + 9); 28 * 9 = 252 <= 256.
      Seeded: element 0 is the filter's bias, channel s at
      [1 + 9s, 1 + 9s + 9); 1 + 28 * 9 = 253 <= 256. The bias element is
      replicated into every super-block of a filter so the weight index is
      the same in every block; it is accumulated once per filter, from
      super-block 0 (the reload paths do not re-seed).
    - Channels 0..13 land in R0, 14..27 in R1; MULT.VE's shared index
      (0..255) addresses all 28. Per-filter row stride:
      ceil(in_channels / 28) * 2 rows (kernel_rows = out_channels times that).

    Data layout (packs "chunked_guard", "chunked")
    - width pads to cols in {16, 32, 64, 128}: one packed spatial row per
      mask-partition group, so CR15.partition = 128 / cols groups (P0 at
      cols = 128). height pads to rows >= ceil(256 / cols): at least 2
      chunks, because the program has a first and a last chunk section.
    - 128 / cols spatial rows share one 128-element chunk (one XMEM row);
      chunks are grouped by channel: row (chunk * channels + ch).
    - The input region starts with in_channels zero rows (one group stride
      of headroom), so the chunk-0 kr = -1 prefetch, at offset
      lr_chunk_base - in_channels, bottoms out at the region base instead of
      underflowing. lr_chunk_base starts at in_channels, and chunk_limit is
      pre-shifted by the same stride: (num_chunks - 1) * in_channels +
      in_channels.

    Units
    - x, filters, border_mask, y, in_channels (the group stride), fpb,
      kernel_rows, chunk_limit and sb_rows are XMEM rows. One chunk is one
      row, so the row-space chunk stride is the hardwired CR1.
    - slot_stride (128), ring_adv (384) and walk_step are R_CYCLIC ELEMENT
      offsets: the ring is 512 elements in every format, so they are
      mode-blind and must not be scaled.
    - Ring rotation: lr_read (the kr = 0 slot) advances by -128
      (= +384 mod 512) per channel. walk_step = 384 - 2 * cols - 2 brings
      lr_walk from this channel's tap 9 (lr_read + cols + 1) to the next
      channel's tap 1 ((lr_read - 128) - cols - 1).

    Border masks (scratch border_mask, one row, ../masks.py:border_mask)
    - One R_MASK blob, loaded once at init, carries three 16-byte slots; a
      bit of 1 KEEPS the lane. Slot 0 keeps every lane (interior and kr = 0
      taps), slot 3 zeroes packed row 0 (chunk-0 kr = -1 taps), slot 6
      zeroes the last packed row (chunk N-1 kr = +1 taps). No mid-program
      R_MASK reload. The blob is a bit mask, the same in every format; only
      its row address scales.
    - Left/right edge columns are zeroed by mask_shift, not by slots, with
      CR15.partition = cols groups: kc = -1 shifts by +1 (lr_shift_p1,
      zeroes each row's first column), kc = 0 by 0, kc = +1 by -1
      (lr_shift_m1, zeroes each row's last column).

    Cyclic ring and walking pointer
    - The 512-element R_CYCLIC holds three vertically neighbouring chunks of
      one input channel: kr = -1 at lr_read - 128, kr = 0 at lr_read,
      kr = +1 at lr_read + 128 (mod 512).
    - lr_walk walks the 9 MULT.RC.VE taps: lr_read - cols - 1, - cols,
      - cols + 1, - 1, + 0, + 1, + cols - 1, + cols, + cols + 1. At tap K the
      LR slot writes lr_walk = (tap K-1's offset) + step and MULT.RC.VE reads
      it live. Steps: tap 1 + walk_step; taps 4 and 7 + lr_cols_m2
      (cols - 2); the other taps + 1.
    - Load schedule, each load in the same word as the mult that needs it:
      tap 1 loads this channel's kr = +1 (chunk 0) or kr = -1; tap 4 loads
      the next channel's kr = -1 / kr = 0 one iteration ahead; tap 7 loads
      the next channel's kr = 0 / kr = +1 into this channel's dead kr = -1
      slot. lr_off_zero holds this channel's kr = 0 row (lr_chunk_base +
      lr_ch_ctr) and is bumped at tap 2.

    LR allocation (the lr_* aliases below)
      LR0 = 0, LR1 = cols - 2, LR2 lr_off_zero, LR3 lr_walk, LR4 lr_read,
      LR5 lr_write, LR6 kernel element index, LR7 output row, LR8 chunk base
      row, LR9 = -1 (mask shift), LR10 channel row counter, LR11 clamp limit,
      LR12 super-block row offset, LR13 = +1 (mask shift), LR14 scratch.
      LR15 is free.

    Supported configurations: width 1..128 (wider: conv_universal_wide384),
    any height >= 1 (padded), in_channels >= 1, out_channels >= 1.
    KNOWN LATENT BUG (not fixed here; D3): num_chunks = rows * cols // 128
    rounds down while the input pack writes ceil(rows / (128 / cols)) chunks
    of every channel, so a shape with rows * cols not a multiple of 128
    (height 17 at width 16, height 9 at width 32, height 5 at width 64)
    computes only num_chunks chunks (its last rows are never written), and
    the old ipu_apps harness, which sized the input region for num_chunks,
    overran it into the filters. No case uses such a shape.
-#}
{%- set lr_zero = "LR0" -%}
{%- set lr_cols_m2 = "LR1" -%}
{%- set lr_off_zero = "LR2" -%}
{%- set lr_walk = "LR3" -%}
{%- set lr_read = "LR4" -%}
{%- set lr_write = "LR5" -%}
{%- set lr_kern_idx = "LR6" -%}
{%- set lr_out_ptr = "LR7" -%}
{%- set lr_chunk_base = "LR8" -%}
{%- set lr_shift_m1 = "LR9" -%}
{%- set lr_ch_ctr = "LR10" -%}
{%- set lr_clamp = "LR11" -%}
{%- set lr_sb_off = "LR12" -%}
{%- set lr_shift_p1 = "LR13" -%}
{%- set lr_scratch = "LR14" -%}
{#- The variants: seeded (has_bias or apply_relu) seeds R_ACC from the bias
    element; apply_relu picks the activation. -#}
{%- if has_bias or apply_relu %}
{%- set cr_seed = "CR1" -%}
{%- set kern_idx_init = "ADD " ~ lr_kern_idx ~ " " ~ lr_zero ~ " " ~ lr_zero -%}
{%- else %}
{%- set cr_seed = "CR0" -%}
{%- set kern_idx_init = "DEC " ~ lr_kern_idx ~ " 1" -%}
{%- endif %}
{%- set activation = "relu" if apply_relu else "identity" -%}
# ===========================================================================
# Initialization
# ===========================================================================

    SET                 {{ lr_zero }} CR0;
    LD.MASK   {{ lr_zero }} {{ cr.border_mask }};;

    ADD                 {{ lr_cols_m2 }} {{ lr_zero }} {{ cr.cols }};        # {{ lr_cols_m2 }} = cols (temp)
    # lr_chunk_base starts at ONE GROUP STRIDE, not 0: the input data is placed that far
    # above the base of x so the g0 kr=-1 prefetch (lr_chunk_base - in_channels) bottoms out at
    # row 0 instead of underflowing.  chunk_limit is pre-shifted to match.
    ADD                 {{ lr_chunk_base }} {{ lr_zero }} {{ cr.in_channels }};
    SET                 {{ lr_out_ptr }} CR0;;

    DEC                 {{ lr_cols_m2 }} 2;                           # {{ lr_cols_m2 }} = cols - 2 (permanent)
    ADD                 {{ lr_shift_p1 }} {{ lr_zero }} CR1;         # {{ lr_shift_p1 }} = +1 (mask_shift for kc=-1, permanent)
    SUB                 {{ lr_shift_m1 }} {{ lr_zero }} CR1;;        # {{ lr_shift_m1 }} = -1 (mask_shift for kc=+1, permanent)

# ===========================================================================
# Section 1: Chunk 0 (top border) — the kr=-1 row is masked (mask slot 3).
# ===========================================================================

    SET                 {{ lr_sb_off }} CR0;;

g0_filter_loop:
    LD.MULT        R0 {{ lr_sb_off }} {{ cr.filters }};
    SET                 {{ lr_kern_idx }} CR0;;

    SET                 {{ lr_ch_ctr }} CR0;
    MULT.EE             {{ lr_kern_idx }} {{ cr_seed }} 0 {{ lr_zero }} CR15;   # R0[0] * seed (bias or 0), broadcast
    ACC.ADD.FIRST;;                                     # R_ACC = seed (bias or 0; conv taps add on top)

    ADD                 {{ lr_scratch }} {{ lr_sb_off }} CR1;
    ADD                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr.fpb }};
    LD.MULT        R1 {{ lr_scratch }} {{ cr.filters }};;
    BLT.S                 {{ cr.in_channels }} {{ lr_clamp }} g0_clamp;;
    B                   g0_ch_loop;;
g0_clamp:
    ADD                 {{ lr_clamp }} {{ lr_zero }} {{ cr.in_channels }};;

g0_ch_loop:
    # Preamble: load FIRST-ch-of-block's kr=0 (slot 0) and kr=+1 (slot 128).
    # Its kr=-1 will be loaded same-cycle at the body's iter-0 tap 1 (g0:
    # masked via slots 4/3/5).  Seed walking-pointer state.
    #
    # Note: lr_ch_ctr is NOT reset here.  filter_loop / g0_filter_loop initialise
    # lr_ch_ctr = 0 for a fresh filter; reload paths keep lr_ch_ctr at its end-of-body
    # value so the preamble loads the correct channel within the next FPB
    # super-block.

    # Cy 1: lr_off_zero = E0 = ch0 kr=0 ext = lr_chunk_base+lr_ch_ctr; load ch0 kr=0 → slot 0.
    SET                 {{ lr_read }} CR0;
    SET                 {{ lr_write }} CR0;
    ADD                 {{ lr_off_zero }} {{ lr_chunk_base }} {{ lr_ch_ctr }};
    LD.MULT.CYCLIC {{ lr_off_zero }} {{ cr.x }} {{ lr_write }};;

    # Cy 2: lr_write = 384; load ch0 kr=-1 (ext = E0 - in_channels) → slot 384 (masked by slot 3
    #     in body, value don't-care).  lr_walk = -384.
    ADD                 {{ lr_write }} {{ lr_zero }} {{ cr.ring_adv }};
    SUB                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    SUB                 {{ lr_walk }} {{ lr_zero }} {{ cr.ring_adv }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};;

    # Cy 3: lr_walk = -384 + 1 = -383.
    ADD                 {{ lr_walk }} {{ lr_walk }} CR1;;

    # Cy 4: lr_walk = cols - 383.  lr_write := lr_read+128 = 128.  lr_kern_idx := -1.
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ cr.cols }};
    ADD                 {{ lr_write }} {{ lr_zero }} {{ cr.slot_stride }};
    {{ kern_idx_init }};;                        # tap 1 INC lands on the first weight (past the bias when seeded)

    B                   g0_tap_body;;

g0_tap_body:
    # 9 cyc/ch body — role-rotating ring (see mn_tap_body).  G0 specialization: kr=-1
    # taps (1-3) mask the top row via mask slot 3.  Loads/lr_write identical to mn.
    # All channels in chunk 0 have kr=-1 out of bounds (masked), so the kr=-1 data is
    # don't-care.  THIS ch kr=+1 (real) is loaded at tap 1, read at taps 7-9.

    # --- tap 1: kr=-1 kc=-1.  slot 3 zeros row 0; kc=-1 shift.  Load THIS ch kr=+1
    #     (ext = lr_off_zero + in_channels) into the kr=+1 slot (lr_write = lr_read+128).
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ cr.walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 2: kr=-1 kc=0.  slot 3, no shift.  Bump lr_off_zero to E'.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_off_zero }} {{ lr_off_zero }} CR1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 3: kr=-1 kc=+1.  slot 3 + kc=+1 shift.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 3 {{ lr_shift_m1 }} CR15;
    ACC.ADD;;

    # --- tap 4: kr=0 kc=-1.  Prefetch NEXT kr=-1 (ext = E' - in_channels) into the SPARE slot.
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    SUB                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 5.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 6.  lr_ch_ctr += CR1 (the channel row counter).
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} CR1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} CR15;
    ACC.ADD;;

    # --- tap 7: kr=+1 kc=-1.  Prefetch NEXT kr=0 (ext = E') into THIS dead kr=-1 slot.
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    LD.MULT.CYCLIC {{ lr_off_zero }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 8.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 9: lr_write += 128 -> next iter tap-1 slot.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} CR15;
    ACC.ADD;
    BLT.S                 {{ lr_ch_ctr }} {{ lr_clamp }} g0_tap_body;;

    # Advance kernel offset; check for more channel groups.
    ADD                 {{ lr_sb_off }} {{ lr_sb_off }} {{ cr.sb_rows }};
    BLT.S                 {{ lr_ch_ctr }} {{ cr.in_channels }} g0_reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    # ACTIVATE.QUANTIZE reads R_ACC LIVE and ST.AAQ reads
    # post_aaq_reg LIVE (store dispatches after aaq in the per-cycle order),
    # so they co-issue in one word — no separate store cycle needed.
    ACTIVATE.QUANTIZE {{ activation }} CR15;
    ST.AAQ {{ lr_out_ptr }} {{ cr.y }};;

    ADD                 {{ lr_out_ptr }} {{ lr_out_ptr }} CR1;
    BLT.S                 {{ lr_sb_off }} {{ cr.kernel_rows }} g0_filter_loop;;

    B                   main_setup;;

g0_reload:
    LD.MULT        R0 {{ lr_sb_off }} {{ cr.filters }};
    SET                 {{ lr_kern_idx }} CR0;;

    ADD                 {{ lr_scratch }} {{ lr_sb_off }} CR1;
    ADD                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr.fpb }};
    LD.MULT        R1 {{ lr_scratch }} {{ cr.filters }};;
    BLT.S                 {{ cr.in_channels }} {{ lr_clamp }} g0_reload_clamp;;
    B                   g0_ch_loop;;
g0_reload_clamp:
    ADD                 {{ lr_clamp }} {{ lr_zero }} {{ cr.in_channels }};
    B                   g0_ch_loop;;

# ===========================================================================
# Section 2: Chunks 1 .. N-2 (main loop) — all rows real from x.
# ===========================================================================

main_setup:
    # Chunk 1 = guard (in_channels) + one chunk stride (in_channels) = 2 * in_channels under the guard bias.
    # lr_chunk_base still holds the guard seed (in_channels) from init, so one more + in_channels gets there.
    ADD                 {{ lr_chunk_base }} {{ lr_chunk_base }} {{ cr.in_channels }};;
    BLT.S                 {{ lr_chunk_base }} {{ cr.chunk_limit }} row_loop;;

    B                   gN_section;;

row_loop:
    SET                 {{ lr_sb_off }} CR0;;

filter_loop:
    LD.MULT        R0 {{ lr_sb_off }} {{ cr.filters }};
    SET                 {{ lr_kern_idx }} CR0;;

    SET                 {{ lr_ch_ctr }} CR0;
    MULT.EE             {{ lr_kern_idx }} {{ cr_seed }} 0 {{ lr_zero }} CR15;   # R0[0] * seed (bias or 0), broadcast
    ACC.ADD.FIRST;;                                     # R_ACC = seed (bias or 0; conv taps add on top)

    ADD                 {{ lr_scratch }} {{ lr_sb_off }} CR1;
    ADD                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr.fpb }};
    LD.MULT        R1 {{ lr_scratch }} {{ cr.filters }};;
    BLT.S                 {{ cr.in_channels }} {{ lr_clamp }} mn_clamp;;
    B                   ch_loop;;
mn_clamp:
    ADD                 {{ lr_clamp }} {{ lr_zero }} {{ cr.in_channels }};;

ch_loop:
    # Preamble (9-cyc role scheme): load ch0's kr=0 (slot0) and kr=-1 (slot3); ch0's
    # kr=+1 is loaded by the body at tap 1.  lr_read=0; lr_write enters body = 128
    # (= lr_read+128, ch0 kr=+1 slot1).  lr_off_zero = E0 (ch0 kr=0 ext, NOT bumped:
    # the body bumps it at tap 2).  walk seed = cols-383 (for -128 rotation, walk_step=384-2cols-2).

    # Cy 1: lr_off_zero = E0 = ch0 kr=0 ext = lr_chunk_base+lr_ch_ctr; load ch0 kr=0 → slot 0.
    SET                 {{ lr_read }} CR0;
    SET                 {{ lr_write }} CR0;
    ADD                 {{ lr_off_zero }} {{ lr_chunk_base }} {{ lr_ch_ctr }};
    LD.MULT.CYCLIC {{ lr_off_zero }} {{ cr.x }} {{ lr_write }};;

    # Cy 2: lr_write = 384 (= ring_adv); ext = E0 - in_channels (ch0 kr=-1) → slot 384.
    #     lr_walk = -384 (= -ring_adv; toward cols-383 via +1 then +cols below).
    ADD                 {{ lr_write }} {{ lr_zero }} {{ cr.ring_adv }};
    SUB                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    SUB                 {{ lr_walk }} {{ lr_zero }} {{ cr.ring_adv }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};;

    # Cy 3: lr_walk = -384 + 1 = -383.
    ADD                 {{ lr_walk }} {{ lr_walk }} CR1;;

    # Cy 4: lr_walk = cols - 383 (tap-1 seed).  lr_write := lr_read+128 = 128.
    #     lr_kern_idx := -1 (tap 1's INC brings it to 0).  lr_off_zero stays = E0.
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ cr.cols }};
    ADD                 {{ lr_write }} {{ lr_zero }} {{ cr.slot_stride }};
    {{ kern_idx_init }};;                        # tap 1 INC lands on the first weight (past the bias when seeded)

    B                   mn_tap_body;;

mn_tap_body:
    # 9 cyc/ch — all loads use x.  Role-rotating 4-slot ring; lr_read (kr=0 slot)
    # advances -128 (= +384 mod 512) per channel.  The walk is UNCHANGED from the
    # 10-cyc version (cols-stride, masked straddle).  Slot lifetimes (user): kr=0 slot
    # live ALL 9 taps (NEVER a load target); kr=-1 slot live taps 1-3; kr=+1 slot live
    # taps 7-9.  Per loop the 3 loads NEVER touch the kr=0 slot:
    #   tap 1: load THIS ch kr=+1 (read taps 7-9 this loop) -> kr=+1 slot (lr_read+128).
    #   tap 4: prefetch NEXT ch kr=-1 -> the SPARE slot (lr_read+256).
    #   tap 7: prefetch NEXT ch kr=0  -> THIS dead kr=-1 slot (lr_read+384).
    # lr_write threads a uniform +128 mod 512 chain (enter = lr_read+128).  lr_read is
    # dead (chain self-sustains).  lr_off_zero holds THIS kr=0 ext at tap 1 (for THIS
    # kr=+1 = +in_channels), then is bumped to E' (NEXT kr=0 ext) at tap 2 for taps 4/7.

    # --- tap 1: kr=-1 kc=-1.  Load THIS ch kr=+1 (ext = lr_off_zero + in_channels) into the
    #     kr=+1 slot (lr_write = lr_read+128).  Read by taps 7-9 this loop (snapshot OK).
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ cr.walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 2.  Bump lr_off_zero to E' = NEXT ch kr=0 ext (for taps 4/7 prefetch).
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_off_zero }} {{ lr_off_zero }} CR1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 3.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} CR15;
    ACC.ADD;;

    # --- tap 4: kr=0 kc=-1.  Prefetch NEXT kr=-1 (ext = E' - in_channels) into the SPARE slot
    #     (lr_write = lr_read+256 after the tap-3 +128).
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    SUB                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 5.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 6.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} CR1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} CR15;
    ACC.ADD;;

    # --- tap 7: kr=+1 kc=-1.  Prefetch NEXT kr=0 (ext = E' = lr_off_zero LIVE) into
    #     THIS dead kr=-1 slot (lr_write = lr_read+384 after the tap-5 +128).
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    LD.MULT.CYCLIC {{ lr_off_zero }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 8.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 9: lr_write += 128 -> next iter tap-1 slot (= next lr_read + 128).
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} CR15;
    ACC.ADD;
    BLT.S                 {{ lr_ch_ctr }} {{ lr_clamp }} mn_tap_body;;

mn_after_block:
    ADD                 {{ lr_sb_off }} {{ lr_sb_off }} {{ cr.sb_rows }};
    BLT.S                 {{ lr_ch_ctr }} {{ cr.in_channels }} reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    # ACTIVATE.QUANTIZE reads R_ACC LIVE and ST.AAQ reads
    # post_aaq_reg LIVE (store dispatches after aaq in the per-cycle order),
    # so they co-issue in one word — no separate store cycle needed.
    ACTIVATE.QUANTIZE {{ activation }} CR15;
    ST.AAQ {{ lr_out_ptr }} {{ cr.y }};;

    ADD                 {{ lr_out_ptr }} {{ lr_out_ptr }} CR1;
    BLT.S                 {{ lr_sb_off }} {{ cr.kernel_rows }} filter_loop;;

    ADD                 {{ lr_chunk_base }} {{ lr_chunk_base }} {{ cr.in_channels }};;
    BLT.S                 {{ lr_chunk_base }} {{ cr.chunk_limit }} row_loop;;

    B                   gN_section;;

reload:
    LD.MULT        R0 {{ lr_sb_off }} {{ cr.filters }};
    SET                 {{ lr_kern_idx }} CR0;;

    ADD                 {{ lr_scratch }} {{ lr_sb_off }} CR1;
    ADD                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr.fpb }};
    LD.MULT        R1 {{ lr_scratch }} {{ cr.filters }};;
    BLT.S                 {{ cr.in_channels }} {{ lr_clamp }} mn_reload_clamp;;
    B                   ch_loop;;
mn_reload_clamp:
    ADD                 {{ lr_clamp }} {{ lr_zero }} {{ cr.in_channels }};
    B                   ch_loop;;

# ===========================================================================
# Section 3: Last chunk (bottom border) — kr=+1 out-of-bounds row zeroed by
#            mask slot 6 (no R_MASK reload; one blob carries all 3 slots).
# ===========================================================================

gN_section:
    # No R_MASK reload: the single blob loaded at init carries slot 0 (none),
    # slot 3 (top-row zero, used by g0), and slot 6 (bottom-row zero, used here).
    # gN's kr=+1 taps select slot 6 directly.
    SET                 {{ lr_sb_off }} CR0;;

gN_filter_loop:
    LD.MULT        R0 {{ lr_sb_off }} {{ cr.filters }};
    SET                 {{ lr_kern_idx }} CR0;;

    SET                 {{ lr_ch_ctr }} CR0;
    MULT.EE             {{ lr_kern_idx }} {{ cr_seed }} 0 {{ lr_zero }} CR15;   # R0[0] * seed (bias or 0), broadcast
    ACC.ADD.FIRST;;                                     # R_ACC = seed (bias or 0; conv taps add on top)

    ADD                 {{ lr_scratch }} {{ lr_sb_off }} CR1;
    ADD                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr.fpb }};
    LD.MULT        R1 {{ lr_scratch }} {{ cr.filters }};;
    BLT.S                 {{ cr.in_channels }} {{ lr_clamp }} gN_clamp;;
    B                   gN_ch_loop;;
gN_clamp:
    ADD                 {{ lr_clamp }} {{ lr_zero }} {{ cr.in_channels }};;

gN_ch_loop:
    # Preamble: load first-ch-of-block kr=0 (real) and kr=+1 (real lr_off_zero; bottom row masked).
    # Mirrors g0_ch_loop/ch_loop: lr_off_zero = lr_chunk_base+lr_ch_ctr so reload paths load the
    # correct channel offset within each super-block.

    # Cy 1: lr_off_zero = E0 = ch0 kr=0 ext = lr_chunk_base+lr_ch_ctr; load ch0 kr=0 → slot 0.
    SET                 {{ lr_read }} CR0;
    SET                 {{ lr_write }} CR0;
    ADD                 {{ lr_off_zero }} {{ lr_chunk_base }} {{ lr_ch_ctr }};
    LD.MULT.CYCLIC {{ lr_off_zero }} {{ cr.x }} {{ lr_write }};;

    # Cy 2: lr_write = 384; load ch0 kr=-1 (real, ext = E0 - in_channels) → slot 384.
    #     lr_walk = -384.
    #     (lr_ch_ctr is NOT reset here — filter_loop sets it to 0 for a fresh filter;
    #      reload re-enters here with lr_ch_ctr at its end-of-block value.)
    ADD                 {{ lr_write }} {{ lr_zero }} {{ cr.ring_adv }};
    SUB                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    SUB                 {{ lr_walk }} {{ lr_zero }} {{ cr.ring_adv }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};;

    # Cy 3: lr_walk = -384 + 1 = -383.
    ADD                 {{ lr_walk }} {{ lr_walk }} CR1;;

    # Cy 4: lr_walk = cols - 383.  lr_write := 128.  lr_kern_idx := -1.
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ cr.cols }};
    ADD                 {{ lr_write }} {{ lr_zero }} {{ cr.slot_stride }};
    {{ kern_idx_init }};;                        # tap 1 INC lands on the first weight (past the bias when seeded)

    B                   gN_tap_body;;

gN_tap_body:
    # 9 cyc/ch body — role-rotating ring (see mn_tap_body).  gN specialization: kr=+1
    # taps (7-9) mask the BOTTOM packed row via mask slot 6.  Only that one row is out
    # of bounds; the other kr=+1 neighbours are real, so THIS ch kr=+1 is loaded with
    # real data (lr_off_zero + in_channels) at tap 1, like mn.  Loads/lr_write identical to mn.

    # --- tap 1: kr=-1 kc=-1.  Load THIS ch kr=+1 (ext = lr_off_zero + in_channels) into the
    #     kr=+1 slot (lr_write = lr_read+128).
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ cr.walk_step }};
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 2.  Bump lr_off_zero to E'.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_off_zero }} {{ lr_off_zero }} CR1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 3.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} CR15;
    ACC.ADD;;

    # --- tap 4: kr=0 kc=-1.  Prefetch NEXT kr=-1 (ext = E' - in_channels) into the SPARE slot.
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    SUB                 {{ lr_scratch }} {{ lr_off_zero }} {{ cr.in_channels }};
    LD.MULT.CYCLIC {{ lr_scratch }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 5.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 6.  lr_ch_ctr += CR1 (the channel row counter).
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    ADD                 {{ lr_ch_ctr }} {{ lr_ch_ctr }} CR1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 0 {{ lr_shift_m1 }} CR15;
    ACC.ADD;;

    # --- tap 7: kr=+1 kc=-1.  Bottom border: slot 6 zeros the last packed row; kc=-1
    #     shift.  Prefetch NEXT kr=0 (ext = E') into THIS dead kr=-1 slot.
    ADD                 {{ lr_walk }} {{ lr_walk }} {{ lr_cols_m2 }};
    INC                 {{ lr_kern_idx }} 1;
    LD.MULT.CYCLIC {{ lr_off_zero }} {{ cr.x }} {{ lr_write }};
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 6 {{ lr_shift_p1 }} CR15;
    ACC.ADD;;

    # --- tap 8: kr=+1 kc=0.  slot 6 = bottom row only.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 6 {{ lr_zero }} CR15;
    ACC.ADD;;

    # --- tap 9: kr=+1 kc=+1.  slot 6 + kc=+1 shift.  lr_write += 128 -> next tap-1 slot.
    INC                 {{ lr_walk }} 1;
    INC                 {{ lr_kern_idx }} 1;
    INC.MODP2       {{ lr_write }} {{ cr.slot_stride }} 9;
    MULT.RC.VE          {{ lr_walk }} {{ lr_kern_idx }} 6 {{ lr_shift_m1 }} CR15;
    ACC.ADD;
    BLT.S                 {{ lr_ch_ctr }} {{ lr_clamp }} gN_tap_body;;

    ADD                 {{ lr_sb_off }} {{ lr_sb_off }} {{ cr.sb_rows }};
    BLT.S                 {{ lr_ch_ctr }} {{ cr.in_channels }} gN_reload;;

    # All input channels done — store aaq_result (128 B int8), advance by 128.
    # ACTIVATE.QUANTIZE reads R_ACC LIVE and ST.AAQ reads
    # post_aaq_reg LIVE (store dispatches after aaq in the per-cycle order),
    # so they co-issue in one word — no separate store cycle needed.
    ACTIVATE.QUANTIZE {{ activation }} CR15;
    ST.AAQ {{ lr_out_ptr }} {{ cr.y }};;

    ADD                 {{ lr_out_ptr }} {{ lr_out_ptr }} CR1;
    BLT.S                 {{ lr_sb_off }} {{ cr.kernel_rows }} gN_filter_loop;;

end:
    HALT;;

gN_reload:
    LD.MULT        R0 {{ lr_sb_off }} {{ cr.filters }};
    SET                 {{ lr_kern_idx }} CR0;;

    ADD                 {{ lr_scratch }} {{ lr_sb_off }} CR1;
    ADD                 {{ lr_clamp }} {{ lr_ch_ctr }} {{ cr.fpb }};
    LD.MULT        R1 {{ lr_scratch }} {{ cr.filters }};;
    BLT.S                 {{ cr.in_channels }} {{ lr_clamp }} gN_reload_clamp;;
    B                   gN_ch_loop;;
gN_reload_clamp:
    ADD                 {{ lr_clamp }} {{ lr_zero }} {{ cr.in_channels }};
    B                   gN_ch_loop;;
