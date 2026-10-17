{#- ==========================================================================
    unfold_8x8x240.asm -- splits an 8x8x240 tensor into its four stride-2 phases

    Rearranges an 8x8x240 spatial tensor into 4 stride-2 decimated streams,
    channel-major, one channel per row. The streams are the standard
    stride-2 space-to-depth decomposition -- stream s takes phase
    (s // 2, s % 2) of every 2x2 block. They are NOT image quadrants;
    on/on_inv are just ACC.STRIDE's even/odd selector encodings.
    Related: L5 sibling of unfold_32x32x144 (L3) and unfold_16x16x192 (L4),
    its geometry re-derived, not ported (1 stripe, elements_in_row=16);
    fold_8x8x240 is its inverse; feeds layernorm_16x240.

    Geometry derived for L5, not copied from L3/L4:
      L3 (32x32x144): 8 stripes x (4 spatial rows x 32 cols), elements_in_row=32
      L4 (16x16x192): 2 stripes x (8 spatial rows x 16 cols), elements_in_row=16
      L5 (8x8x240):   1 stripe  x (8 spatial rows x  8 cols), elements_in_row=16

    INPUT CONTRACT -- the paired-row order. W=8 is not an encodable
    elements_in_row (only 16/32/64), so a whole 8x8 channel (64 elements)
    sits in lanes 0..63 of ONE row and ACC.STRIDE views it as 4 rows x 16
    cols -- each view row holding TWO spatial rows side by side. ACC.STRIDE's
    vertical selector splits view rows even/odd, so a plain row-major pack
    (view row r = spatial rows 2r, 2r+1) would make v=on select spatial rows
    {0,1,4,5} instead of the even rows {0,2,4,6}. Spatial rows are therefore
    packed in the order
      view row 0 = [row 0 | row 2]
      view row 1 = [row 1 | row 3]
      view row 2 = [row 4 | row 6]
      view row 3 = [row 5 | row 7]
    i.e. lanes [8k, 8k+8) hold spatial row (0, 2, 1, 3, 4, 6, 5, 7)[k]
    (kernel.toml's paired_rows pack). With it, the four (h, v) selector pairs
    yield exactly the four stride-2 phases, 16 tokens each, row-major.
    Nothing on the IPU produces this order: a caller staging real input must
    permute the spatial rows first. Getting it wrong does not look like
    garbage -- it silently selects the wrong stride-2 phase per stream, a
    plausible-looking row shuffle that is harder to catch than an obviously
    corrupt result. Lanes 64..127 of an input row never reach the output.

    Input x: 240 rows, row ch = channel ch, 64 valid lanes + padding.

    Output y (one channel per row, stream after stream):
      Stream s, channel c: row c of stream s.
      Each row carries 16 valid FP32 tokens in lanes 0..15; lanes 16..127
      are STALE and must be ignored. A stream fills only 16 of the 32
      elements of one ACC.STRIDE slot, ACTIVATE.QUANTIZE identity stages all
      128 lanes of R_ACC, and ST.AAQ writes the whole row. Rows are never
      shared between channels. This is the deliberate per-stream layout; a
      packed [k][p*n] variant is not implemented.

    Pass-through multiply: MULT.RC.VV with R_CYCLIC holding 1.0 in every lane
    (loaded once from the ones row at startup, never overwritten) and the
    input row in R0 (LD.MULT R0):
      MULT.RC.VV LR0 R0 0 LR0: R0[i] x R_CYCLIC[i] = x[i] x 1.0 = x[i].

    Streams (ACC.STRIDE with elements_in_row=16, all into slot 0):
      stream 0: ACC.STRIDE 16 on     on     -> even cols, even spatial rows
      stream 1: ACC.STRIDE 16 on_inv on     -> odd  cols, even spatial rows
      stream 2: ACC.STRIDE 16 on     on_inv -> even cols, odd  spatial rows
      stream 3: ACC.STRIDE 16 on_inv on_inv -> odd  cols, odd  spatial rows
      Each selects 16 valid tokens into R_ACC lanes 0..15 (offset LR0 = slot 0).
    ACC.STRIDE encodings:
      elements_in_row: "16" (enum 0)
      horizontal: on=1 (even cols), on_inv=2 (odd cols)
      vertical:   on=1 (even rows), on_inv=2 (odd rows)

    ABI: kernel.toml's [abi] binds the CRs and LRs below.
      CR0          x, read through the hardwired 0: x must sit at row 0
      CR8          the ones row (R_CYCLIC's 1.0 multiplier)
      CR9..CR12    streams 0..3 of y
      LR5, LR6     source and destination row stride per channel (1)
      LR11         the channel loop's bound (240)
    LR0 (R_CYCLIC / ACC.STRIDE slot 0), LR4 (source row offset), LR8
    (destination row offset) and LR10 (channel counter) start at their reset
    value, 0. CR1 (hardwired 1) is the channel loop's increment. CR0 and CR1
    are read-only: a host write to either raises.

    MULT CYCLE-START CONTRACT (issue #157): MULT.RC.VV reads its R0 data as it
    was at the start of the cycle, so a MULT cannot consume a row loaded by an
    LD.MULT in its own instruction word -- it would see the PREVIOUS contents
    of R0. `;;` ends one instruction word = one cycle, so a load and a MULT
    in the same word always execute in the same cycle regardless of textual
    order; co-issuing them is fine, consuming the same-cycle load is not.
    Here there is a single stripe, so R0 holds the SAME channel row for all
    four streams: one load per channel, primed one instruction word ahead,
    feeds all 4 MULTs. The load for channel ch+1 rides in the last store word
    of channel ch, which is why src_off's ADD must precede it.

    ST.AAQ reads its offset as the forwarded value, after the word's LR
    writes -- never co-issue an ADD on the offset LR with the store, or the
    row lands one row late. dst_off advances in its own instruction word at
    the end of the channel loop, clear of every store.

    Registers are referred to below by the symbolic names defined in the
    register-name block. The assembler's Jinja2 preprocessor substitutes them
    before parsing, so the emitted binary is byte-identical to the raw form.
    NOTE: Jinja runs before comment stripping, so '#' comments in the program
    must not contain Jinja delimiters -- the preprocessor would try to
    execute them.
========================================================================== -#}

# ---------------------------------------------------------------------------
# Register names (Jinja2 preprocessor; pure source-level substitution)
# ---------------------------------------------------------------------------
{% set slot0      = "LR0"  %}  {# const 0: r_cyclic slot 0 / mask / acc.stride slot 0 #}
{% set src_off    = "LR4"  %}  {# src row offset within the stripe, += 1 per ch #}
{% set src_stride = "LR5"  %}  {# 1 = src stride per channel (rows) #}
{% set dst_stride = "LR6"  %}  {# 1 = dst stride per channel (rows) #}
{% set dst_off    = "LR8"  %}  {# dst row offset = ch #}
{% set ch_index   = "LR10" %}  {# channel counter 0..239 #}
{% set ch_limit   = "LR11" %}  {# 240 = C #}

{% set SRC_BASE   = "CR0"  %}  {# stripe base (row 0) #}
{% set ONE        = "CR1"  %}  {# hardwired read-only 1 #}
{% set ONES_BASE  = "CR8"  %}  {# one row of 1.0, for r_cyclic init #}
{% set DST_S0     = "CR9"  %}  {# stream 0 output base #}
{% set DST_S1     = "CR10" %}  {# stream 1 output base #}
{% set DST_S2     = "CR11" %}  {# stream 2 output base #}
{% set DST_S3     = "CR12" %}  {# stream 3 output base #}
{% set DSTRUCT    = "CR15" %}  {# reserved dstructure register #}

    LD.MULT.CYCLIC {{ slot0 }} {{ ONES_BASE }} {{ slot0 }};;   # r_cyclic[0..127] = 1.0

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..239)
# ---------------------------------------------------------------------------

    LD.MULT        R0 {{ src_off }} {{ SRC_BASE }};;   # prime: channel 0

ch_loop:

    # One stripe row in r0 serves all four streams. Each stream MULTs it
    # through by 1.0 and decimates with its own (h, v) selector pair into
    # r_acc slot 0, then stages and stores in one word.

    # -- Stream 0  (h=on=even cols, v=on=even spatial rows) ------------------
    MULT.RC.VV {{ slot0 }} R0 0 {{ slot0 }} {{ DSTRUCT }};
    ACC.STRIDE 16 on on {{ slot0 }};;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ dst_off }} {{ DST_S0 }};;

    # -- Stream 1  (h=on_inv=odd cols, v=on=even spatial rows) ---------------
    MULT.RC.VV {{ slot0 }} R0 0 {{ slot0 }} {{ DSTRUCT }};
    ACC.STRIDE 16 on_inv on {{ slot0 }};;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ dst_off }} {{ DST_S1 }};;

    # -- Stream 2  (h=on=even cols, v=on_inv=odd spatial rows) ---------------
    MULT.RC.VV {{ slot0 }} R0 0 {{ slot0 }} {{ DSTRUCT }};
    ACC.STRIDE 16 on on_inv {{ slot0 }};;
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ dst_off }} {{ DST_S2 }};;

    # -- Stream 3  (h=on_inv=odd cols, v=on_inv=odd spatial rows) ------------
    MULT.RC.VV {{ slot0 }} R0 0 {{ slot0 }} {{ DSTRUCT }};
    ACC.STRIDE 16 on_inv on_inv {{ slot0 }};;
    # src_off must advance BEFORE the load that uses the next channel's offset:
    # LR sub-slots run ahead of LOAD within a word, so this ADD gets its own word.
    ADD                 {{ src_off }} {{ src_off }} {{ src_stride }};;
    # Prefetch channel ch+1 into r0 for the next iteration's first MULT.
    # dst_off is read by the store as the forwarded value, so it is NOT touched in this word.
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ dst_off }} {{ DST_S3 }};
    LD.MULT R0 {{ src_off }} {{ SRC_BASE }};;

    # -- Advance pointers; loop ----------------------------------------------
    ADD                 {{ dst_off }} {{ dst_off }} {{ dst_stride }};;
    ADD                 {{ ch_index }} {{ ch_index }} {{ ONE }};;
    BLT.S                 {{ ch_index }} {{ ch_limit }} ch_loop;;

end:
    HALT;;
