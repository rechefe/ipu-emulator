{#- ==========================================================================
    fold_8x8x240.asm -- rebuilds an 8x8x240 tensor from its four stride-2 phases

    The inverse of unfold_8x8x240: takes the four stride-2 streams that
    kernel produces and writes the 8x8x240 spatial tensor, one channel per
    row, in PLAIN row-major order (lane = row*8 + col). Stream s holds phase
    (s // 2, s % 2) of every 2x2 block -- the standard stride-2
    space-to-depth decomposition, NOT an image quadrant.
    Related: fold_16x16x192 (L4) and fold_32x32x144 (L3) are its siblings.
    fold(unfold(x)) == x bit for bit: the x1.0 pass-through multiply is exact,
    and the fold/unfold round-trip test checks it.

    Input x (one row per channel per stream, stream after stream):
      Stream s, channel c: row c of stream s.
      Only lanes 0..15 of a row are valid: the 16 tokens of the 4x4 decimated
      grid, row-major (src_lane = dr*4 + dc). Lanes 16..127 are unfold's
      stale R_ACC padding; no ACC.RESHAPE call reads past lane 15, so they
      never reach the output.

    Output y:
      240 rows; row ch holds channel ch's 8x8 grid in lanes 0..63, TRUE
      row-major order (lane = row*8 + col). No ACC.RESHAPE call targets lanes
      64..127, so they hold whatever R_ACC held, and they are NOT part of this
      kernel's output contract.

    OUTPUT LAYOUT -- a deliberate departure from L3/L4: fold_32x32x144 and
    fold_16x16x192 write exactly the layout their unfold reads. For L5 that
    would mean reproducing unfold_8x8x240's paired-row input order
    ([0|2] [1|3] [4|6] [5|7]) on the way out. But ACC.RESHAPE's destination
    indices are arbitrary lanes in [0, 127] -- unlike ACC.STRIDE's fixed
    view-row/view-col structure -- so the scatter can target TRUE spatial
    (row, col) positions directly. Both tables were derived and
    partition-checked before this kernel was written:
      paired-row destination table: the 64 valid lanes partition cleanly,
        uniform +32 per call, call-0 bases 0/1/16/17 for streams 0/1/2/3.
      plain destination table (chosen): also partitions cleanly with a
        uniform +32 per call, call-0 bases 0/1/8/9 for streams 0/1/2/3 --
        see ACC.RESHAPE MAPPING below.
    Both are equally simple to encode; plain row-major was chosen because it
    is a genuine [C, H, W] tensor (no consumer has to know the paired-row
    order to read it), and fold needs no output-side inverse of unfold's
    input-side packing. Fold's SOURCE side still accounts for the paired-row
    order implicitly: it consumes unfold's OUTPUT streams, which unfold's own
    (h, v) ACC.STRIDE selectors already put in real stride-2 coordinates --
    fold never sees a paired row.

    STREAM -> REAL (row, col) MAPPING -- verified, not assumed (the lesson of
    fold_16x16x192: never trust a derived layout claim about a stream whose
    loads run one instruction word ahead of their MULT without checking the
    emulator's output). A per-channel, per-lane marker tensor (value =
    ch*1e6 + row*1000 + col) was packed in the paired-row order and run
    through the real unfold_8x8x240; its streams, decoded lane by lane,
    matched x[ch, r_ph::2, c_ph::2].reshape(-1) for stream s (r_ph = s//2,
    c_ph = s%2) exactly. unfold's (h, v) selectors resolve the paired-row
    order internally; nothing of it leaks into a stream's meaning.

    ACC.RESHAPE MAPPING (the stride-2 decimation inverted with the plain
    destination formula lane = row*8 + col; brute-force partition-checked):
      Each stream contributes its 16 tokens (src_lane = dr*4 + dc) to 16 of
      the destination row's 64 valid lanes, in 2 ACC.RESHAPE calls of 8
      elements each:
        source (the same for every stream): [0..7] for call 0, [8..15] for
          call 1 -- one stripe, so no window shifting.
        destination, stream s, call c: base(s) + table, where
          base(s0)=0, base(s1)=1, base(s2)=8, base(s3)=9, and
            call0 = base + [0,2,4,6,16,18,20,22]
            call1 = base + [32,34,36,38,48,50,52,54]   (call0 + 32)
      The 8 (stream, call) tables cover destination lanes 0..63 exactly once
      each (64 distinct indices, no overlap). Lanes 64..127 are never
      targeted: a channel has only 64 spatial elements.

    INDEX TABLES. SET copies a CR into an LR, so each table is held in two
    CRs, four little-endian lane indices per CR, and copied into an LR pair
    (LRDn = LRn, LRn+1) when needed:
      SOURCE table LRD2 (LR2, LR3) is SET from CR2, CR3 ([0,1,2,3] /
      [4,5,6,7]) at the start of every stream's 2-call block, then walked +8
      per call with ADDBI.
      DESTINATION table LRD4 (LR4, LR5) is SET from CR4, CR5 (stream 0's
      call-0 table, [0,2,4,6] / [16,18,20,22]); streams 1/2/3 then walk it to
      their base with one ADDBI (+1, +8, +9). Between a stream's 2 calls it
      walks +32 (uniform, unlike fold_32x32x144's +16/+48/+16: L5's 4x4
      decimated grid gives a uniform step, like fold_16x16x192's). It is
      rebuilt once per stream per channel for auditability; a cycle-tuned
      version could keep it across channels.

    ABI: kernel.toml's [abi] binds the CRs and LRs below.
      CR2, CR3     source table [0,1,2,3] / [4,5,6,7]
      CR4, CR5     stream 0's destination table [0,2,4,6] / [16,18,20,22]
      CR8          the ones row (R_CYCLIC's 1.0 multiplier)
      CR9..CR12    streams 0..3 of x
      CR13         y
      LR7, LR9     source and destination row stride per channel (1)
      LR11         the channel loop's bound (240)
    LR0 (R_CYCLIC slot 0), LR6 (source row offset), LR8 (destination row
    offset) and LR10 (channel counter) start at 0, as kernel.toml's [abi]
    binds them (slot0, src_row, dst_row, channel).
    CR1 (hardwired 1) is the channel loop's increment.

    CR0 IS NOT A FREE BASE: CR0 (0) and CR1 (1) are hardwired, and a host
    write to either raises. unfold_8x8x240 reads its input through CR0 only
    because that base is row 0. Fold's output base is not 0 (y sits after
    x), so it is bound to CR13 -- the trap fold_16x16x192 and
    fold_32x32x144 also note.

    MULT CYCLE-START CONTRACT (issue #157, as in fold_16x16x192 and
    fold_32x32x144): MULT.RC.VV reads its Ra data (R0) as it was at the
    start of the cycle, so it cannot consume a row loaded by an LD.MULT in
    the same instruction word. Co-issuing a load and a MULT is fine only when
    the MULT consumes a row loaded earlier. This kernel primes the first
    load before ch_loop and keeps every later load one instruction word ahead
    of the MULT that consumes it. Rotation: stream 0's block prefetches
    stream 1's row (same channel), stream 1 prefetches stream 2, stream 2
    prefetches stream 3, and stream 3 prefetches stream 0 of the NEXT
    channel (advancing LR6 first, in its own instruction word).

    ACTIVATE.EXACT reads its offset operand as the forwarded value, after the word's
    LR writes -- never co-issue an ADD on the offset LR (LR8) with the store,
    or the row lands one row late. Here LR8 advances in its own instruction
    word at the very end of the channel loop, well clear of the store.
========================================================================== -#}

{% set groups = [
     ("s0", 0, "CR9",  "CR10"),
     ("s1", 1, "CR10", "CR11"),
     ("s2", 8, "CR11", "CR12"),
     ("s3", 9, "CR12", "CR9"),
   ] %}

    LD.MULT.CYCLIC LR0 CR8 LR0;;       # r_cyclic[0..127] = 1.0

    LD.MULT        R0 LR6 CR9;;        # prime: stream 0 row of ch 0

# ---------------------------------------------------------------------------
# Main channel loop  (ch = 0..239)
# ---------------------------------------------------------------------------

ch_loop:

{% for g in groups %}
{%- set name = g[0] %}
{%- set dest_base = g[1] %}
{%- set next_cr = g[3] %}
    # -- Stream {{ name }}: dest table = base{% if dest_base != 0 %} + {{ dest_base }}{% endif %} -----------------------------------
    SET                 LR2 CR2;
    SET                 LR3 CR3;;           # source table <- [0,1,2,3,4,5,6,7]
    SET                 LR4 CR4;
    SET                 LR5 CR5;;           # dest table   <- stream-0 base
{%- if dest_base != 0 %}
    ADDBI               LRD4 {{ dest_base }};;      # dest table <- {{ name }} base
{%- endif %}
{%- if name == "s3" %}
    # lr6 (src offset) must advance BEFORE the load that uses next channel's
    # offset: LR sub-slots run ahead of LOAD within a word (same hazard
    # fold_16x16x192/fold_32x32x144 document for their src-offset LR), so
    # this ADD gets its own word.
    ADD                 LR6 LR6 LR7;;       # src offset: next channel (+1 row)
{%- endif %}
    LD.MULT R0 LR6 {{ next_cr }};
    MULT.RC.VV LR0 R0 0 LR0 CR15.L {{ cr.qf }}.L;;          # multiply {{ name }} row (prefetch next stream)
    ACC.RESHAPE LRD2 LRD4 0;;               # call0: src[0..7] -> dest[{{ dest_base }},{{ dest_base+2 }},{{ dest_base+4 }},{{ dest_base+6 }},{{ dest_base+16 }},{{ dest_base+18 }},{{ dest_base+20 }},{{ dest_base+22 }}]
    ADDBI               LRD2 8;
    ADDBI               LRD4 32;;
    ACC.RESHAPE LRD2 LRD4 0;;               # call1: src[8..15] -> dest[+32 of call0]
{% endfor %}
    ACTIVATE.EXACT identity CR15.L CR13 LR8;; # ch <- DST_BASE_ROW + ch (current channel)

    # -- Advance pointers; loop -------------------------------------------------
    ADD                 LR8 LR8 LR9;;       # dst offset: next channel
    ADD                 LR10 LR10 CR1;;
    BLT.S                 LR10 LR11 ch_loop;; # loop while ch < 240

end:
    HALT;;
