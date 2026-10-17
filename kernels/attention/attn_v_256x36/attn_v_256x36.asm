{#- attn_v_256x36.asm -- attn@V (AGG kernel), query-major scores -> channel-major output.

    Layer:   L3
    Scope:   single-stream
    Layout:  unpacked
    Shape:   256tok (2 groups x 128) x 36chan (head_dim), 4 heads
    Related: attn_v_64x48 (L4) / attn_v_16x60 (L5) are ports of this kernel;
             consumes qk_scores_256x36's query-major output; attn_v_bcast_36
             is the key-major (no-AGG) sibling for the same layer
    Tests:   //kernels/attention/attn_v_256x36; the family's test_datapath.py
             holds it bit for bit to the AGG datapath mirror

    Per head (4 heads, head_dim D=36, N=256 tokens):
      O[i, t] = sum_s P[i, s] * V[s, t]      i,s in [0,256), t in [0,36)

    Layouts (channel-major activations; .asm operands are ROW numbers, issue #179):
      P query-major  : P[i, s] at PBASE + h*512 + i*2 + s//128 rows
                       (query i's 256 scores contiguous -> two 128-key chunks)
      V channel-major: V[s, t] at VBASE + (h*36 + t)*2 + s//128 rows
                       (value channel's 256 keys contiguous -> two 128-key chunks)
      O channel-major: O[i, t] at OBASE + (h*36 + t)*2 + i//128 rows

    Compute model:
      Lanes = keys.  MULT.RC.VV: MULT_RES[s] = P[i,s] * V[s,t]  (element-wise,
      R0 = P chunk, R_CYCLIC = V chunk).  AGG.SUM[.FIRST] dest=local reduces the
      128 live MULT_RES lanes into R_ACC[local] -- collision-free, no ACC, no reset.
      chunk 0 -> AGG.SUM.FIRST (clean write, keys 0..127 partial);
      chunk 1 -> AGG.SUM       (the partial R_ACC[local] from chunk 0 + keys 128..255).
      V chunk reused across all 128 queries of a group; only P (R0) reloads.

    Loop nest:  for h in 0..3 { for t in 0..35 { for g in 0..1 {
                    chunk 0 (keys 0..127):   for local 0..127: load P; MULT; AGG.SUM.FIRST
                    chunk 1 (keys 128..255): for local 0..127: load P; MULT; AGG.SUM
                    ACTIVATE.QUANTIZE identity + ST.AAQ
                      -> O[g*128 .. g*128+127, t]  (one row: a 128-lane FP32 column segment)
                }}}

    Addressing is split offset(LR) + base(CR):
      chan offset chan_off advances by 2 rows monotonically across all 144 (h,t)
        steps (hc = h*36 + t runs 0..143 contiguously) -> single running V/O
        channel ptr.
      head P offset head_p_off advances by 512 rows per head.

    Inner-loop pipeline (live-read offsets, per matmul template):
      LD.MULT offset is read forwarded -> ADD p_ptr +256 co-issues; start p_ptr at
      (first P row addr - 256) so the first live load hits the first row.
      AGG dest_slot is read post-LR-write (forwarded); INC agg_slot co-issues,
      so start agg_slot at -1 -> AGG sees 0,1,..,127.
      BLT.S (Control stage) reads the cycle-start value -1..126 -> bound 126
      gives exactly 128 instruction words (dest 0..127).

      MULT CYCLE-START CONTRACT (issue #157): MULT.RC.VV reads its R0 DATA from the
      cycle-start value, so it cannot consume a P chunk loaded by
      LD.MULT in its own instruction word -- it would see the previous R0. Each of the
      four inner loops therefore primes query 0's P chunk before the loop, and
      every instruction word multiplies the chunk loaded last cycle while prefetching the
      next query's. Only R0 needed shifting: the V chunk in R_CYCLIC is loaded by
      LD.MULT.CYCLIC two instruction words ahead of the loop and never reloaded inside
      it, so it was already safe under the cycle-start rule.
      agg_slot is unaffected by this -- INC stays co-issued and the dest
      sequence is still 0..127.

    Output O is FP32 (R_ACC) stored one group per row, like the transformer
    matmuls. All .asm operands are ROW numbers (issue #179), so O and the P/V
    inputs share the same row-unit addressing regardless of element width:
    O channel offset advances 2 rows/step (=2 groups), group g=1 is at +1 row.
    Inputs P,V advance 2 rows per query-stride step, 1 row per chunk.

    Registers are referred to below by the symbolic names defined in the
    register-name block. The assembler's Jinja2 preprocessor substitutes them
    before parsing, so the emitted binary is byte-identical to the raw form.
    NOTE: Jinja runs before comment stripping, so '#' comments must not
    contain Jinja delimiters -- the preprocessor would try to execute them.

    The register-name block below names every register. The ones the harness
    sets are kernel.toml's [abi] bindings (cr.<name>, lr.<name>); CR0 and CR1
    are the hardwired 0 and 1, and the other LRs are the program's own.

    ---------------------------------------------------------------------------
    Register names (Jinja2 preprocessor; pure source-level substitution)
    ---------------------------------------------------------------------------
-#}
{% set p_ptr            = "LR0"  %}  {# P inner load offset (live +2 rows per query) #}
{% set v_ptr            = "LR1"  %}  {# V chunk offset within the current channel #}
{% set rc_slot0         = "LR2"  %}  {# const 0: R_CYCLIC index #}
{% set agg_slot         = "LR3"  %}  {# AGG dest slot: -1..126 at cycle start, 0..127 after the INC #}
{% set chan_off         = "LR4"  %}  {# value-channel offset, += 2 rows per (h,t) step #}
{% set out_chunk        = "LR5"  %}  {# O row offset for the current group #}
{% set head_p_off       = "LR6"  %}  {# head P offset, += 512 rows per head #}
{% set chan_index       = "LR7"  %}  {# head-channel counter t #}
{% set head_index       = "LR8"  %}  {# head counter h #}
{% set group_p_off      = "LR9"  %}  {# group P offset (head base, then + g*256 rows) #}
{% set out_chan_off     = "LR10" %}  {# O channel offset, += 2 rows per (h,t) step #}
{% set p_query_stride   = "LR11" %}  {# 2 = P query stride, in rows (live add) #}

{% set ZERO             = "CR0"  %}  {# const 0 #}
{% set ONE              = "CR1"  %}  {# hardwired read-only 1 #}
{% set P_BASE           = cr.b  %}  {# query-major P base #}
{% set V_BASE           = cr.a  %}  {# channel-major V base #}
{% set OUT_BASE         = cr.y  %}  {# channel-major O base #}
{% set P_QUERY_STRIDE   = cr.pv_stride  %}  {# 2 = P query stride (rows) #}
{% set CHUNK_OFF        = cr.chunk_off  %}  {# 1 = second key-chunk offset (rows) #}
{% set P_GROUP_STRIDE   = cr.p_group1_off  %}  {# 256 = 128 queries x 2 rows (group g=1) #}
{% set P_HEAD_STRIDE    = cr.p_head_stride  %}  {# 512 = P head stride (rows) #}
{% set INNER_BOUND      = cr.inner_bound  %}  {# 126 = inner bound -> 128 AGG instruction words #}
{% set CHAN_COUNT       = cr.d %}  {# 36 = head channels t #}
{% set HEAD_COUNT       = cr.n_head %}  {# 4 = heads #}
{% set OUT_GROUP_STRIDE = cr.out_group_stride %}  {# 1 = O group stride (rows) #}
{% set OUT_CHAN_STRIDE  = cr.out_chan_stride %}  {# 2 = O channel stride (rows) #}
{% set DSTRUCT          = "CR15" %}  {# reserved dstructure register #}

    SET {{ rc_slot0 }} {{ ZERO }};;                                # rc write index = 0 (const)
    SET {{ chan_off }} {{ ZERO }};;                                # value-channel offset = 0
    SET {{ head_p_off }} {{ ZERO }};;                              # head P offset = 0
    SET {{ head_index }} {{ ZERO }};;                              # head counter = 0
    SET {{ out_chan_off }} {{ ZERO }};;                            # O channel offset = 0
    SET {{ p_query_stride }} {{ P_QUERY_STRIDE }};;                # P query stride = 2 rows

head_loop:
    SET {{ chan_index }} {{ ZERO }};;                              # t counter = 0
    ADD {{ group_p_off }} {{ head_p_off }} {{ ZERO }};;            # group P offset = head P offset (g=0)

t_loop:
    # ===================== group g = 0 (queries 0..127) =====================
    # ---- chunk 0: keys 0..127 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ ZERO }};;                    # V chunk0 offset = chan
    LD.MULT.CYCLIC {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;  # R_CYCLIC = V[0..127, t]   (base VBASE)
    SUB {{ p_ptr }} {{ group_p_off }} {{ p_query_stride }};;       # P inner start = group P off - 2 rows
    SUB {{ agg_slot }} {{ rc_slot0 }} {{ ONE }};;                  # dest/inner counter = -1
    LD.MULT R0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P chunk
g0c0_loop:
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM.FIRST {{ agg_slot }} {{ DSTRUCT }};
    LD.MULT R0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT.S {{ agg_slot }} {{ INNER_BOUND }} g0c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ CHUNK_OFF }};;               # V chunk1 offset = chan + 1 row
    LD.MULT.CYCLIC {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;  # R_CYCLIC = V[128..255, t]
    ADD {{ p_ptr }} {{ group_p_off }} {{ CHUNK_OFF }};;            # P chunk1 base = group P off + 1 row
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 2 rows startup
    SUB {{ agg_slot }} {{ rc_slot0 }} {{ ONE }};;
    LD.MULT R0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P chunk
g0c1_loop:
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM {{ agg_slot }} {{ DSTRUCT }};
    LD.MULT R0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT.S {{ agg_slot }} {{ INNER_BOUND }} g0c1_loop;;

    ADD {{ out_chunk }} {{ out_chan_off }} {{ ZERO }};;            # O g=0 offset = O chan offset
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ out_chunk }} {{ OUT_BASE }};;  # O[0..127, t] = R_ACC   (base OBASE)

    # ===================== group g = 1 (queries 128..255) ===================
    # ---- chunk 0: keys 0..127 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ ZERO }};;                    # V chunk0 offset = chan (same channel)
    LD.MULT.CYCLIC {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;
    ADD {{ p_ptr }} {{ group_p_off }} {{ P_GROUP_STRIDE }};;       # g=1 P base = group P off + 256 rows
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 2 rows startup
    SUB {{ agg_slot }} {{ rc_slot0 }} {{ ONE }};;
    LD.MULT R0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P chunk
g1c0_loop:
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM.FIRST {{ agg_slot }} {{ DSTRUCT }};
    LD.MULT R0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT.S {{ agg_slot }} {{ INNER_BOUND }} g1c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD {{ v_ptr }} {{ chan_off }} {{ CHUNK_OFF }};;               # V chunk1 offset = chan + 1 row
    LD.MULT.CYCLIC {{ v_ptr }} {{ V_BASE }} {{ rc_slot0 }};;
    ADD {{ p_ptr }} {{ group_p_off }} {{ P_GROUP_STRIDE }};;       # g=1 P base
    ADD {{ p_ptr }} {{ p_ptr }} {{ CHUNK_OFF }};;                  # + 1 row (chunk1)
    SUB {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;             # minus 2 rows startup
    SUB {{ agg_slot }} {{ rc_slot0 }} {{ ONE }};;
    LD.MULT R0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};;  # prime query 0's P chunk
g1c1_loop:
    MULT.RC.VV {{ rc_slot0 }} R0 0 {{ rc_slot0 }} {{ DSTRUCT }};
    AGG.SUM {{ agg_slot }} {{ DSTRUCT }};
    LD.MULT R0 {{ p_ptr }} {{ P_BASE }};
    ADD {{ p_ptr }} {{ p_ptr }} {{ p_query_stride }};
    INC {{ agg_slot }} 1;
    BLT.S {{ agg_slot }} {{ INNER_BOUND }} g1c1_loop;;

    ADD {{ out_chunk }} {{ out_chan_off }} {{ OUT_GROUP_STRIDE }};; # O g=1 offset = O chan offset + 1 row
    ACTIVATE.QUANTIZE identity {{ DSTRUCT }};
    ST.AAQ {{ out_chunk }} {{ OUT_BASE }};;  # O[128..255, t] = R_ACC

    # ----- next t: advance value-channel offset (+2 rows) and O offset (+2 rows), t++ -----
    ADD {{ chan_off }} {{ chan_off }} {{ P_QUERY_STRIDE }};;       # chan += 2 rows
    ADD {{ out_chan_off }} {{ out_chan_off }} {{ OUT_CHAN_STRIDE }};; # O chan offset += 2 rows
    INC {{ chan_index }} 1;;                                       # t++
    BLT.S {{ chan_index }} {{ CHAN_COUNT }} t_loop;;

    # ----- next head: head P offset += 512 rows, head++ -----
    ADD {{ head_p_off }} {{ head_p_off }} {{ P_HEAD_STRIDE }};;    # head P offset += 512 rows
    INC {{ head_index }} 1;;
    BLT.S {{ head_index }} {{ HEAD_COUNT }} head_loop;;

end:
    HALT;;
