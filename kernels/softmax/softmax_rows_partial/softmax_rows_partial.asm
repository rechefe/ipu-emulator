{#- ==========================================================================
    softmax_rows_partial.asm -- packed row softmax for n <= 128, P = 128/ps rows/chunk

    y[r, :] = softmax(x[r, :]) for a (rows, n) matrix with n <= 128. Each row
    sits in a ps-lane slot, ps = partition_size(n) (the next power of two >= n,
    16..128), so P = 128 / ps logical rows share one 128-element chunk (P = 1, 2,
    4 or 8 for n in 65..128, 33..64, 17..32, 1..16). The harness packs and
    unpacks x and y that way (ipu_kernels.softmax.pack row_slots: 0.0 padding,
    the last chunk filled up with padding slots), so the files keep the plain
    (rows, n) layout. Only the numerators are unpacked (one chunk per row),
    which keeps the per-row reduce cheap.

    C_VEC = log2(e) is resident in R0 (broadcast); the x chunk is in R_CYCLIC.
    Per partition p, MULT.RC.VV with rc_idx = p*ps (an ELEMENT offset, as
    LD.MULT.CYCLIC's index; issue #182/PR #196) reads R_CYCLIC[p*ps + i] * R0[i]
    = c * x[p][i] into MULT_RES elements 0..n-1; the masked AGG and
    ACTIVATE.EXACT name cr.dstruct.L (a DS with valid_elements = n) to act on
    those n elements only. The full-row drains (maxvec, rvec, the output chunk)
    name cr.maxvec_base.H instead, a DS with valid_elements = 128, so they
    cover the whole row vector. (Every MULT/AGG/ACTIVATE.EXACT names its DS
    half explicitly; every MULT reads the case's QF from cr.dstruct.H.)

    Addressing: cyclic and mult loads take (offset_lr, base_cr, index_lr); the
    base CRs are fixed and the chunk's row offset walks in an LR. The partition
    slide rc_idx is a SEPARATE intra-register offset (0, ps, ...), not an XMEM
    address.

    Layout: x and y PACKED; num UNPACKED (1 chunk per row); maxvec = c * xmax
    and rvec = 1/sum are 128-element scalar vectors (one slot per row).

    GROUP LOOP: maxvec/rvec are single shared 128-element vectors (one slot per
    logical row), so at most 128 logical rows -- cr.group_chunks = 128/P chunks
    -- can be in flight at once. All four passes therefore run over a GROUP of
    up to that many chunks, and the whole group is processed (Pass 1..4) before
    the next begins. Beyond loop bookkeeping this is a CORRECTNESS requirement:
    lr_row feeds AGG's dest slot and MULT.RC.VE's `src` scalar-select, both of
    which index R0 for 0..127 and silently switch to the never-loaded R1 at 128
    (with one global row counter, P=1/n=128 broke at 129 rows and P=2/n=64 at
    130). Resetting lr_row to 0 each group keeps every such index in range. The
    last group is short: lr_cbound = min(cr.group_chunks, total_chunks -
    chunks_done), so no padding chunk is processed. Pass 3 counts rows, not
    chunks; the LR slot has no multiply, so instead of cbound * P it takes
    Pass 1's lr_row, which steps once per partition per chunk and so ends at
    exactly that product. The next group's base is likewise the walked pointer
    (lr_coff after Pass 4).

    PASS-4 MASKS: Pass 4 re-packs P logical rows into one chunk, partition p
    adding its MULT.RC.VE result into the shared R_ACC (ACC.ADD, ACC.ADD.FIRST
    for p = 0). MULT.RC.VE always produces 128 real elements, and only
    [p*ps, p*ps+n) are this partition's; the rest is other rows' data, picked up
    by the rc_idx wrap-around placement. ACC.ADD does not mask, so without a mask
    each partition's out-of-range elements leaked into its neighbours' slots
    (invisible at low chunk counts, where the leaked values stayed under the
    test tolerance: P=2 broke at 35 chunks, P=4 at 9, P=8 at 3). Passes 1-3 were
    never affected: AGG masks by valid_elements. The fix: an 8-slot R_MASK image
    (128 bytes, one row below rvec, cr.rvec + 1; kernel.toml's rvec region,
    scratch.py:rvec_mask) where slot p < P keeps bits [p*ps, p*ps+n) (bit 1 =
    keep) and LD.MASK loads it once per Pass-4 entry. MULT.RC.VE's mask_offset is
    an assemble-time immediate (and mask_shift moves only +/-3 bits), so the
    per-partition body is unrolled with Jinja (p4_partition_block(p, is_last))
    into p4_run_p1, _p2, _p4 and _p8, each with a literal mask_offset = p per
    partition; a runtime dispatch on cr.parts (P) at the top of p4_chunk enters
    one, and each ends with a branch to the shared p4_drain. Each partition then
    adds zeros outside its own slot.

    Open: the group loop re-loads C_VEC and re-runs the Pass 1..4 prologue per
    group; negligible for large groups, but a per-128-rows overhead that could
    be hoisted.

    Registers. All XMEM operands are ROW numbers; one row is 128 FP32 lanes
    (issue #179). The harness places the regions and sets the CRs from
    kernel.toml ([abi]):
      cr.x      x, the input logits         cr.y     y, the output
      cr.num    num, the numerators         cr.log2e C_VEC: log2(e) in every lane
      cr.max    maxvec row                  cr.rvec  rvec row; the R_MASK image at + 1
      cr.row_stride   = 1 (one row per chunk)
      cr.maxvec_base  LOW: 128, maxvec's index base in R1 (R1's element 128 +
                        row); HIGH: the full-row DS (valid_elements = 128)
      cr.device_rows  = total chunks (the group loop bound)
      cr.ring         = 512: R_CYCLIC's ring in ELEMENTS, for the Pass-4 repack's
                        reverse-slide SUB src_a (not an XMEM stride: rc_idx wraps
                        at the ring, not the row)
      cr.ps           = the partition's element stride
      cr.group_chunks = chunks per full group (128 / P)
      cr.parts        = P
      cr.dstruct      LOW: the DS with valid_elements = n; HIGH: the case's QF
      CR0 = 0 and CR1 = 1 (-> the 1.0 scalar, and the +1 step) are read-only.
========================================================================== -#}

{%- set lr_slide  = "LR0" -%}  {#- intra-register partition slide (0, ps, ...: elements) -#}
{%- set lr_row    = "LR1" -%}  {#- logical-row index in the group = AGG dest slot -#}
{%- set lr_cyc    = "LR2" -%}  {#- constant 0 (cyclic index) -#}
{%- set lr_chunk  = "LR3" -%}  {#- chunk counter -#}
{%- set lr_p      = "LR4" -%}  {#- partition counter (0..P-1) -#}
{%- set lr_coff   = "LR5" -%}  {#- chunk row offset (one row per chunk) -#}
{%- set lr_num    = "LR6" -%}  {#- numerator chunk row offset (per row) -#}
{%- set lr_maxid  = "LR7" -%}  {#- R1 element index for maxvec[row] select (128 + row); Pass 4 reuses as lr_mask_row -#}
{%- set lr_mask_row    = "LR7" -%}  {#- Pass 4 only: the R_MASK image's row, cr.rvec + 1 -#}
{%- set lr_rslide = "LR8" -%}  {#- Pass-4 repack slide = RING - p*ps (element offset) -#}
{%- set lr_c512   = "LR9" -%}  {#- holds RING=512 (for SUB src_a; SUB needs LR src_a) -#}
{%- set lr_cdone  = "LR10" -%} {#- chunks completed by previous groups -#}
{%- set lr_cbound = "LR11" -%} {#- this group's chunk count = min(128/P, remaining) -#}
{%- set lr_gbase  = "LR12" -%} {#- group base as a chunk/row number (input & output) -#}
{%- set lr_ctotal = "LR13" -%} {#- total chunks (SUB needs an LR src_a) -#}
{%- set lr_bound  = "LR14" -%} {#- this group's logical-row count = cbound * P (Pass 3) -#}

{#- ===================================================================== -#}
{#- PASS 1 -- maxvec[row] = c*xmax over each partition's N elements.          -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}} CR0 ;;
    SET {{lr_cdone}} CR0 ;
    SET {{lr_gbase}} CR0 ;
    SET {{lr_ctotal}} {{ cr.device_rows }} ;;                   {#- total chunks (SUB needs LR src_a) -#}

group_loop:
{#- ---- exact group size: lr_cbound = min(128/P, total_chunks - chunks_done) -#}
    SUB {{lr_cbound}} {{lr_ctotal}} {{lr_cdone}} ;;   {#- remaining chunks (>=1) -#}
    BLT.S {{lr_cbound}} {{ cr.group_chunks }} group_size_set ;;          {#- fewer than a full group -> keep -#}
    SET {{lr_cbound}} {{ cr.group_chunks }} ;;                         {#- ... else cap at 128/P chunks -#}
group_size_set:

    LD.MULT R0 {{lr_cyc}} {{ cr.log2e }} ;;          {#- R0 = C_VEC (broadcast) -#}
    SET {{lr_row}} CR0 ;
    SET {{lr_chunk}} CR0 ;
    ADD {{lr_coff}} {{lr_gbase}} CR0 ;;        {#- input walks from this group's base -#}

p1_chunk:
    LD.MULT.CYCLIC {{lr_coff}} {{ cr.x }} {{lr_cyc}} ;;   {#- R_CYCLIC = x chunk -#}
    SET {{lr_p}} CR0 ;
    SET {{lr_slide}} CR0 ;;

p1_part:
    MULT.RC.VV    {{lr_slide}} R0 0 {{lr_cyc}} {{ cr.dstruct }}.L {{ cr.dstruct }}.H ;          {#- c*x[p] at elements 0..N-1 -#}
    AGG.MAX.FIRST {{lr_row}} {{ cr.dstruct }}.L ;;                          {#- masked -> maxvec[row] -#}
    ADD {{lr_slide}} {{lr_slide}} {{ cr.ps }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;
    ADD {{lr_p}} {{lr_p}} CR1 ;;
    BLT.S {{lr_p}} {{ cr.parts }} p1_part ;;
    ADD {{lr_coff}} {{lr_coff}} {{ cr.row_stride }} ;
    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;;
    BLT.S {{lr_chunk}} {{lr_cbound}} p1_chunk ;;

{#- lr_row stepped once per partition per chunk, so it now equals this group's -#}
{#- logical row count (cbound*P). Capture it -- the LR slot has no multiply,   -#}
{#- and Pass 3 (which counts rows, not chunks) needs exactly this bound.       -#}
    ADD {{lr_bound}} {{lr_row}} CR0 ;;
    ACTIVATE.EXACT identity {{ cr.maxvec_base }}.H {{ cr.max }} {{lr_cyc}} ;;  {#- reads R_ACC forwarded; MAXVEC <- maxvec -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[row] = 2^(c*x[p] - maxvec[row]).  Unpacked, 1 chunk/row. -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}} CR0 ;;
    LD.MULT R0 {{lr_cyc}} {{ cr.log2e }} ;;          {#- R0 = C_VEC -#}
    LD.MULT R1 {{lr_cyc}} {{ cr.max }} ;;          {#- R1 = maxvec (elem row = maxvec[row]) -#}
    SET {{lr_row}} CR0 ;
    SET {{lr_chunk}} CR0 ;
    ADD {{lr_coff}} {{lr_gbase}} CR0 ;;         {#- input walks from this group's base -#}
    SET {{lr_num}} CR0 ;
    SET {{lr_maxid}} {{ cr.maxvec_base }}.L ;;                      {#- R1 element index = 128 (cr.maxvec_base) + row -#}

p2_chunk:
    LD.MULT.CYCLIC {{lr_coff}} {{ cr.x }} {{lr_cyc}} ;;
    SET {{lr_p}} CR0 ;
    SET {{lr_slide}} CR0 ;;

p2_part:
    MULT.RC.VV {{lr_slide}} R0 0 {{lr_cyc}} {{ cr.dstruct }}.L {{ cr.dstruct }}.H ;            {#- c*x[p] -#}
    ACC.ADD.FIRST ;;
    MULT.EE {{lr_maxid}} CR1 0 {{lr_cyc}} {{ cr.dstruct }}.L {{ cr.dstruct }}.H ;              {#- maxvec[row]*1.0 -#}
    ACC.SUB ;                                                {#- R_ACC = c*x[p] - maxvec[row] -#}
    ACTIVATE.EXACT exp2 {{ cr.dstruct }}.L {{ cr.num }} {{lr_num}} ;;  {#- reads R_ACC forwarded; masked -> num[row] elements 0..N-1; NUM[row] (unpacked) -#}
    ADD {{lr_slide}} {{lr_slide}} {{ cr.ps }} ;
    ADD {{lr_num}} {{lr_num}} {{ cr.row_stride }} ;
    ADD {{lr_maxid}} {{lr_maxid}} CR1 ;;
    ADD {{lr_row}} {{lr_row}} CR1 ;
    ADD {{lr_p}} {{lr_p}} CR1 ;;
    BLT.S {{lr_p}} {{ cr.parts }} p2_part ;;
    ADD {{lr_coff}} {{lr_coff}} {{ cr.row_stride }} ;
    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;;
    BLT.S {{lr_chunk}} {{lr_cbound}} p2_chunk ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[row] = SUM num[row]; rvec = 1/sumvec.  Per row.       -#}
{#- ===================================================================== -#}
    SET {{lr_num}} CR0 ;
    SET {{lr_row}} CR0 ;
    SET {{lr_cyc}} CR0 ;;

p3_row:
    LD.MULT.CYCLIC {{lr_num}} {{ cr.num }} {{lr_cyc}} ;;      {#- R_CYCLIC = num[row] (read at cycle start: visible NEXT cycle) -#}
    MULT.RC.VE    {{lr_cyc}} CR1 0 {{lr_cyc}} {{ cr.dstruct }}.L {{ cr.dstruct }}.H ;           {#- *1.0 -#}
    AGG.SUM.FIRST {{lr_row}} {{ cr.dstruct }}.L ;;                          {#- masked -> sumvec[row] -#}
    ADD {{lr_num}} {{lr_num}} {{ cr.row_stride }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    BLT.S {{lr_row}} {{lr_bound}} p3_row ;;      {#- this group's row count (from Pass 1) -#}

    ACTIVATE.EXACT reciprocal {{ cr.maxvec_base }}.H {{ cr.rvec }} {{lr_cyc}} ;;  {#- reads R_ACC forwarded; RVEC <- 1/sum -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[row] = num[row]*rvec[row], RE-PACKED to chunk elements.     -#}
{#-   per chunk: per partition p: R_CYCLIC=num[row]; MULT.RC.VE with        -#}
{#-   rc_idx = lr_rslide = (RING - p*ps) mod RING places product at        -#}
{#-   elements p*ps. mask_offset=p selects a per-partition R_MASK slot (built  -#}
{#-   at cr.rvec + 1 (scratch.py:rvec_mask), loaded once below) with bits-#}
{#-   [p*ps, p*ps+N) set and every other bit clear -- so MULT.RC.VE's own   -#}
{#-   masking (mask_shift=0, i.e. the slot's raw bits, unshifted) restricts -#}
{#-   MULT_RES to EXACTLY this partition's elements, zeroing everywhere else   -#}
{#-   (PadMode.ZERO). ACC.ADD/ACC.ADD.FIRST then only ever add zero outside -#}
{#-   a partition's own range, so partitions can never contaminate each     -#}
{#-   other's R_ACC elements regardless of P (fixes the cross-partition        -#}
{#-   contamination bug in the header, uniformly for P=1/2/4/8).            -#}
{#-   RING = R_CYCLIC's full ring size, 512 ELEMENTS (rc_idx is an element  -#}
{#-   offset, matching LD.MULT.CYCLIC's index -- issue #182/PR #196),  -#}
{#-   NOT the 128-element row size -- MULT.RC's rc_idx wraps at the ring    -#}
{#-   boundary, not the row boundary. p=0 -> rc_idx=0 directly (RING-0=RING -#}
{#-   would itself need a further wrap); p>=1 -> RING - p*ps.               -#}
{#-                                                                        -#}
{#-   mask_offset is a compile-time immediate (baked into the instruction), -#}
{#-   so the per-partition sequence is unrolled per P value via Jinja       -#}
{#-   (p4_run_p1/p2/p4/p8 below) rather than a single runtime loop; a       -#}
{#-   runtime dispatch on cr.parts (P) picks which block to enter.              -#}
{#- ===================================================================== -#}
{%- macro p4_partition_block(p, is_last) %}
    {%- if p == 0 %}
    MULT.RC.VE {{lr_cyc}} {{lr_row}} {{p}} {{lr_cyc}} {{ cr.dstruct }}.L {{ cr.dstruct }}.H ;        {#- rc_idx=0 directly; mask_offset={{p}} -#}
    ACC.ADD.FIRST ;;
    {%- else %}
    LD.MULT.CYCLIC {{lr_num}} {{ cr.num }} {{lr_cyc}} ;
    SUB {{lr_rslide}} {{lr_c512}} {{lr_slide}} ;;                    {#- rslide = RING - {{p}}*ps -#}
    MULT.RC.VE {{lr_rslide}} {{lr_row}} {{p}} {{lr_cyc}} {{ cr.dstruct }}.L {{ cr.dstruct }}.H ;    {#- mask_offset={{p}} -#}
    ACC.ADD ;;
    {%- endif %}
    ADD {{lr_num}} {{lr_num}} {{ cr.row_stride }} ;
    ADD {{lr_row}} {{lr_row}} CR1 ;;
    {%- if not is_last %}
    ADD {{lr_slide}} {{lr_slide}} {{ cr.ps }} ;;                          {#- slide = ({{p + 1}})*ps -#}
    {%- endif %}
{%- endmacro %}
    SET {{lr_cyc}} CR0 ;;
    LD.MULT R0 {{lr_cyc}} {{ cr.rvec }} ;;          {#- R0 = rvec (elem row = rvec[row]) -#}
    SET {{lr_row}} CR0 ;
    SET {{lr_chunk}} CR0 ;
    ADD {{lr_coff}} {{lr_gbase}} CR0 ;;         {#- output chunk offset (group base) -#}
    SET {{lr_num}} CR0 ;
    SET {{lr_c512}} {{ cr.ring }} ;
    SET {{lr_mask_row}} {{ cr.rvec }} ;;                        {#- lr_c512 = RING (512 elements); lr_mask_row = cr.rvec -#}
    ADD {{lr_mask_row}} {{lr_mask_row}} CR1 ;;             {#- lr_mask_row = cr.rvec + 1, the R_MASK image -#}
    LD.MASK {{lr_mask_row}} CR0 ;;         {#- R_MASK <- the image (8 per-partition slots), loaded once -#}
p4_chunk:
    LD.MULT.CYCLIC {{lr_num}} {{ cr.num }} {{lr_cyc}} ;
    SET {{lr_slide}} CR0 ;
    SET {{lr_p}} CR1 ;;
    BEQ {{ cr.parts }} {{lr_p}} p4_run_p1 ;;
    ADD {{lr_p}} {{lr_p}} CR1 ;;
    BEQ {{ cr.parts }} {{lr_p}} p4_run_p2 ;;
    ADD {{lr_p}} {{lr_p}} CR1 ;;
    ADD {{lr_p}} {{lr_p}} CR1 ;;
    BEQ {{ cr.parts }} {{lr_p}} p4_run_p4 ;;
    B p4_run_p8 ;;

p4_run_p1:
{{- p4_partition_block(0, true) }}
    B p4_drain ;;

p4_run_p2:
{%- for p in range(2) %}
{{ p4_partition_block(p, p == 1) }}
{%- endfor %}
    B p4_drain ;;

p4_run_p4:
{%- for p in range(4) %}
{{ p4_partition_block(p, p == 3) }}
{%- endfor %}
    B p4_drain ;;

p4_run_p8:
{%- for p in range(8) %}
{{ p4_partition_block(p, p == 7) }}
{%- endfor %}
    B p4_drain ;;

p4_drain:
    ACTIVATE.EXACT identity {{ cr.maxvec_base }}.H {{ cr.y }} {{lr_coff}} ;;  {#- reads R_ACC forwarded; packed output chunk -#}
    ADD {{lr_coff}} {{lr_coff}} {{ cr.row_stride }} ;
    ADD {{lr_chunk}} {{lr_chunk}} CR1 ;;
    BLT.S {{lr_chunk}} {{lr_cbound}} p4_chunk ;;

{#- ---- advance: chunks_done += cbound; loop while done < total ------------ -#}
{#- Pass 4 stepped lr_coff once per chunk, so it now equals gbase + cbound -- -#}
{#- exactly the next group's base chunk (the LR slot has no multiply).        -#}
    ADD {{lr_cdone}} {{lr_cdone}} {{lr_cbound}} ;
    ADD {{lr_gbase}} {{lr_coff}} CR0 ;;
    BLT.S {{lr_cdone}} {{lr_ctotal}} group_loop ;;

end:
    HALT ;;
