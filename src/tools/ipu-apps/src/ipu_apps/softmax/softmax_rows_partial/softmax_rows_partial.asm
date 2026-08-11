{#- ==========================================================================
    softmax_rows_partial.asm -- packed row-softmax for N<=128, P=128/ps rows/chunk

    P logical rows share one 128-lane chunk. C_VEC=log2(e) resident in R0
    (broadcast); x-chunk in r_cyclic. Per partition p, MULT.RC.VV rc_idx=p*ps
    (ELEMENT offset -- matches LDR_CYCLIC_MULT_REG's index; issue #182/PR #196)
    reads r_cyclic[p*ps + i]*R0[i] = c*x[p][i] into mult_res lanes 0..N-1; masked
    AGG/ACTIVATE.QUANTIZE name CR15 (valid_elements=N) to act on those N lanes
    only. The full-row drains (MAXVEC/RVEC/output chunk) instead name CR8, whose
    value 128 decodes to valid_elements=128, so they cover the whole 128-lane
    row vector. (Master ISA: every MULT/AGG/ACTIVATE.QUANTIZE names its
    dstructure CR explicitly; the old full_xmem_row 0/1 lane-count flag is gone.)

    Addressing: cyclic/mult loads take (offset_lr, base_cr, index_lr); base CRs
    are fixed, the chunk byte offset walks in an LR. The partition slide rc_idx is
    a SEPARATE intra-register offset (0, ps, ...), not an XMEM address.

    Layout: input/output PACKED; numerators UNPACKED (1 chunk/row); maxvec=c*xmax
    and rvec=1/sum are 128-lane scalar vectors (slot per row).

    CR map (CR0=0, CR1=1 read-only):
      CR2=OUT_BASE  CR3=CVEC  CR4=NUM_BASE  CR5=MAXVEC  CR6=RVEC
      CR7=1 chunk/row stride (.asm XMEM operands are rows -- issue #179)
      CR8=128 (R1 byte-idx base; also full-row dstructure CR: valid_elements=128)
      CR9=total_chunks (group loop bound)  CR10=INPUT_BASE  CR11=512 (RING in
      ELEMENTS, for the Pass-4 repack's reverse-slide SUB src_a -- NOT an XMEM
      stride, so it stays element-valued even though CR7 switched to rows;
      MULT.RC's rc_idx wraps at the ring boundary, not the row boundary)
      CR12=ps partition element stride  CR13=chunks_per_group (=128/P)  CR14=P

    GROUP LOOP: maxvec/rvec are single shared 128-lane vectors (one slot per
    logical row), so at most 128 logical rows -- i.e. CR13 = 128/P chunks -- can
    be in flight at once. All four passes therefore run over a GROUP of up to
    that many chunks, and the whole group is processed (Pass 1..4) before the
    next begins. Beyond loop bookkeeping this is a CORRECTNESS requirement:
    lr_row feeds AGG's dest slot and MULT.RC.VE's `src` scalar-select, both of
    which index R0 for 0..127 and silently switch to the never-loaded R1 at 128.
    Resetting lr_row to 0 each group keeps every such index in range. The last
    group is short: lr_cbound = min(CR13, total_chunks - chunks_done).
========================================================================== -#}

{%- set lr_slide  = "lr0" -%}  {#- intra-register partition slide (0, ps*4, ...) -#}
{%- set lr_row    = "lr1" -%}  {#- global logical-row index = AGG dest slot -#}
{%- set lr_cyc    = "lr2" -%}  {#- constant 0 (cyclic index) -#}
{%- set lr_chunk  = "lr3" -%}  {#- chunk counter -#}
{%- set lr_p      = "lr4" -%}  {#- partition counter (0..P-1) -#}
{%- set lr_coff   = "lr5" -%}  {#- chunk byte offset (chunk * 512) -#}
{%- set lr_num    = "lr6" -%}  {#- numerator chunk byte offset (per row) -#}
{%- set lr_maxid  = "lr7" -%}  {#- R1 byte index for maxvec[row] select (128 + row); Pass 4 reuses as lr_two -#}
{%- set lr_two    = "lr7" -%}  {#- Pass 4 only: literal 2, for the P==2 dispatch check (dead in Pass 4 otherwise) -#}
{%- set lr_rslide = "lr8" -%}  {#- Pass-4 repack slide = RING - p*ps (element offset) -#}
{%- set lr_c512   = "lr9" -%}  {#- holds RING=512 (for SUB src_a; SUB needs LR src_a) -#}
{%- set lr_cdone  = "lr10" -%} {#- chunks completed by previous groups -#}
{%- set lr_cbound = "lr11" -%} {#- this group's chunk count = min(128/P, remaining) -#}
{%- set lr_gbase  = "lr12" -%} {#- group base as a chunk/row number (input & output) -#}
{%- set lr_ctotal = "lr13" -%} {#- total chunks (SUB needs an LR src_a) -#}
{%- set lr_bound  = "lr14" -%} {#- this group's logical-row count = cbound * P (Pass 3) -#}

{#- ===================================================================== -#}
{#- PASS 1 -- maxvec[row] = c*xmax over each partition's N lanes.          -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}} cr0 ;;
    SET {{lr_cdone}} cr0 ;
    SET {{lr_gbase}} cr0 ;
    SET {{lr_ctotal}} cr9 ;;                   {#- total chunks (SUB needs LR src_a) -#}

group_loop:
{#- ---- exact group size: lr_cbound = min(128/P, total_chunks - chunks_done) -#}
    SUB {{lr_cbound}} {{lr_ctotal}} {{lr_cdone}} ;;   {#- remaining chunks (>=1) -#}
    BLT {{lr_cbound}} cr13 group_size_set ;;          {#- fewer than a full group -> keep -#}
    SET {{lr_cbound}} cr13 ;;                         {#- ... else cap at 128/P chunks -#}
group_size_set:

    LDR_MULT_REG r0 {{lr_cyc}} cr3 ;;          {#- R0 = C_VEC (broadcast) -#}
    SET {{lr_row}} cr0 ;
    SET {{lr_chunk}} cr0 ;
    ADD {{lr_coff}} {{lr_gbase}} cr0 ;;        {#- input walks from this group's base -#}

p1_chunk:
    LDR_CYCLIC_MULT_REG {{lr_coff}} cr10 {{lr_cyc}} ;;   {#- r_cyclic = x chunk -#}
    SET {{lr_p}} cr0 ;
    SET {{lr_slide}} cr0 ;;

p1_part:
    MULT.RC.VV    {{lr_slide}} r0 0 {{lr_cyc}} cr15 ;          {#- c*x[p] at lanes 0..N-1 -#}
    AGG.MAX.FIRST {{lr_row}} cr15 ;;                          {#- masked -> maxvec[row] -#}
    ADD {{lr_slide}} {{lr_slide}} cr12 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;
    ADD {{lr_p}} {{lr_p}} cr1 ;;
    BLT {{lr_p}} cr14 p1_part ;;
    ADD {{lr_coff}} {{lr_coff}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    BLT {{lr_chunk}} {{lr_cbound}} p1_chunk ;;

{#- lr_row stepped once per partition per chunk, so it now equals this group's -#}
{#- logical row count (cbound*P). Capture it -- the LR slot has no multiply,   -#}
{#- and Pass 3 (which counts rows, not chunks) needs exactly this bound.       -#}
    ADD {{lr_bound}} {{lr_row}} cr0 ;;
    ACTIVATE.QUANTIZE identity cr8;                        {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr5 ;;                     {#- MAXVEC <- maxvec -#}

{#- ===================================================================== -#}
{#- PASS 2 -- num[row] = 2^(c*x[p] - maxvec[row]).  Unpacked, 1 chunk/row. -#}
{#- ===================================================================== -#}
    SET {{lr_cyc}} cr0 ;;
    LDR_MULT_REG r0 {{lr_cyc}} cr3 ;;          {#- R0 = C_VEC -#}
    LDR_MULT_REG r1 {{lr_cyc}} cr5 ;;          {#- R1 = maxvec (elem row = maxvec[row]) -#}
    SET {{lr_row}} cr0 ;
    SET {{lr_chunk}} cr0 ;
    ADD {{lr_coff}} {{lr_gbase}} cr0 ;;         {#- input walks from this group's base -#}
    SET {{lr_num}} cr0 ;
    SET {{lr_maxid}} cr8 ;;                      {#- R1 byte index = 128 (CR8) + row -#}

p2_chunk:
    LDR_CYCLIC_MULT_REG {{lr_coff}} cr10 {{lr_cyc}} ;;
    SET {{lr_p}} cr0 ;
    SET {{lr_slide}} cr0 ;;

p2_part:
    MULT.RC.VV {{lr_slide}} r0 0 {{lr_cyc}} cr15 ;            {#- c*x[p] -#}
    acc.add.first ;;
    MULT.EE {{lr_maxid}} cr1 0 {{lr_cyc}} cr15 ;              {#- maxvec[row]*1.0 -#}
    acc.sub ;                                                {#- r_acc = c*x[p] - maxvec[row] -#}
    ACTIVATE.QUANTIZE exp2 cr15;                                     {#- reads r_acc LIVE (upstream fix); masked -> num[row] lanes 0..N-1 -#}
    STR_POST_AAQ_REG {{lr_num}} cr4 ;;                    {#- NUM[row] (unpacked) -#}
    ADD {{lr_slide}} {{lr_slide}} cr12 ;
    ADD {{lr_num}} {{lr_num}} cr7 ;
    ADD {{lr_maxid}} {{lr_maxid}} cr1 ;;
    ADD {{lr_row}} {{lr_row}} cr1 ;
    ADD {{lr_p}} {{lr_p}} cr1 ;;
    BLT {{lr_p}} cr14 p2_part ;;
    ADD {{lr_coff}} {{lr_coff}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    BLT {{lr_chunk}} {{lr_cbound}} p2_chunk ;;

{#- ===================================================================== -#}
{#- PASS 3 -- sumvec[row] = SUM num[row]; rvec = 1/sumvec.  Per row.       -#}
{#- ===================================================================== -#}
    SET {{lr_num}} cr0 ;
    SET {{lr_row}} cr0 ;
    SET {{lr_cyc}} cr0 ;;

p3_row:
    LDR_CYCLIC_MULT_REG {{lr_num}} cr4 {{lr_cyc}} ;;      {#- r_cyclic = num[row] (snapshot: visible NEXT cycle) -#}
    MULT.RC.VE    {{lr_cyc}} cr1 0 {{lr_cyc}} cr15 ;           {#- *1.0 -#}
    AGG.SUM.FIRST {{lr_row}} cr15 ;;                          {#- masked -> sumvec[row] -#}
    ADD {{lr_num}} {{lr_num}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    BLT {{lr_row}} {{lr_bound}} p3_row ;;      {#- this group's row count (from Pass 1) -#}

    ACTIVATE.QUANTIZE reciprocal cr8;                       {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_cyc}} cr6 ;;                     {#- RVEC <- 1/sum -#}

{#- ===================================================================== -#}
{#- PASS 4 -- out[row] = num[row]*rvec[row], RE-PACKED to chunk lanes.     -#}
{#-   per chunk: per partition p: r_cyclic=num[row]; MULT.RC.VE with        -#}
{#-   rc_idx = lr_rslide = (RING - p*ps) mod RING places product at        -#}
{#-   lanes p*ps. mask_offset=p selects a per-partition R_MASK slot (built  -#}
{#-   by the harness in PART_MASK_ADDR, loaded once below) with bits        -#}
{#-   [p*ps, p*ps+N) set and every other bit clear -- so MULT.RC.VE's own   -#}
{#-   masking (mask_shift=0, i.e. the slot's raw bits, unshifted) restricts -#}
{#-   mult_res to EXACTLY this partition's lanes, zeroing everywhere else   -#}
{#-   (PadMode.ZERO). acc.add/acc.add.first then only ever add zero outside -#}
{#-   a partition's own range, so partitions can never contaminate each     -#}
{#-   other's r_acc lanes regardless of P (fixes the cross-partition        -#}
{#-   contamination bug in STATUS.md, uniformly for P=1/2/4/8).             -#}
{#-   RING = r_cyclic's full ring size, 512 ELEMENTS (rc_idx is an element  -#}
{#-   offset, matching LDR_CYCLIC_MULT_REG's index -- issue #182/PR #196),  -#}
{#-   NOT the 128-element row size -- MULT.RC's rc_idx wraps at the ring    -#}
{#-   boundary, not the row boundary. p=0 -> rc_idx=0 directly (RING-0=RING -#}
{#-   would itself need a further wrap); p>=1 -> RING - p*ps.               -#}
{#-                                                                        -#}
{#-   mask_offset is a compile-time immediate (baked into the instruction), -#}
{#-   so the per-partition sequence is unrolled per P value via Jinja       -#}
{#-   (p4_run_p1/p2/p4/p8 below) rather than a single runtime loop; a       -#}
{#-   runtime dispatch on cr14 (P) picks which block to enter.              -#}
{#- ===================================================================== -#}
{%- macro p4_partition_block(p, is_last) %}
    {%- if p == 0 %}
    MULT.RC.VE {{lr_cyc}} {{lr_row}} {{p}} {{lr_cyc}} cr15 ;        {#- rc_idx=0 directly; mask_offset={{p}} -#}
    acc.add.first ;;
    {%- else %}
    LDR_CYCLIC_MULT_REG {{lr_num}} cr4 {{lr_cyc}} ;
    SUB {{lr_rslide}} {{lr_c512}} {{lr_slide}} ;;                    {#- rslide = RING - {{p}}*ps -#}
    MULT.RC.VE {{lr_rslide}} {{lr_row}} {{p}} {{lr_cyc}} cr15 ;    {#- mask_offset={{p}} -#}
    acc.add ;;
    {%- endif %}
    ADD {{lr_num}} {{lr_num}} cr7 ;
    ADD {{lr_row}} {{lr_row}} cr1 ;;
    {%- if not is_last %}
    ADD {{lr_slide}} {{lr_slide}} cr12 ;;                          {#- slide = ({{p + 1}})*ps -#}
    {%- endif %}
{%- endmacro %}
    SET {{lr_cyc}} cr0 ;;
    LDR_MULT_REG r0 {{lr_cyc}} cr6 ;;          {#- R0 = rvec (elem row = rvec[row]) -#}
    SET {{lr_row}} cr0 ;
    SET {{lr_chunk}} cr0 ;
    ADD {{lr_coff}} {{lr_gbase}} cr0 ;;         {#- output chunk offset (group base) -#}
    SET {{lr_num}} cr0 ;
    SET {{lr_c512}} cr11 ;
    SET {{lr_two}} cr6 ;;                        {#- lr_c512 = RING (512 elements); lr_two = RVEC row (for mask row below) -#}
    ADD {{lr_two}} {{lr_two}} cr1 ;;             {#- lr_two = RVEC row + 1 = PART_MASK row -#}
    LDR_MULT_MASK_REG {{lr_two}} cr0 ;;         {#- R_MASK <- PART_MASK image (8 per-partition slots), loaded once -#}
p4_chunk:
    LDR_CYCLIC_MULT_REG {{lr_num}} cr4 {{lr_cyc}} ;
    SET {{lr_slide}} cr0 ;
    SET {{lr_p}} cr1 ;;
    BEQ cr14 {{lr_p}} p4_run_p1 ;;
    ADD {{lr_p}} {{lr_p}} cr1 ;;
    BEQ cr14 {{lr_p}} p4_run_p2 ;;
    ADD {{lr_p}} {{lr_p}} cr1 ;;
    ADD {{lr_p}} {{lr_p}} cr1 ;;
    BEQ cr14 {{lr_p}} p4_run_p4 ;;
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
    ACTIVATE.QUANTIZE identity cr8;                         {#- reads r_acc LIVE (upstream fix) -#}
    STR_POST_AAQ_REG {{lr_coff}} cr2 ;;                    {#- packed output chunk -#}
    ADD {{lr_coff}} {{lr_coff}} cr7 ;
    ADD {{lr_chunk}} {{lr_chunk}} cr1 ;;
    BLT {{lr_chunk}} {{lr_cbound}} p4_chunk ;;

{#- ---- advance: chunks_done += cbound; loop while done < total ------------ -#}
{#- Pass 4 stepped lr_coff once per chunk, so it now equals gbase + cbound -- -#}
{#- exactly the next group's base chunk (the LR slot has no multiply).        -#}
    ADD {{lr_cdone}} {{lr_cdone}} {{lr_cbound}} ;
    ADD {{lr_gbase}} {{lr_coff}} cr0 ;;
    BLT {{lr_cdone}} {{lr_ctotal}} group_loop ;;

end:
    BKPT ;;
