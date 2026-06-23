# Agent A — attn@V (AGG kernel), query-major scores -> channel-major output.
#
# Per head (4 heads, head_dim D=36, N=256 tokens):
#   O[i, t] = sum_s P[i, s] * V[s, t]      i,s in [0,256), t in [0,36)
#
# Layouts (channel-major activations, byte addresses):
#   P query-major  : P[i, s] at PBASE + h*65536 + i*256 + s
#                    (query i's 256 scores contiguous -> two 128-key chunks)
#   V channel-major: V[s, t] at VBASE + (h*36 + t)*256 + s
#                    (value channel's 256 keys contiguous -> two 128-key chunks)
#   O channel-major: O[i, t] at OBASE + (h*36 + t)*256 + i
#
# Compute model (post-merge AGG, commit d67c441):
#   Lanes = keys.  MULT.RC.VV: mult_res[s] = P[i,s] * V[s,t]  (element-wise,
#   R0 = P chunk, R_CYCLIC = V chunk).  AGG.SUM[.FIRST] dest=local reduces the
#   128 live mult_res lanes into R_ACC[local] -- collision-free, no ACC, no reset.
#   chunk 0 -> AGG.SUM.FIRST (clean write, keys 0..127 partial);
#   chunk 1 -> AGG.SUM       (snapshot R_ACC[local] from chunk 0 + keys 128..255).
#   V chunk reused across all 128 queries of a group; only P (R0) reloads.
#
# Loop nest:  for h in 0..3 { for t in 0..35 { for g in 0..1 {
#                 chunk 0 (keys 0..127):   for local 0..127: load P; MULT; AGG.SUM.FIRST
#                 chunk 1 (keys 128..255): for local 0..127: load P; MULT; AGG.SUM
#                 STR_ACC_REG -> O[g*128 .. g*128+127, t]  (512B FP32 column segment)
#             }}}
#
# Addressing is split offset(LR) + base(CR):
#   chan offset lr4 advances by 256 monotonically across all 144 (h,t) steps
#     (hc = h*36 + t runs 0..143 contiguously) -> single running V/O channel ptr.
#   head P offset lr6 advances by 65536 per head.
#
# Inner-loop pipeline (live-read offsets, per matmul template):
#   LDR_MULT_REG offset is read LIVE -> ADD lr0 +256 co-issues; start lr0 at
#   (first P row addr - 256) so the first live load hits the first row.
#   AGG dest_slot is read from SNAPSHOT; INC lr3 co-issues -> AGG sees 0,1,..,127.
#   BLT reads SNAPSHOT -> bound 127 gives exactly 128 bundles (dest 0..127).
#
# Output O is FP32 (R_ACC, 4 bytes/elem) stored as 512-byte group rows, like the
# transformer matmuls.  So O addressing uses FP32 strides (separate from the
# 1-byte input strides): O channel offset advances 1024/step (=2 groups*512),
# group g=1 is at +512.  Inputs P,V stay 1 byte/elem (256 stride, 128 chunk).
#
# CRs (harness):
#   cr0=0  cr1=1(hw)  cr2=PBASE  cr3=VBASE  cr4=OBASE
#   cr5=256  cr6=128  cr7=32768(=128*256)  cr8=65536(P head stride)
#   cr9=127(inner bound)  cr10=36(t count)  cr11=4(head count)
#   cr12=512(O group stride)  cr13=1024(O channel stride)
# LRs:
#   lr0=P inner load offset (live +256)   lr1=V chunk offset    lr2=0 (rc index)
#   lr3=dest/inner counter 0..127         lr4=value-channel offset (+256/step)
#   lr5=O chunk offset      lr6=head P offset (+65536/head)
#   lr7=t counter           lr8=head counter   lr9=group P offset
#   lr10=O channel offset (+1024/step)    lr11=256 (P query stride, live add)

    SET lr2 cr0;;                       # rc write index = 0 (const)
    SET lr4 cr0;;                       # value-channel offset = 0
    SET lr6 cr0;;                       # head P offset = 0
    SET lr8 cr0;;                       # head counter = 0
    SET lr10 cr0;;                      # O channel offset = 0
    SET lr11 cr5;;                      # P query stride = 256

head_loop:
    SET lr7 cr0;;                       # t counter = 0
    ADD lr9 lr6 cr0;;                   # group P offset = head P offset (g=0)

t_loop:
    # ===================== group g = 0 (queries 0..127) =====================
    # ---- chunk 0: keys 0..127 ----
    ADD lr1 lr4 cr0;;                   # V chunk0 offset = chan
    LDR_CYCLIC_MULT_REG lr1 cr3 lr2;;   # R_CYCLIC = V[0..127, t]   (base VBASE)
    SUB lr0 lr9 lr11;;                  # P inner start = group P off - 256
    SET lr3 cr0;;                       # dest/inner counter = 0
g0c0_loop:
    LDR_MULT_REG r0 lr0 cr2; ADD lr0 lr0 lr11; INC lr3 1; MULT.RC.VV lr2 r0 0 lr2; AGG.SUM.FIRST lr3 1; BLT lr3 cr9 g0c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD lr1 lr4 cr6;;                   # V chunk1 offset = chan + 128
    LDR_CYCLIC_MULT_REG lr1 cr3 lr2;;   # R_CYCLIC = V[128..255, t]
    ADD lr0 lr9 cr6;;                   # P chunk1 base = group P off + 128
    SUB lr0 lr0 lr11;;                  # minus 256 startup
    SET lr3 cr0;;
g0c1_loop:
    LDR_MULT_REG r0 lr0 cr2; ADD lr0 lr0 lr11; INC lr3 1; MULT.RC.VV lr2 r0 0 lr2; AGG.SUM lr3 1; BLT lr3 cr9 g0c1_loop;;

    ADD lr5 lr10 cr0;;                  # O g=0 offset = O chan offset
    STR_ACC_REG lr5 cr4;;               # O[0..127, t] = R_ACC   (base OBASE)

    # ===================== group g = 1 (queries 128..255) ===================
    # ---- chunk 0: keys 0..127 ----
    ADD lr1 lr4 cr0;;                   # V chunk0 offset = chan (same channel)
    LDR_CYCLIC_MULT_REG lr1 cr3 lr2;;
    ADD lr0 lr9 cr7;;                   # g=1 P base = group P off + 32768
    SUB lr0 lr0 lr11;;                  # minus 256 startup
    SET lr3 cr0;;
g1c0_loop:
    LDR_MULT_REG r0 lr0 cr2; ADD lr0 lr0 lr11; INC lr3 1; MULT.RC.VV lr2 r0 0 lr2; AGG.SUM.FIRST lr3 1; BLT lr3 cr9 g1c0_loop;;

    # ---- chunk 1: keys 128..255 ----
    ADD lr1 lr4 cr6;;                   # V chunk1 offset = chan + 128
    LDR_CYCLIC_MULT_REG lr1 cr3 lr2;;
    ADD lr0 lr9 cr7;;                   # g=1 P base
    ADD lr0 lr0 cr6;;                   # + 128 (chunk1)
    SUB lr0 lr0 lr11;;                  # minus 256 startup
    SET lr3 cr0;;
g1c1_loop:
    LDR_MULT_REG r0 lr0 cr2; ADD lr0 lr0 lr11; INC lr3 1; MULT.RC.VV lr2 r0 0 lr2; AGG.SUM lr3 1; BLT lr3 cr9 g1c1_loop;;

    ADD lr5 lr10 cr12;;                 # O g=1 offset = O chan offset + 512
    STR_ACC_REG lr5 cr4;;               # O[128..255, t] = R_ACC

    # ----- next t: advance value-channel offset (+256 in) and O offset (+1024), t++ -----
    ADD lr4 lr4 cr5;;                   # chan += 256
    ADD lr10 lr10 cr13;;                # O chan offset += 1024
    INC lr7 1;;                         # t++
    BLT lr7 cr10 t_loop;;

    # ----- next head: head P offset += 65536, head++ -----
    ADD lr6 lr6 cr8;;                   # head P offset += 65536
    INC lr8 1;;
    BLT lr8 cr11 head_loop;;

end:
    BKPT;;
