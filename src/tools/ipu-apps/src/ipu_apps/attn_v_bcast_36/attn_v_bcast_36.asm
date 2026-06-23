# attn@V, key-major scores -> channel-major output  (Agent B, broadcast / no AGG)
#
# Per attention head h in [0,4):
#   O[i,t] = sum_s P[i,s] * V[s,t]
#     i = query in [0,256)  (2 groups of 128 = the SIMD lanes)
#     t = head channel in [0,36)   (global value channel = h*36 + t)
#     s = key (contraction) in [0,256)
#
# LANES = QUERIES.  Standard matmul-broadcast template (V in the weight/scalar
# role, P key-major in the streamed-data role):
#   mult_res[i] = P[i,s] * V[s,t]   ;  ACC[.FIRST] accumulates over s
#   After 256 keys, R_ACC[i] = O[i,t] for the 128 queries of this group.
# No AGG, no collision: every cycle is a full-width scalar*vector MULT + ACC, and
# all 256 values of channel (h,t) live in R0++R1, so the scalar index s (0..255)
# selects V[s,t] directly with NO mid-loop reload.
#
# Inputs (1 byte/element):
#   P key-major (head-major): P[i,s] at PBASE + h*65536 + s*256 + i.
#     Group g, key s -> 128 queries at PBASE + h*65536 + s*256 + g*128.
#   V channel-major: V[s, chan] at VBASE + chan*256 + s,  chan = h*36 + t.
#     Per channel the 256 values load into R0 (s=0..127) and R1 (s=128..255).
# Output (FP32 R_ACC, 512-byte group rows — transformer-matmul convention):
#   O[i,t] at OBASE + chan*1024 + g*512 + local*4,  i = g*128 + local.
#
# VLIW timing (same as matmul_144x144 inner loop):
#   LR slot runs first; XMEM + MULT then read LR values LIVE (post-ADD); BLT reads
#   the SNAPSHOT (pre-ADD).  Single-cycle inner body:
#     LDR_CYCLIC P[h,g,s_live]; ADD data ptr; ADD s; MULT.RC.VE src=s_live; ACC; BLT
#   Startup offsets (one-cycle pipeline align):
#     data ptr  lr4 = first_addr - 256   (ADD +256 -> live = first_addr)
#     key index lr5 = -1                  (ADD +1   -> live s = 0)
#     BLT bound cr8 = 254  (first_index=0, width=256 -> 0+256-2=254)
#
# CRs (set by harness; loop bounds in CRs to free LRs):
#   cr0  = 0            (hardwired; const-0, also r_cyclic XMEM base)
#   cr1  = 1            (read-only hardwired constant; loop steps)
#   cr2  = PBASE        (key-major scores base)
#   cr3  = VBASE        (channel-major value base; R0 = V[0:127, chan])
#   cr4  = OUTPUT_BASE  (channel-major output base)
#   cr5  = 128          (R1 source offset within a V channel)
#   cr6  = -1           (key-index startup)
#   cr7  = 65536        (P head stride = 256 keys * 256 queries)
# Loop bounds are count-1 (counter ADD shares the BLT bundle -> BLT reads the
# pre-ADD snapshot; branch is taken while snapshot < bound):
#   cr8  = 254          (key-loop bound: width 256, peeled + startup)
#   cr9  = 35           (t-loop bound: 36 channels per head)
#   cr10 = 3            (head-loop bound: 4 heads)
#   cr11 = 1            (g-loop bound: 2 groups)
# LRs (set by harness):
#   lr0  = 0            (const: r_cyclic index 0; mask_shift)
#   lr1  = 256          (P key stride / V channel stride)
#   lr2  = 512          (output-row stride)
#   lr3  = 128          (group query offset within a key column)
# Maintained / temporary registers:
#   lr4  = P data ptr      lr5  = key index s
#   lr6  = head counter    lr7  = channel (t) counter   lr9  = group counter
#   lr10 = chan*256       (R0 source offset; += 256 per channel)
#   lr11 = chan*256 + 128 (R1 source offset; += 256 per channel)
#   lr12 = h*65536        (P head base; += 65536 per head)
#   lr13 = g*128          (P group offset; reset 0 per channel, += 128 per g)
#   lr14 = output offset  (+= 512 per STR; rows visited in (h,t,g) order)

    SET     lr10 cr0;;                 # R0 source offset = chan*256 = 0
    SET     lr11 cr5;;                 # R1 source offset = chan*256 + 128 = 128
    SET     lr12 cr0;;                 # P head base = 0
    SET     lr14 cr0;;                 # output offset = 0

    SET     lr6  cr0;;                 # head counter (0..3)
h_loop:

    SET     lr7  cr0;;                 # channel (t) counter (0..35)
t_loop:

    # -- load V[:, chan] into R0 (s=0..127) and R1 (s=128..255) ----------------
    LDR_MULT_REG r0 lr10 cr3;;         # R0 = V[0:127,   chan]  (VBASE + chan*256)
    LDR_MULT_REG r1 lr11 cr3;;         # R1 = V[128:255, chan]  (VBASE + chan*256 + 128)

    SET     lr13 cr0;;                 # P group offset = g*128 = 0
    SET     lr9  cr0;;                 # g counter (0..1)
g_loop:

    # -- key loop: s = 0..255 over the 128-query group g -----------------------
    # P[h,g,s] address = PBASE + h*65536 + s*256 + g*128.  data ptr lr4 = start-256.
    SET     lr4  cr2;;                 # PBASE
    ADD     lr4  lr4  lr12;;           # + h*65536
    ADD     lr4  lr4  lr13;;           # + g*128  -> P[h,g,s=0]
    SUB     lr4  lr4  lr1;;            # - 256 startup (ADD +256 -> live s=0)
    SET     lr5  cr6;;                 # key index startup = -1 (ADD +1 -> s=0)

    # Peeled first key (s=0): ACC.FIRST seeds r_acc.
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr1; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC.FIRST; BLT lr5 cr8 s_loop;;
    B s_done;;
s_loop:
    LDR_CYCLIC_MULT_REG lr4 cr0 lr0; ADD lr4 lr4 lr1; ADD lr5 lr5 cr1;
    MULT.RC.VE lr0 lr5 0 lr0; ACC; BLT lr5 cr8 s_loop;;
s_done:

    # -- store channel-major row O[g*128:+128, chan] ---------------------------
    STR_ACC_REG lr14 cr4;;             # 512B FP32 -> OBASE + chan*1024 + g*512
    ADD     lr14 lr14 lr2;;            # advance output-row offset by 512

    ADD     lr13 lr13 lr3;;            # next group: P offset += 128
    ADD     lr9  lr9  cr1; BLT lr9 cr11 g_loop;;

    ADD     lr10 lr10 lr1;;            # next channel: R0 source += 256
    ADD     lr11 lr11 lr1;;            # next channel: R1 source += 256
    ADD     lr7  lr7  cr1; BLT lr7 cr9 t_loop;;

    ADD     lr12 lr12 cr7;;            # next head: P head base += 65536
    ADD     lr6  lr6  cr1; BLT lr6 cr10 h_loop;;

end:
    BKPT;;
