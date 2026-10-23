{#- matmul_x128_c4.asm: the template of the registry.toml entries
    matmul_240x480_x128. The notes below are matmul_240x480_x128's; the
    shape reaches the program through kernel.toml's [abi] (rows and indices,
    not bytes; the byte figures below are the old harness's).

    Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 5 FFN2)

    Layer:   L5
    Scope:   single-stream
    Layout:  unpacked
    Shape:   N_TOK=16, K=480, N=240
    Related: L5 FFN2 (single-stream, contracts the 480-wide FFN hidden back to
             240). Sibling family: matmul_240x240_x128 (OutProj),
             matmul_480x240_x128 (FFN1, the expansion this kernel reverses),
             matmul_720x240_x128 (QKV). Consumed by proj_ffn2_240_p4 (all 4
             streams, one invocation, identity activation) and feeds
             residual_add_16x240 downstream (full-row output).
    Tests:   //kernels/matmul:matmul_240x480_x128 (the registry entry)

    Computes the L5 FFN2 matmul: the 480-wide FFN hidden representation
    contracted back down to 240 output channels, over 16 tokens (single token
    group; ROW-numbered operands per issue #179). K=480 splits into four
    weight-row chunks with the data pointer advancing one row at a time
    continuously across all four and the per-chunk scalar index resetting;
    the first chunk's first iteration is peeled to seed r_acc via
    ACC.ADD.FIRST, every other step accumulates with ACC.ADD. Store
    activation is identity.

    Single token group (N_TOK=16 <= 128): one accumulate+store pass per output j.

    All .asm operands are ROW numbers (issue #179), one row = 128 lanes.
    D: channel-major [K=480 channels, 16 tokens], one channel per WHOLE row (16
       valid, padded to 128).  Row k at DATA_BASE + k.
    W: output-major [240 out_ch, 480 in_ch], NO transposition; 4 chunks/row-quads
       per output channel.
       W[j, 0..127] at WEIGHTS_BASE + j*4 + 0
       W[j, 128..255] at WEIGHTS_BASE + j*4 + 1
       W[j, 256..383] at WEIGHTS_BASE + j*4 + 2
       W[j, 384..479] at WEIGHTS_BASE + j*4 + 3 (padded to a whole row)
    C: channel-major [240 out_ch, 16 tokens] FP32, one row per channel at
       OUTPUT_BASE + j.

    LR4 (data ptr) advances continuously by 1 row across chunks; LR5 reset to -1 per chunk.
    Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT.S).

    CRs: CR0=DATA_BASE, CR9=WEIGHTS_BASE, CR2=WB+1 row, CR3=WB+2 rows, CR4=WB+3 rows, CR5=OUTPUT_BASE, CR6=-1 row (data startup), CR8=-1 (chunk startup)
    LRs: LR0=0, LR2=1 row (data stride), LR3=1 row (output stride), LR6=126 (width-128 bound),
         LR7=0 (out ptr), LR8=0 (weight offset), LR9=0 (j), LR10=240 (j limit),
         LR11=94 (tail-chunk bound, width=96), LR12=4 rows (W_STRIDE)

    Memory layout (row counts):
      DATA:    480 rows
      WEIGHTS: 960 rows (240 output channels x 4 rows/channel)
      OUTPUT:  240 rows
-#}

j_loop:
    SET LR4 CR6;
    LD.MULT R0 LR8 CR9;;  # data startup -128; R0 = W[j, chunk0]
    SET LR5 CR8;;                            # chunk0 fixed_idx startup: -1
    SUB LR5 LR5 CR1;;                        # biased to -2 (load runs a bundle ahead)

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    # Prime k=0's chunk one bundle ahead: MULT.RC.VE reads r_cyclic from the
    # start-of-cycle snapshot (issue #157), so it cannot consume a chunk loaded
    # in its OWN bundle -- it would multiply against the previous row. chunk1+
    # must NOT re-prime: their first row is already in flight from the previous
    # chunk's trailing prefetch.
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;;

    MULT.RC.VE LR0 LR5 0 LR0 CR15;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR6 k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE LR0 LR5 0 LR0 CR15;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR6 k_chunk0;;

after_chunk0:

    SET LR5 CR8;
    LD.MULT R0 LR8 CR2;;  # chunk1 startup; R0 = W[j, chunk1]

k_chunk1:
    MULT.RC.VE LR0 LR5 0 LR0 CR15;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR6 k_chunk1;;

    SET LR5 CR8;
    LD.MULT R0 LR8 CR3;;  # chunk2 startup; R0 = W[j, chunk2]

k_chunk2:
    MULT.RC.VE LR0 LR5 0 LR0 CR15;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR6 k_chunk2;;

    SET LR5 CR8;
    LD.MULT R0 LR8 CR4;;  # chunk3 startup; R0 = W[j, chunk3]

k_chunk3:
    MULT.RC.VE LR0 LR5 0 LR0 CR15;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR11 k_chunk3;;

    # Hardware store path (see docs/content/specs/MAS-stage-aaq-str.md section 7.0):
    # AaQ and STR are consecutive pipeline stages WITHIN one VLIW word
    # (CTRL -> MULT -> ACC -> AaQ -> STR), so STR consumes this cycle's AaQ
    # result and the store is free. `identity` + valid_elements=128 (the CR
    # default) makes it a lane-for-lane FP32 copy of r_acc in wide mode.
    ACTIVATE.QUANTIZE identity CR15;
    ST.AAQ LR7 CR5;;  # store 512B -> OUTPUT[j] (first 64B valid)
    ADD LR7 LR7 LR3;;                        # advance output ptr (packed)

    ADD LR8 LR8 LR12;
    ADD LR9 LR9 CR1;;  # next j
    BLT.S LR9 LR10 j_loop;;

end:
    HALT;;
