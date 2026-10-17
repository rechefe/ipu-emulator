{#- matmul_x128_c2.asm: the template of the registry.toml entries
    matmul_192x192_x128, matmul_240x240_x128, matmul_720x240_x128,
    matmul_384x192_x128 and matmul_480x240_x128 (SiLU: the activation
    parameter renders ACTIVATE.QUANTIZE). The notes below are
    matmul_192x192_x128's; the shape reaches the program through
    kernel.toml's [abi] (rows and indices, not bytes; the byte figures below
    are the old harness's).

    Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 4 OutProj)

    Layer:   L4
    Scope:   single-stream
    Layout:  unpacked
    Shape:   N_TOK=64, K=192, N=192
    Related: L4 OutProj (single-stream). Sibling family: matmul_384x192_x128
             (FFN1, K=192->384, silu), matmul_576x192_x128 (QKV, K=192->576),
             matmul_192x384_x128 (FFN2, K=384->192). Consumed by
             proj_outproj_192_p4, which wraps the same arithmetic across all 4
             pixel streams in one invocation.
    Tests:   //src/tools/ipu-apps:test_matmul_192x192_x128

    Computes the L4 OutProj matmul: 192 input channels contracted down to 192
    output channels, over 64 tokens (a single token group, since N_TOK=64 fits
    in one 128-lane row). K=192 splits into two weight chunks (0..127 from R0,
    128..191 from R0 reloaded); the data pointer advances continuously across
    chunks while the per-chunk scalar index resets, with the first chunk's
    first iteration peeled to seed r_acc via ACC.ADD.FIRST and the remainder
    accumulated with ACC.ADD (per the MULT snapshot contract described below).

    Single token group (N_TOK=64 <= 128): one accumulate+store pass per output j.

    D: channel-major [K=192 channels, 128 tokens]  (64 valid, padded to 128)
       Row k at DATA_BASE + k*128
    W: output-major [192 out_ch, 192 in_ch], NO transposition; 2 chunks of <=128 bytes
       W[j, 0..127] at WEIGHTS_BASE + j*256 + 0
       W[j, 128..191] at WEIGHTS_BASE + j*256 + 128 (padded to 128)
    C: channel-major [192 out_ch, 64 tokens] FP32, packed at OUTPUT_BASE + j*256

    LR4 (data ptr) advances continuously by 128 across chunks; LR5 reset to -1 per chunk.
    Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT.S).

    MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
    the start-of-cycle snapshot while keeping the LR index live, so it cannot
    consume the chunk LD.MULT.CYCLIC loads in its own bundle. `;;` ends one
    VLIW word = one cycle = one snapshot, so a load and a MULT in the same bundle
    always run in the same cycle regardless of textual order -- co-issuing is
    fine, consuming the same-cycle load is not.
      chunk0 primes k=0's row, then each loop body multiplies the row loaded last
      cycle while prefetching the next. LR4 walks CONTINUOUSLY into chunk1 while
      LR5 resets, so chunk1's first row is ALREADY IN FLIGHT from chunk0's
      trailing prefetch: chunk1 must NOT re-prime.
      LR5 keeps its ADD co-issued with the MULT and the BLT.S (moving it to its own
      word would cost a cycle per contraction step). MULT reads LR5 LIVE and BLT.S
      reads the SNAPSHOT, exactly as before, so LR6/LR11 stay valid; only chunk0's
      startup is biased down by one (-1 -> -2) because the load now runs a bundle
      ahead. chunk1's startup is NOT biased -- chunk0's trailing prefetch already
      supplied that step of phase.

    CRs: CR0=DATA_BASE, CR9=WEIGHTS_BASE, CR2=WB+128, CR5=OUTPUT_BASE, CR6=-128 (data startup), CR8=-1 (chunk startup)
    LRs: LR0=0, LR2=128 (data stride), LR3=256 (output stride), LR6=126 (width-128 bound),
         LR7=0 (out ptr), LR8=0 (weight offset), LR9=0 (j), LR10=192 (j limit),
         LR11=62 (tail-chunk bound, width=64), LR12=256 (W_STRIDE)

    Memory layout:
      DATA:    192 x 128 B      =   24576 B (0x00000..0x05FFF)
      WEIGHTS: 192 rows x 256 B =   49152 B (0x10000..0x1BFFF)
      OUTPUT:  192 rows x 256 B =   49152 B (0x20000..0x2BFFF)
-#}
{%- set act = "silu" if activation == "silu" else "identity" -%}

j_loop:
    SET LR4 CR6;
    LD.MULT R0 LR8 CR9;;  # data startup -128; R0 = W[j, chunk0]
    SET LR5 CR8;;                            # chunk0 fixed_idx startup: -1
    SUB LR5 LR5 CR1;;                        # biased to -2 (load runs a bundle ahead)

    # Prime k=0's row for chunk0 (see the snapshot note in the header).
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;;

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
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
    # No re-prime: chunk1's first row is already in flight from chunk0's
    # trailing prefetch, and LR5 is NOT biased here for the same reason.
    SET LR5 CR8;
    LD.MULT R0 LR8 CR2;;  # chunk1 startup; R0 = W[j, chunk1]

k_chunk1:
    MULT.RC.VE LR0 LR5 0 LR0 CR15;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR11 k_chunk1;;

    ACTIVATE.QUANTIZE {{ act }} CR15;
    ST.AAQ LR7 CR5;;  # store 512B -> OUTPUT[j] (first 256B valid)
    ADD LR7 LR7 LR3;;                        # advance output ptr (packed)

    ADD LR8 LR8 LR12;
    ADD LR9 LR9 CR1;;  # next j
    BLT.S LR9 LR10 j_loop;;

end:
    HALT;;
