{#- matmul_x128_c3.asm: the template of the registry.toml entries
    matmul_192x384_x128. The notes below are matmul_192x384_x128's; the
    shape reaches the program through kernel.toml's [abi] (rows and indices,
    not bytes; the byte figures below are the old harness's).

    Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 4 FFN2)

    Layer:   L4
    Scope:   single-stream
    Layout:  unpacked
    Shape:   N_TOK=64, K=384, N=192
    Related: L4 FFN2 (single-stream, contracts the 384-wide FFN hidden back to
             192). Sibling family: matmul_192x192_x128 (OutProj),
             matmul_384x192_x128 (FFN1, the expansion this kernel reverses),
             matmul_576x192_x128 (QKV). Consumed by proj_ffn2_192_p4 (all 4
             streams, one invocation, identity activation). The L4
             layernorm->matmul seam (the mobilevit chains' test_seam_layernorm_to_matmul.py)
             runs its sibling matmul_192x192_x128.
    Tests:   //kernels/matmul:matmul_192x384_x128 (the registry entry)

    Computes the L4 FFN2 matmul: the 384-wide FFN hidden representation
    contracted back down to 192 output channels, over 64 tokens (single token
    group). K=384 splits into three weight chunks (widths [128, 128, 128])
    with the data pointer advancing continuously across all three and the
    per-chunk scalar index resetting; the first chunk's first iteration is
    peeled to seed r_acc via ACC.ADD.FIRST, every other step accumulates with
    ACC.ADD. Store activation is identity.

    Single token group (N_TOK=64 <= 128): one accumulate+store pass per output j.

    D: channel-major [K=384 channels, 128 tokens]  (64 valid, padded to 128)
       Row k at DATA_BASE + k*128
    W: output-major [192 out_ch, 384 in_ch], NO transposition; 3 chunks of <=128 bytes
       W[j, 0..127] at WEIGHTS_BASE + j*384 + 0
       W[j, 128..255] at WEIGHTS_BASE + j*384 + 128
       W[j, 256..383] at WEIGHTS_BASE + j*384 + 256
    C: channel-major [192 out_ch, 64 tokens] FP32, packed at OUTPUT_BASE + j*256

    LR4 (data ptr) advances continuously by 128 across chunks; LR5 reset to -1 per chunk.
    Loop bound per chunk = width-2 (do-while, live MULT/XMEM, snapshot BLT.S).

    CRs: CR0=DATA_BASE, CR9=WEIGHTS_BASE, CR2=WB+128, CR3=WB+256, CR5=OUTPUT_BASE, CR6=-128 (data startup), CR8=-1 (chunk startup)
    LRs: LR0=0, LR2=128 (data stride), LR3=256 (output stride), LR6=126 (width-128 bound),
         LR7=0 (out ptr), LR8=0 (weight offset), LR9=0 (j), LR10=192 (j limit),
         LR11=126 (tail-chunk bound, width=128), LR12=384 (W_STRIDE)

    Memory layout:
      DATA:    384 x 128 B      =   49152 B (0x00000..0x0BFFF)
      WEIGHTS: 192 rows x 384 B =   73728 B (0x10000..0x21FFF)
      OUTPUT:  192 rows x 256 B =   49152 B (0x30000..0x3BFFF)

    MULT SNAPSHOT CONTRACT (issue #157): MULT.RC.VE reads its r_cyclic DATA from
    the start-of-cycle snapshot while keeping the LR index live, so it cannot
    consume the chunk LD.MULT.CYCLIC loads in its own bundle. `;;` ends one
    VLIW word = one cycle = one snapshot, so a load and a MULT in the same bundle
    always run in the same cycle regardless of textual order -- co-issuing is
    fine, consuming the same-cycle load is not.
      chunk0 primes k=0's row, then each loop body multiplies the row loaded last
      cycle while prefetching the next. LR4 walks CONTINUOUSLY across all THREE
      chunks while LR5 resets per chunk, so each later chunk's first row is
      ALREADY IN FLIGHT from the previous chunk's trailing prefetch: only chunk0
      primes, and chunk1/chunk2 must not.
      LR5 keeps its ADD co-issued with the MULT and the BLT.S (moving it to its own
      word would cost a cycle per contraction step). MULT reads LR5 LIVE and BLT.S
      reads the SNAPSHOT, exactly as before, so LR6 stays valid; only chunk0's
      startup is biased down by one (-1 -> -2) because the load now runs a bundle
      ahead. chunk1/chunk2 startups are NOT biased.
-#}

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
    MULT.RC.VE LR0 LR5 0 LR0 CR15.L {{ cr.qf }}.L;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR6 k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE LR0 LR5 0 LR0 CR15.L {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR6 k_chunk0;;

after_chunk0:

    # No re-prime/bias: chunk1's first row is already in flight.
    SET LR5 CR8;
    LD.MULT R0 LR8 CR2;;  # chunk1 startup; R0 = W[j, chunk1]

k_chunk1:
    MULT.RC.VE LR0 LR5 0 LR0 CR15.L {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR6 k_chunk1;;

    # No re-prime/bias: chunk2's first row is already in flight.
    SET LR5 CR8;
    LD.MULT R0 LR8 CR3;;  # chunk2 startup; R0 = W[j, chunk2]

k_chunk2:
    MULT.RC.VE LR0 LR5 0 LR0 CR15.L {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR6 k_chunk2;;

    ACTIVATE.EXACT identity CR15.L CR5 LR7;;  # store 512B -> OUTPUT[j] (first 256B valid)
    ADD LR7 LR7 LR3;;                        # advance output ptr (packed)

    ADD LR8 LR8 LR12;
    ADD LR9 LR9 CR1;;  # next j
    BLT.S LR9 LR10 j_loop;;

end:
    HALT;;
