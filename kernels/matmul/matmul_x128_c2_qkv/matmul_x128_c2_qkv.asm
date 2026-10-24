{#- matmul_x128_c2_qkv.asm: the template of the registry.toml entries
    matmul_576x192_x128 (matmul_x128_c2's body with its output base in CR3).
    The notes below are matmul_576x192_x128's; the shape reaches the program
    through kernel.toml's [abi] (rows and indices, not bytes; the byte
    figures below are the old harness's).

    Transformer matmul: C[j, t] = sum_k W[j, k] * D[k, t]   (Layer 4 QKV)

    Layer:   L4
    Scope:   single-stream
    Layout:  unpacked
    Shape:   N_TOK=64, K=192, N=576
    Related: L4 fused QKV projection (single-stream, 3x output = Q+K+V stacked,
             each 192 wide). Sibling family: matmul_192x192_x128 (OutProj),
             matmul_384x192_x128 (FFN1), matmul_192x384_x128 (FFN2). Consumed
             by proj_qkv_192_p4 (all 4 streams, one invocation, identity
             activation) and by the seam test
             the mobilevit chains' test_seam_qkv_matmul_to_scores_km.py (this
             kernel's Q rows feed attn_scores_km_64x48 by row selection).
    Tests:   //kernels/matmul:matmul_576x192_x128 (the registry entry)

    Computes the L4 fused QKV matmul: 192 input channels projected to 576
    output channels (Q, K, V concatenated, 192 each), over 64 tokens (single
    token group). K=192 splits into two weight chunks ([128, 64] widths) with
    the data pointer advancing continuously and the per-chunk scalar index
    resetting; the first chunk's first iteration is peeled to seed r_acc via
    ACC.ADD.FIRST, the rest accumulate with ACC.ADD. Store activation is
    identity.

    Single token group (N_TOK=64 <= 128): one accumulate+store pass per output j.

    D: channel-major [K=192 channels, 128 tokens]  (64 valid tokens, padded to 128)
       Row k at DATA_BASE + k*128
    W: output-major [576 out_ch, 192 in_ch], NO transposition
       W[j, 0..127]   at WEIGHTS_BASE + j*256
       W[j, 128..191] at WEIGHTS_BASE + j*256 + 128 (padded to 128 bytes)
    C: channel-major [576 out_ch, 64 tokens] FP32, packed contiguously
       Row j at OUTPUT_BASE + j*256  (256 = 64 tokens * 4 bytes)

    Algorithm: K=192 split into two chunks per j ([128, 64]).
      Per chunk: load W[j, chunk*128..] into R0, run width k-steps.
      MULT.VE.CYCLIC R0[LR5] x r_cyclic[:] — LR5 cycles 0..width-1 per chunk.
      LR4 (data pointer) advances continuously by 128; NOT reset between chunks.
      LR5 reset to -1 at the start of each chunk via SET LR5 CR8.

    Loop-bound formula (do-while, live MULT/XMEM, snapshot BLT.S):
      body runs for live index [lr5_start+1 .. bound+1]; bound = width-2.
      chunk0 width=128 -> bound 126 (LR6);  chunk1 width=64 -> bound 62 (LR11).

    CRs: CR0=DATA_BASE, CR9=WEIGHTS_BASE (off read-only CR1), CR2=WEIGHTS_BASE+128,
          CR3=OUTPUT_BASE
          CR6=-128 (data startup), CR8=-1 (per-chunk fixed_idx startup)
    LRs (preset by harness):
      LR0=0    (const: r_cyclic write-index 0)
      LR2=128  (data stride: 128 bytes/channel)
      LR3=256  (output stride: 64 tokens * 4 bytes, packed)
      LR6=126  (chunk0 bound: width=128)
      LR7=0    (output pointer, incremented by LR3 each j)
      LR8=0    (weight byte offset, incremented by LR12 each j)
      LR9=0    (j counter)
      LR10=576 (j-loop limit)
      LR11=62  (chunk1 bound: width=64)
      LR12=256 (W_STRIDE = 256 bytes per j)

    Memory layout:
      DATA:    192 x 128 B       =  24 576 B (0x00000..0x05FFF)
      WEIGHTS: 576 rows x 256 B  = 147 456 B (0x10000..0x33FFF)
      OUTPUT:  576 rows x 256 B  = 147 456 B (0x40000..0x63FFF, +256B spill)

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
      reads the SNAPSHOT, exactly as before, so the bounds stay valid; only
      chunk0's startup is biased down by one (-1 -> -2) because the load now runs
      a bundle ahead. chunk1's startup is NOT biased -- chunk0's trailing prefetch
      already supplied that step of phase.
-#}

j_loop:
    SET LR4 CR6;
    LD.MULT R0 LR8 CR9;;  # data startup -128; R0 = W[j, 0..127]
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
    # No re-prime: chunk1's first row is already in flight from chunk0's
    # trailing prefetch, and LR5 is NOT biased here for the same reason.
    SET LR5 CR8;
    LD.MULT R0 LR8 CR2;;  # chunk1 startup; R0 = W[j, 128..191]+zeros

k_chunk1:
    MULT.RC.VE LR0 LR5 0 LR0 CR15.L {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 CR0 LR0;
    ADD LR4 LR4 LR2;
    ADD LR5 LR5 CR1;
    BLT.S LR5 LR11 k_chunk1;;

    ACTIVATE.EXACT identity CR15.L CR3 LR7;;  # store 512B -> OUTPUT[j] (first 256B valid)
    ADD LR7 LR7 LR3;;                        # advance output ptr (packed, 256B)

    ADD LR8 LR8 LR12;
    ADD LR9 LR9 CR1;;  # next j: weight offset += W_STRIDE, j++
    BLT.S LR9 LR10 j_loop;;

end:
    HALT;;
