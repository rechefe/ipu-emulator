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
      chunk0 width=128 -> bound 126 (CR13.L);  chunk1 width=64 -> bound 62 (CR13.H).

    CRs: CR2=DATA_BASE, CR9=WEIGHTS_BASE (off read-only CR1; every chunk: LR1 = the chunk's row),
          CR3=OUTPUT_BASE
          CR6=-128 (data startup), CR8=-1 (per-chunk fixed_idx startup)
    LRs: the program sets LR0 (0), LR7 (out ptr), LR8 (weight offset) and LR9 (j) to 0 in the
         word before j_loop (LR0 in j_loop's first word) and LR4, LR5 itself, and LR1 (W[j]'s
         chunk row: LR8 + chunk, the offset of the chunk in table a) before each chunk load. The constants the
         loops read are CRs (kernel.toml's [abi]): CR11 = data stride, CR12 = W_STRIDE, CR10 = j
         limit, CR13.L = chunk0 bound (126, width 128), CR13.H = chunk1 bound (62, width 64).
         LR3's output stride is the hardwired CR1 (1 row).

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

    SET LR7 CR0;                             # output pointer, weight offset, j: all start at 0
    SET LR8 CR0;
    SET LR9 CR0;;

j_loop:
    SET LR4 CR6;
    SET LR0 CR0;                             # R_CYCLIC slot 0
    LD.MULT R0 LR8 CR9;;  # data startup -128; R0 = W[j, 0..127]
    SET LR5 CR8;;                            # chunk0 fixed_idx startup: -1
    SUB LR5 LR5 CR1;;                        # biased to -2 (load runs a bundle ahead)

    # Prime k=0's row for chunk0 (see the snapshot note in the header).
    LD.MULT.CYCLIC LR4 {{ cr.b }} LR0;
    ADD LR4 LR4 {{ cr.d_stride }};
    ADD LR5 LR5 CR1;;

    # Peeled first k-iter (k=0): ACC.FIRST seeds r_acc (replaces RESET_ACC).
    MULT.RC.VE LR0 LR5 0 LR0 CR15.L {{ cr.qf }}.L;
    ACC.ADD.FIRST;
    LD.MULT.CYCLIC LR4 {{ cr.b }} LR0;
    ADD LR4 LR4 {{ cr.d_stride }};
    ADD LR5 LR5 CR1;
    BLT.S LR5 {{ cr.bounds }}.L k_chunk0;;
    B after_chunk0;;

k_chunk0:
    MULT.RC.VE LR0 LR5 0 LR0 CR15.L {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 {{ cr.b }} LR0;
    ADD LR4 LR4 {{ cr.d_stride }};
    ADD LR5 LR5 CR1;
    BLT.S LR5 {{ cr.bounds }}.L k_chunk0;;

after_chunk0:
    # No re-prime: chunk1's first row is already in flight from chunk0's
    # trailing prefetch, and LR5 is NOT biased here for the same reason.
    SET LR5 CR8;
    ADD LR1 LR8 CR1;
    LD.MULT R0 LR1 CR9;;  # chunk1 startup; R0 = W[j, 128..191]+zeros

k_chunk1:
    MULT.RC.VE LR0 LR5 0 LR0 CR15.L {{ cr.qf }}.L;
    ACC.ADD;
    LD.MULT.CYCLIC LR4 {{ cr.b }} LR0;
    ADD LR4 LR4 {{ cr.d_stride }};
    ADD LR5 LR5 CR1;
    BLT.S LR5 {{ cr.bounds }}.H k_chunk1;;

    ACTIVATE.EXACT identity CR15.L CR3 LR7;;  # store 512B -> OUTPUT[j] (first 256B valid)
    ADD LR7 LR7 CR1;;                        # advance output ptr (packed, 256B)

    ADD LR8 LR8 {{ cr.w_stride }};
    ADD LR9 LR9 CR1;;  # next j: weight offset += W_STRIDE, j++
    BLT.S LR9 {{ cr.n }} j_loop;;

end:
    HALT;;
