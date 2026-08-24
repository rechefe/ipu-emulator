# Fully-connected layer: out[s, j] = sum_i in[s, i] * W[j, i], one sample
# per outer iteration, contraction width 128 (one full SIMD vector).
#
# Layer:   n/a (generic FC primitive, not tied to a MobileViT L3/L4/L5 shape)
# Scope:   single-stream
# Layout:  unpacked
# Shape:   128 input neurons x 64 output neurons, 10 samples/rows
# Status:  validated
# Related: matmul_128x64x128.asm (same contraction width, this kernel's
#          single-sample-per-row streaming shape is the basis it was built
#          from); weights are pre-transposed by the harness (see
#          fully_connected/__init__.py's _load_and_transpose_weights) so
#          element i of every output neuron lands in one XMEM row.
# Tests:   bazel test //src/tools/ipu-apps:test_fully_connected_wide
#
# Outer loop: one iteration per sample s (lr0 counts samples, bound cr7 =
# SAMPLES_NUM). Inner loop: one iteration per input neuron i (lr5 counts
# elements 0..INPUT_NEURONS-1, bound cr11), broadcasting in[s,i] from r0
# via MULT.RC.VE against the weight row streamed through r_cyclic, ACC.ADD
# (.FIRST at i=0) accumulating all 128 contractions into one output row.
#
# MULT reads r_cyclic from the start-of-cycle snapshot (issue #157), so the
# chunk consumed by MULT.RC.VE must be loaded a cycle earlier than it's
# used -- the loop is peeled to prime element 0's load before entering, then
# each body loads the NEXT element's chunk while multiplying the chunk
# loaded last cycle (see the load/index-scheduling comment inline below).
#
# Store path: ACTIVATE.QUANTIZE and STR_POST_AAQ_REG are consecutive
# pipeline stages within one VLIW word (CTRL -> MULT -> ACC -> AaQ -> STR,
# docs/content/specs/stage-aaq-str.md section 7.0), so the store costs no
# extra cycle. This replaces the simulation-only STR_ACC_REG, which has no
# hardware equivalent.
#
# BREAK;; before the outer-loop BLT is a debugger single-step aid only --
# in the normal ("run") harness mode BREAK is a no-op and BLT executes
# unconditionally on the same pass, so the outer sample loop runs for real.
    SET                 lr0 cr6 ;;
    SET                 lr1 cr7 ;;
    SET                 lr2 cr8 ;;

input_loop:
    LDR_MULT_REG        r0 lr0 cr0;;

    SET                 lr5 cr10 ;;
    SET                 lr6 cr11 ;;
    SET                 lr15 cr12 ;;

    {#- MULT reads r_cyclic from the start-of-cycle snapshot (issue #157), so
        the chunk consumed by MULT.RC.VE must be loaded a cycle earlier than
        it's used: prime element 0's load here, then each loop body loads the
        NEXT element's chunk while multiplying the chunk loaded last cycle.
        lr5 (R0 scalar-select index, read LIVE by MULT) must equal the CURRENT
        element on the cycle MULT runs;
        lr14 (load offset, also read LIVE, by;
        LDR) must already hold the NEXT element's offset on that same cycle.
        LR sub-slots run before LOAD/MULT within a word, so lr14's increment
        for element e+1 must land in the word BEFORE the load that consumes
        it -- it cannot share a word with that load (the load would see the
        already-bumped value and skip an element). lr5 shares a word with the
        MULT that consumes it, same as the original code, so MULT sees the
        element lr5 was just advanced to. -#}
    SET                 lr14 cr9 ;;                         {#- lr14 = -128 -#}
    ADD                 lr14 lr14 cr3 ;;                    {#- lr14 = element 0's offset (0) -#}
    LDR_CYCLIC_MULT_REG lr14 cr13 lr15 ;;                   {#- load row 0 (lr14 unchanged this word) -#}
    ADD                 lr14 lr14 cr3 ;
    ADD                 lr5  lr5  cr4 ;;                     {#- lr14 -> element 1's offset; lr5 -> 0 -#}

    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD.FIRST;
    LDR_CYCLIC_MULT_REG lr14 cr13 lr15;
    BNE                 lr5 lr6 element_loop_pre;;
    B                   after_element_loop;;

element_loop_pre:
    ADD                 lr14 lr14 cr3;
    ADD                 lr5 lr5 cr4;;                        {#- lr14 -> next load offset; lr5 -> element just loaded -#}

element_loop:
    MULT.RC.VE          lr15 lr5 0 lr15 cr15;
    ACC.ADD;
    LDR_CYCLIC_MULT_REG lr14 cr13 lr15;
    BNE                 lr5 lr6 element_loop_pre;;

after_element_loop:
    # Hardware store path: AaQ and STR are consecutive pipeline stages WITHIN
    # one VLIW word (CTRL -> MULT -> ACC -> AaQ -> STR, see
    # docs/content/specs/stage-aaq-str.md section 7.0), so STR consumes this
    # cycle's AaQ result and the store costs no extra cycle. Replaces the
    # simulation-only STR_ACC_REG, which is not implemented in real hardware.
    ACTIVATE.QUANTIZE   identity cr15;
    STR_POST_AAQ_REG lr7 cr2;;
    ADD                 lr7 lr7 cr5;
    ADD                 lr0 lr0 cr3;;

    BREAK;;

    BLT                 lr0 lr1 input_loop;;

end:
    BKPT;;
