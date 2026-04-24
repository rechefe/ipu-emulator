# Softmax-256 Assembly Implementation
# Applies row-wise softmax to 256x256 matrix using numerically stable algorithm:
#   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
#
# Memory layout:
#   CR0 = input/output base (256 rows × 2 chunks × 128 bytes = 65536 bytes, in-place)
#   CR1 = constant region base:
#           [+0,   128B] all elements = -1  (used to negate max via mult.ve.aaq)
#           [+128, 128B] all elements = log2(e) ≈ 1.4427 (used in log_loop)
#   CR9 = identity scalar = 1  (used by mult.ve.cr for passthrough)
#
# Register allocation:
#   lr0  = row counter (0..255)
#   lr1  = total rows = 256
#   lr2  = chunk counter within row (0..1)
#   lr3  = chunks per row = 2
#   lr4  = row base chunk offset = lr0 * 2  (chunk index of first chunk in current row)
#   lr5  = current chunk address (temp, recomputed each sub-loop)
#   lr6  = 0  (zero constant: cyclic S0 offset, mask slot 0, mask shift 0)
#   lr7  = 128 (cyclic S1 offset — used when loading constants into S1)
#   lr8  = 1  (loop step)
#
# AAQ register usage (changes per sub-step within each row):
#   aaq0 = row max (written by find_max, read by negate_max and subtract_loop)
#   aaq1 = negated row max = -max (written by negate_max, read by subtract_loop)
#   aaq2 = log2(e) constant (written during init, read by log_loop)
#   aaq3 = 1/sum(exp) = reciprocal of normalisation sum (written by sum_loop, read by div_loop)
#
# NOTE: pow_loop (2^x approximation) is not implemented — the ISA has no native
#       exp or 2^x instruction. Options: LUT in CR memory, polynomial approximation
#       via repeated mult.ve, or a future dedicated instruction.

# ===========================================================================
# Initialisation: load log2(e) constant into aaq2 once (shared across all rows)
# ===========================================================================
#
# Strategy: load the 128-byte log2(e) chunk (CR1+128) into r_cyclic S0,
# then passthrough via mult.ve.cr (identity), acc.first, agg max → aaq2.
# Since all 128 elements are the same value, max = that value. ✓

    set     lr6  0;
    set     lr7  128;;

    set     lr8  1;;          # step size and second-chunk offset

    # Load log2(e) chunk (CR1 offset 128) into r_cyclic S0
    ldr_cyclic_mult_reg lr7 cr1 lr6;;

    # Passthrough: mult_res = r_cyclic[S0] * 1, acc.first, then aggregate max → aaq2
    mult.ve.cr  lr6 lr6 lr6 cr9;
    acc.first;;
    agg max value cr0 aaq2;;

    set     lr0  0;
    set     lr1  256;;

    set     lr3  2;;          # chunks per row

# ===========================================================================
# Row loop (256 rows)
# ===========================================================================

row_loop:

    # Compute row base: lr4 = lr0 * 2  (chunk index of row start)
    add     lr4 lr0 lr0;;

    # =========================================================================
    # Step 1: Find row maximum across both 128-element chunks
    # =========================================================================
    #
    # Load chunk 0, passthrough, acc.first → r_acc = chunk0 data.
    # agg max seeds aaq0 from r_acc (max of 128 elements).
    # Load chunk 1, acc.max.first aaq0 → r_acc[i] = max(chunk1[i], aaq0).
    # agg max again → aaq0 = global row max.
    #
    # Note: acc.max.first semantics: r_acc[i] = max(mult_res[i], aaq[idx]).
    # After ldr_cyclic + mult.ve.cr passthrough, mult_res = chunk1 data.
    # So r_acc[i] = max(chunk1[i], row_max_so_far), then agg max gives true row max.

    # Chunk 0 → S0
    ldr_cyclic_mult_reg lr4 cr0 lr6;;

    mult.ve.cr  lr6 lr6 lr6 cr9;
    acc.first;;

    # agg max with aaq0's current value — first call just overwrites with max(chunk0)
    agg max value cr0 aaq0;;

    # Chunk 1: offset = lr4 + 1
    add     lr5 lr4 lr8;;
    ldr_cyclic_mult_reg lr5 cr0 lr6;;

    # acc.max.first: r_acc[i] = max(mult_res[i]=chunk1[i], aaq0=chunk0_max)
    mult.ve.cr  lr6 lr6 lr6 cr9;
    acc.max.first aaq0;;

    # agg max: aaq0 = max over r_acc = true row max
    agg max value cr0 aaq0;;

    # =========================================================================
    # Step 2: Compute -max → store in aaq1
    # =========================================================================
    #
    # Load the all-(-1) chunk from CR1+0 into r_cyclic S0.
    # mult.ve.aaq: mult_res[i] = r_cyclic[S0+i] * aaq0[max] = (-1) * max = -max.
    # acc.first → r_acc[i] = -max (all 128 lanes identical).
    # agg max → aaq1 = max(r_acc) = -max. ✓

    ldr_cyclic_mult_reg lr6 cr1 lr6;;

    mult.ve.aaq lr6 lr6 lr6 aaq0;
    acc.first;;

    agg max value cr0 aaq1;;

    # =========================================================================
    # Step 3: Subtract max from each element (x_i - max), store in-place
    # =========================================================================
    #
    # For each chunk: load → r_cyclic, passthrough via mult.ve.cr,
    # acc.add_aaq.first aaq1 → r_acc[i] = data[i] + (-max) = data[i] - max.
    # Store back.
    #
    # blt timing: incr lr2 in the cycle BEFORE blt (LR writes live; blt reads snapshot).

    set     lr2 0;;

subtract_loop:

    add     lr5 lr4 lr2;;
    ldr_cyclic_mult_reg lr5 cr0 lr6;;

    mult.ve.cr  lr6 lr6 lr6 cr9;
    acc.add_aaq.first aaq1;;

    str_acc_reg lr5 cr0;;

    incr    lr2 1;;           # incr in separate cycle so blt reads updated snapshot
    blt     lr2 lr3 subtract_loop;;

    # =========================================================================
    # Step 4: Multiply each element by log2(e) to prepare for 2^x, store in-place
    # =========================================================================
    #
    # aaq2 holds log2(e) (set during init).
    # For each chunk: load → r_cyclic, mult.ve.aaq → mult_res[i] = data[i] * log2(e),
    # acc.first, store.

    set     lr2 0;;

log_loop:

    add     lr5 lr4 lr2;;
    ldr_cyclic_mult_reg lr5 cr0 lr6;;

    mult.ve.aaq lr6 lr6 lr6 aaq2;
    acc.first;;

    str_acc_reg lr5 cr0;;

    incr    lr2 1;;
    blt     lr2 lr3 log_loop;;

    # =========================================================================
    # Step 5: Compute 2^(log2(e)*(x-max)) = exp(x-max) for each element
    # =========================================================================
    #
    # TODO: No native 2^x / exp instruction in the ISA.
    # Options:
    #   a) LUT: store a 256-entry lookup table in CR memory, index per element.
    #   b) Polynomial: approximate 2^x via Taylor series using mult.ve + acc.
    #   c) Future instruction: add a dedicated exp instruction to the ISA.
    #
    # For now, data passes through unchanged (identity), so the result is
    # log2(e)*(x-max) rather than exp(x-max). Replace this block when exp is available.

    set     lr2 0;;

pow_loop:

    add     lr5 lr4 lr2;;
    ldr_cyclic_mult_reg lr5 cr0 lr6;;

    # TODO: replace mult.ve.cr passthrough with actual 2^x computation
    mult.ve.cr  lr6 lr6 lr6 cr9;
    acc.first;;

    str_acc_reg lr5 cr0;;

    incr    lr2 1;;
    blt     lr2 lr3 pow_loop;;

    # =========================================================================
    # Step 6: Sum all exp values → compute reciprocal → store in aaq3
    # =========================================================================
    #
    # reset_acc, then for each chunk: load → r_cyclic, passthrough, acc (accumulate).
    # After both chunks, r_acc[i] = sum of all exp values... wait: acc accumulates
    # mult_res into r_acc element-wise. After two chunks both assigned to S0,
    # r_acc[i] = chunk0[i] + chunk1[i], NOT a scalar sum.
    #
    # We want the scalar sum across ALL 256 elements.
    # Use agg sum: for each chunk, agg sum value cr0 aaq3 — but agg overwrites aaq3,
    # not accumulates into it. So we cannot use two sequential agg sum calls.
    #
    # Workaround: use r_acc as accumulator across both chunks.
    #   Chunk 0: acc.first → r_acc = chunk0
    #   Chunk 1: acc       → r_acc[i] = chunk0[i] + chunk1[i]  (element-wise sum)
    # Then agg sum: aaq3 = sum(r_acc[0..127]) = sum(chunk0[i] + chunk1[i]) = total sum.
    # Then agg sum inv: aaq3 = 1 / total_sum.
    #
    # But chunk0[i] and chunk1[i] are different elements of the row (different addresses),
    # and r_acc is 128 words — adding two 128-element chunks element-wise gives
    # r_acc[i] = row[i] + row[128+i].
    # Then sum(r_acc) = sum(row[0..127]) + sum(row[128..255]) = total row sum. ✓

    # Chunk 0 → acc.first
    ldr_cyclic_mult_reg lr4 cr0 lr6;;

    mult.ve.cr  lr6 lr6 lr6 cr9;
    acc.first;;

    # Chunk 1 → acc (accumulate on top)
    add     lr5 lr4 lr8;;
    ldr_cyclic_mult_reg lr5 cr0 lr6;;

    mult.ve.cr  lr6 lr6 lr6 cr9;
    acc;;

    # agg sum inv: aaq3 = 1 / sum(r_acc[0..127]) = 1 / total_row_sum
    agg sum inv cr0 aaq3;;

    # =========================================================================
    # Step 7: Divide each element by sum — multiply by 1/sum (aaq3)
    # =========================================================================
    #
    # For each chunk: load → r_cyclic, mult.ve.aaq with aaq3 (=1/sum),
    # acc.first → r_acc[i] = data[i] * (1/sum) = data[i] / sum.
    # Store back in-place.

    set     lr2 0;;

div_loop:

    add     lr5 lr4 lr2;;
    ldr_cyclic_mult_reg lr5 cr0 lr6;;

    mult.ve.aaq lr6 lr6 lr6 aaq3;
    acc.first;;

    str_acc_reg lr5 cr0;;

    incr    lr2 1;;
    blt     lr2 lr3 div_loop;;

    # =========================================================================
    # Advance to next row
    # =========================================================================

    incr    lr0 1;;
    blt     lr0 lr1 row_loop;;

end:
    bkpt;;
