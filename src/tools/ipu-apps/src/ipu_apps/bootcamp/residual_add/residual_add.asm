# =============================================================================
# BOOTCAMP DRILL 1/3 -- residual_add.asm
# Difficulty: * (LR + XMEM/LOAD only -- no MULT/ACC compute "for real", no AAQ)
# =============================================================================
#
# WHAT THIS PROGRAM DOES
# -----------------------
# Adds two INT8 tensors element-wise: out[v] = A[v] + B[v], for `num_vectors`
# vectors of 128 elements each (one IPU "row"). This is the residual/skip
# connection you see in almost every modern neural net (ResNet, Transformers).
#
# WHY IT'S HERE
# -------------
# There is no dedicated "ADD two vectors" instruction on this ISA -- the only
# vector arithmetic paths run *through* the multiply-accumulate datapath
# (MULT -> ACC -> R_ACC). So even the simplest possible "add two tensors" op
# has to be expressed as a multiply-accumulate:
#
#     r_acc  <-  1 * A[v]     (ACC.ADD.FIRST: overwrite, ignore old r_acc)
#     r_acc +=  1 * B[v]      (ACC.ADD: accumulate)
#     r_acc  ==  A[v] + B[v]
#
# This drill exists to teach the LOAD and XMEM plumbing (LDR_CYCLIC_MULT_REG,
# STR_ACC_REG, row-addressed CR bases, the LR loop-counter idiom) using the
# most boring possible "compute" (multiply-by-one). Drill 2 (conv) swaps the
# boring multiply-by-one for a real multiply-accumulate sliding window.
#
# THE "MULTIPLY BY ONE" TRICK
# ----------------------------
# MULT.RC.VE multiplies every element of R_CYCLIC by a *scalar*. If that
# scalar is 1, the "multiply" is just a copy. CR1 is a hardware-reserved
# register that always reads as the integer 1 (CR0 always reads as 0) --
# so `MULT.RC.VE ..., CR1, ...` is our "identity multiply" / vector copy.
#
# PIPELINE TIMING -- WHY THE LOAD IS ONE CYCLE AHEAD OF THE MULT THAT USES IT
# -----------------------------------------------------------------------------
# Every VLIW word is one hardware cycle. Within a cycle, slots read a
# *snapshot* of the register file taken at the START of that cycle (a
# same-cycle write from one slot is NOT visible to another slot reading the
# same register this cycle -- "read-before-write" semantics). MULT reads
# R_CYCLIC from that same start-of-cycle snapshot, so a `LDR_CYCLIC_MULT_REG`
# in the SAME word as a `MULT.RC.VE` that reads R_CYCLIC would still see the
# *previous* contents of R_CYCLIC, not what was just loaded.
#
# The fix used throughout this codebase: issue the load ONE CYCLE BEFORE the
# multiply that consumes it ("pipelining"). Concretely:
#   cycle N:   load A[v]      into R_CYCLIC          (result visible cycle N+1)
#   cycle N+1: MULT.RC.VE reads R_CYCLIC (= A[v] from cycle N); load B[v]
#              into R_CYCLIC in the SAME word (visible cycle N+2, not now)
#   cycle N+2: MULT.RC.VE reads R_CYCLIC (= B[v] from cycle N+1)
#
# This is why the loop body below always loads the NEXT chunk while
# multiplying the chunk that was loaded last cycle.
#
# XMEM ADDRESSING -- ROWS, NOT BYTES
# ------------------------------------
# `offset`/`base` operands on LDR_CYCLIC_MULT_REG and STR_ACC_REG are ROW
# numbers: one row = 128 elements = 128 bytes in this (narrow, INT8) mode.
# One "vector" of 128 INT8 elements is exactly one row, so walking to the
# next vector is `+= 1` row, not `+= 128` bytes. The harness (__init__.py)
# converts its byte-oriented base addresses into row numbers before writing
# them into the CR registers this program reads.
#
# `LDR_CYCLIC_MULT_REG`'s `index` operand is different: it's an ELEMENT
# index into R_CYCLIC's 512-element ring, and must be one of the four slot
# boundaries 0/128/256/384. We only ever use slot 0 here (index = 0, held in
# LR0) since one 128-element vector fills exactly one slot.
#
# A NOTE ON STR_ACC_REG'S OUTPUT STRIDE
# ---------------------------------------
# STR_ACC_REG always writes the FULL 512 bytes of r_acc (128 lanes x INT32),
# regardless of mode -- unlike LDR_MULT_REG/LDR_CYCLIC_MULT_REG, its transfer
# size does not scale with the input element width. That 512-byte write spans
# 4 rows in this narrow (128-byte-row) mode. So while the INT8 *inputs* step
# by 1 row per vector, the INT32 *output* must step by 4 rows per vector, or
# consecutive stores would overlap. Hence two separate stride CRs below.
#
# CR REGISTERS (set by the Python harness before this program runs):
#   cr0  = 0                    (hardware-reserved: always reads as 0)
#   cr1  = 1                    (hardware-reserved: always reads as 1 -- our "identity" scalar)
#   cr2  = input A base row
#   cr3  = input B base row
#   cr4  = output base row
#   cr5  = input row step (always 1 -- one INT8 vector is one row)
#   cr6  = num_vectors (loop bound)
#   cr7  = output row step (always 4 -- one INT32 r_acc store spans 4 rows)
#
# LR REGISTERS (used by this program):
#   lr0  = 0            (R_CYCLIC slot index -- always slot 0 in this drill)
#   lr1  = input A row cursor AND vector counter (0, 1, 2, ... num_vectors-1)
#          -- steps by cr5 (=1), one row per vector, so it doubles as the
#          loop-bound comparator (the OUTPUT cursor lr2 can't be used for
#          that: it steps by 4, see the STR_ACC_REG stride note above).
#   lr2  = output row cursor (0, 4, 8, ... ) -- steps by cr7, not cr5
#   lr3  = num_vectors, copied into an LR (BLT's bound must itself be an LR/CR;
#          CR6 already qualifies, so lr3 is technically redundant here, but
#          spelling it out keeps the loop-bound source obviously named)

# =============================================================================
# Initialization
# =============================================================================
    SET     lr0 cr0;                 # lr0 = 0 -- the only R_CYCLIC slot we use
    SET     lr1 cr0;;                # lr1 = 0 -- input cursor starts at vector 0

    SET     lr2 cr0;;                # lr2 = 0 -- output cursor starts at vector 0
    SET     lr3 cr6;;                # lr3 = num_vectors (loop bound, as an LR)

    # Prime the pipeline: load A[0] into R_CYCLIC now so it's ready for the
    # first MULT.RC.VE (which runs next cycle, per the timing note above).
    LDR_CYCLIC_MULT_REG lr1 cr2 lr0;;

# =============================================================================
# Main loop: one iteration per 128-element vector
# =============================================================================
loop:
    # A[v] is already sitting in R_CYCLIC (loaded last cycle, or primed
    # above). Copy it into r_acc with ACC.ADD.FIRST (overwrite -- we do NOT
    # want whatever r_acc held from the previous loop iteration). In the same
    # word, kick off the load of B[v] -- it lands in R_CYCLIC next cycle.
    LDR_CYCLIC_MULT_REG lr1 cr3 lr0;
    MULT.RC.VE          lr0 cr1 0 lr0 cr15;
    ACC.ADD.FIRST;;

    # B[v] is now in R_CYCLIC. Multiply-by-1 again and ACC.ADD (accumulate,
    # not overwrite) so r_acc = A[v] + B[v]. While that's happening, also
    # advance lr1 to the NEXT vector and preload A[v+1] so it's ready when
    # the loop comes back around -- keeping the load-one-cycle-ahead pipeline
    # full across iterations.
    ADD                  lr1 lr1 cr5;;

    LDR_CYCLIC_MULT_REG  lr1 cr2 lr0;
    MULT.RC.VE           lr0 cr1 0 lr0 cr15;
    ACC.ADD;;

    # Store the INT32 sum (r_acc is always 128 x INT32 = 512 bytes, regardless
    # of the INT8 input width) to the output row for this vector. The output
    # pointer bump must NOT share this word with the store: LR sub-slots run
    # BEFORE the acc_store slot within a cycle, so bumping lr2 here would
    # store to the *new* (wrong) address instead of the current one. Bump it
    # in the next word instead.
    STR_ACC_REG          lr2 cr4;;

    # Bump the output cursor for next time, then test the LOOP BOUND against
    # lr1 (the input/vector counter -- already advanced to v+1 earlier this
    # iteration), not lr2 (the output cursor steps by 4, not 1, so it can't
    # be compared against a plain vector count). These two must be in
    # separate words from anything that just wrote their operands (";;", not
    # just separate source lines): COND reads its operands from the
    # cycle-start SNAPSHOT, so a same-word write would still be compared
    # against the OLD value, letting the loop run one extra/fewer iteration.
    ADD                  lr2 lr2 cr7;;
    BLT                  lr1 lr3 loop;;

end:
    BKPT;;
