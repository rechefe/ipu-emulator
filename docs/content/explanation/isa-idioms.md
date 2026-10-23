# ISA idioms

The IPU instruction set has no vector move, add, subtract or broadcast. Kernels
build each missing operation from a short sequence of instructions it does
have. This page lists each sequence (an *idiom*, numbered A1 to A15), why the
ISA forces it, and what it costs.

Most idioms trace to one fact: **every `ACC.*` and `AGG.*` instruction consumes
`MULT_RES`**. Every path into the accumulator crosses the multiplier, and the
MULT slot has no move, add, subtract or pass-through form. A multiply by a
constant `1.0` therefore stands in for all of them. It occupies the multiplier
for a cycle without doing arithmetic.

In the examples, `CR1` is the MULT constant `1.0` (in both modes), `CR15.L`
holds the DS (data structure) half and `CR14.L` the QF (quantization format)
half the NX8 elements are decoded with ([Configuration
registers](../reference/isa/configuration-registers.md)). `;` separates the
slots of one instruction word and `;;` ends the word.

Every run counts how often each idiom occurred. To measure them in a kernel,
see [The kernel is too slow](../how-to/debug-a-kernel.md#the-kernel-is-too-slow)
in the debug how-to; the options are in the
[command-line reference](../reference/cli.md).

## A1 — `MOV.RC`: move a vector into the accumulator

A multiply by `1.0` followed by `ACC.ADD.FIRST` copies a vector into `R_ACC`.
Every other identity-multiply idiom on this page is a special case of this one.

```asm
# Form 1 - R_CYCLIC vector x CR one
    LD.MULT.CYCLIC LR2 CR0 LR0 ;;   # R_CYCLIC <- source, one word ahead
    MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; # x 1.0, routing only
    ACC.ADD.FIRST ;;                # ACC.ADD accumulates instead

# Form 2 - R_CYCLIC x an all-ones row resident in R0
    LD.MULT R0 LR2 CR2 ;;           # R0 <- 128 x 1.0, loaded once
    MULT.RC.VV LR0 R0 0 LR0 CR15.L CR14.L ;
    ACC.ADD.FIRST ;;

# Form 3 - one element, broadcast to all 128 lanes
    MULT.EE LR1 CR1 0 LR0 CR15.L CR14.L ;
    ACC.ADD.FIRST ;;
```

`MULT.VE` is the form for a source already in `R0` or `R1`. Form 2 also takes
an `R0`/`R1` register and an XMEM row for the ones vector.

## A2 — `ADD.VV` / `ADD.VS`: vector add

Each addend passes through the multiplier against `1.0`, and the accumulator
sums them.

```asm
# Form 1 - vector + vector (residual add): two passes, one accumulator
    LD.MULT.CYCLIC LR2 CR0 LR0 ;;   # R_CYCLIC <- A
    MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; # A x 1.0
    ACC.ADD.FIRST ;
    LD.MULT.CYCLIC LR2 CR0 LR0 ;;   # R_CYCLIC <- B, in the same word
    MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; # B x 1.0
    ACC.ADD ;;

# Form 2 - vector + scalar
    MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; # vector x 1.0
    ACC.ADD.FIRST ;;
    MULT.EE LR1 CR1 0 LR0 CR15.L CR14.L ;    # scalar broadcast x 1.0
    ACC.ADD ;;
```

## A3 — `SUB.VV` / `NEG`: vector subtract and negate

`ACC.SUB` exists, but its operand still comes from `MULT_RES`, so the
subtrahend passes through the multiplier first.

```asm
# Subtract: stage the subtrahend, subtract in the ACC slot
    MULT.EE LR1 CR1 0 LR0 CR15.L CR14.L ;    # broadcast x 1.0
    ACC.SUB ;;                      # for example softmax's x - max

# Negate, into an empty accumulator
    MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; # x 1.0
    ACC.SUB.FIRST ;;                # R_ACC = -MULT_RES
```

A multiply by a CR that holds `-1.0` costs the same: the sign moves from the
ACC slot to the multiplier.

## A4 — `AGG.SUM.RC` / `AGG.MAX.RC`: reduce a register

A reduction reads only `MULT_RES`, never a vector register.

```asm
    MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; # x 1.0
    AGG.SUM.FIRST LR3 CR15.L ;;     # AGG.SUM accumulates across words
    # AGG.MAX.FIRST reduces to the maximum; ACC.MAX keeps an elementwise running maximum
```

`AGG.*` reduces lanes `0 .. valid_elements-1` of its DS half to the one slot
`LR[LR3] % 128`.

## A5 — `BCAST` / `SPLAT`: replicate one value across lanes

`MULT.EE` is the only broadcast path, so a broadcast costs a multiply.

```asm
# Form 1 - one Ra element -> all 128 lanes
    MULT.EE LR1 CR1 0 LR0 CR15.L CR14.L ;    # x 1.0
    ACC.ADD.FIRST ;;

# Form 2 - windowed broadcast: one partition-sized window -> every partition.
# One pass per partition p: rc_idx = (-partition_size * p) mod 512 selects the
# read window, mask_offset = p the write window. This is the pass for p = 1.
    SET LR4 CR3 ;;                  # CR3 = (-partition_size * p) mod 512
    MULT.RC.VE LR4 CR1 1 LR0 CR15.L CR14.L ; # mask_offset = p = 1
    ACC.ADD ;;

# Form 3 - ACC.RESHAPE: 8 lanes per instruction, still behind the multiplier
    MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; # x 1.0
    ACC.RESHAPE LRD0 LRD2 0 ;;      # MULT_RES[src[i]] -> R_ACC[dst[i]]
```

Form 2 costs one multiply per partition. Form 3 reads 8 elements of an `LRDn`
pair as source and destination lane indices, so a 16-lane partition takes two
`ACC.RESHAPE` instructions, each after an identity multiply that fills
`MULT_RES`.

## A6 — `GATHER` / `SCATTER` / `PERMUTE` / `ROTATE`

A masked multiply by `1.0` moves lanes: `rc_idx` selects the read window and
`mask_offset` the write window.

```asm
# Form 1 - gather or scatter
    MULT.RC.VE LR4 CR1 3 LR0 CR15.L CR14.L ; # x 1.0, write window 3
    ACC.ADD ;;

# Form 2 - rotate across lanes. R_CYCLIC does not wrap at 128, so the row is
# loaded twice, back to back, and read at a shifted offset.
    ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR2 ;; # drain R_ACC to scratch
    LD.MULT.CYCLIC LR2 CR0 LR0 ;;   # copy 1 at R_CYCLIC offset 0
    LD.MULT.CYCLIC LR2 CR0 LR5 ;;   # copy 2 at R_CYCLIC offset 128
    MULT.RC.VE LR6 CR1 0 LR0 CR15.L CR14.L ; # LR6 = rotate amount
    ACC.MAX ;;

# Form 3 - ACC.RESHAPE: any 8-lane permutation, MULT_RES -> R_ACC
    MULT.RC.VE LR0 CR1 0 LR0 CR15.L CR14.L ; # x 1.0
    ACC.RESHAPE LRD0 LRD2 0 ;;
```

## A7 — `REDUCE.SEG`: reduce each partition

Every `AGG.*` instruction produces one scalar, so N partition sums take N
passes.

```asm
# Drain the 128 lanes that hold N partitions of partition_size
    ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR2 ;;
    LD.MULT.CYCLIC LR2 CR0 LR0 ;;   # reload into R_CYCLIC

# One pass per partition p, at rc_idx = partition_size * p
    SET LR4 CR3 ;;                  # CR3 = partition_size * 0
    MULT.RC.VE LR4 CR1 0 LR0 CR15.L CR14.L ; # x 1.0
    ACC.ADD.FIRST ;;
    SET LR4 CR4 ;;                  # CR4 = partition_size * 1
    MULT.RC.VE LR4 CR1 0 LR0 CR15.L CR14.L ; # x 1.0
    ACC.ADD ;;
    # and so on, to p = N-1
```

The DS half's `num_partitions` encodes the grouping, but it feeds only the
mask-shift arithmetic; it moves and reduces no data.

## A8 — `MOV.ACC`: return `R_ACC` to the multiply stage

`R_ACC` has no path back to the multiply-stage inputs. The value goes out to
XMEM and back in.

```asm
# Drain: identity is a move, no activation
    ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR2 ;;

# Reload, into either input
    LD.MULT.CYCLIC LR2 CR0 LR0 ;;   # -> R_CYCLIC
    LD.MULT R0 LR2 CR2 ;;           # -> R0 / R1
```

The move needs no identity multiply. It costs one `xmem_write`, one
`xmem_read` and the round-trip latency. In the NX8 mode it also **quantizes
the row to the QF half's NX8 format on the way out** (with a scale of its own
in the row's metadata), so it is a lossy move. In the FP32 mode
`ACTIVATE.EXACT identity CR15.L CR2 LR2` moves the FP32 row unchanged.

## A9 — `EXP`: only `exp2` exists

The activation unit has `exp2`. A kernel computes `exp(x)` as
`exp2(x * log2(e))`.

```asm
    LD.MULT R0 LR2 CR2 ;;           # R0 <- 128 x log2(e), loaded once
    MULT.RC.VV LR0 R0 0 LR0 CR15.L CR14.L ;  # a real multiply
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE exp2 CR15.L CR14.L CR2 LR2 ;;
```

The multiply is real arithmetic, but a native `exp` would not need it. The
constant also takes a vector register and an XMEM row.

The hardware activations (`ACTIVATE.QUANTIZE`, the architecture spec's list)
are `identity`, `relu`, `relu6`, `reciprocal`, `rsqrt`, `exp2` and `generic`,
which evaluates the table `LD.LUT` loaded. The emulator-only
`ACTIVATE.EXACT` (FP32 mode only) has the 13 exact functions of
`activations.py`: `identity`, `relu`, `relu6`, `sigmoid`, `tanh`, `gelu`,
`softplus`, `elu`, `exp2`, `reciprocal`, `rsqrt`, `silu` and `window`.
Neither has `exp`, `log`, `sqrt`, `abs`, `sign`, `clamp` or `pow`.
`reciprocal` and `rsqrt` are native, so no kernel needs a Newton-Raphson
divide.

## A10 — a fractional scalar in a CR

A CR scalar of `MULT.VE`, `MULT.EE` or `MULT.RC.VE` is a MULT constant (see
[Configuration registers](../reference/isa/configuration-registers.md#mult-constant)).
In the NX8 mode it holds an NX8 element with a scale and an NX type of its own
(`value[7:0]`, `scale[15:8]`, `fe[19:16]`, `sign[23:20]`), so it holds a
fractional constant directly: 0.5 is the element 1.0 (`value` 0) under scale
2^-1 (`scale` 126). In the FP32 mode it is the float32 bit pattern (0.5 is
`0x3F000000`). `CR0` and `CR1` read as 0.0 and 1.0 in both modes.

```asm
    MULT.RC.VE LR0 CR3 0 LR0 CR15.L CR14.L ; # CR3 holds the MULT constant 0.5
    ACC.ADD.FIRST ;;
```

## A11 — `ACC.MUL`: multiply the accumulator by `MULT_RES`

`ACC.ADD` adds `MULT_RES` to `R_ACC`. No instruction computes
`R_ACC[i] *= MULT_RES[i]`, so the accumulator goes through XMEM (A8) and comes
back as a multiplicand.

```asm
    ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR2 ;; # drain R_ACC
    LD.MULT.CYCLIC LR2 CR0 LR0 ;;   # reload into R_CYCLIC
    MULT.RC.VV LR0 R0 0 LR0 CR15.L CR14.L ;  # multiply by R0
    ACC.ADD.FIRST ;;
```

`ACC.ADD` in place of the round trip computes `A + B`, not `A * B`.

## A12 — post-increment addressing: a load feeds the next word's multiply

`MULT.RC.*` reads `R_CYCLIC` at cycle start
([timing model](timing-model.md#rule-2)), so it cannot use a row loaded in the
same instruction word (issue #157). The pointer increment takes an instruction
word of its own in each loop iteration, with the MULT slot idle.

```asm
loop_pre:
    ADD LR1 LR1 CR1 ;;              # a whole word, no multiply
loop:
    MULT.RC.VE LR0 LR1 0 LR0 CR15.L CR14.L ; # real multiply-accumulate
    ACC.ADD ;
    LD.MULT.CYCLIC LR2 CR0 LR0 ;
    ADD LR2 LR2 CR4 ;
    BLT.S LR1 LR5 loop_pre ;;
```

A kernel whose loop counter is an `AGG` `dest_slot` needs no extra word:
`dest_slot` reads the value after the LR writes of the same word, so the
counter starts one lower and its `INC` shares the word with the `AGG`
(the `attn_v_*` kernels).

## A13 — a narrow multiply costs a full one

A kernel that pads a narrow row with zeros to 128 lanes multiplies all 128.
No encoding makes a narrow multiply cheaper.

```asm
# The same word counts 128 lanes whether 16 or 128 hold data
    MULT.RC.VE LR0 LR1 0 LR0 CR15.L CR14.L ; # CR15.L: valid_elements = 128 (the default)
    ACC.ADD ;;
```

`valid_elements` does not gate multiply results: `_mult_mask_and_shift` gates
them with `R_MASK` and the partition-dependent shift. The lane statistic
intersects that mask with the active elements of the multiply's DS half
(`valid_elements` per active partition; `CR15.L` by convention). `AGG`
instructions use `valid_elements` to bound their reduction. A narrower
declared width changes the lane statistic only, not the results or the cycle
count.

## A14 — one accumulator: `MULT.OUTER` and many outputs at once

There is one `R_ACC`, so one output channel is in progress at a time. A kernel
reads its whole input block again for each output channel.

```asm
j_loop:
    SET LR2 CR2 ;;                  # rewind the data pointer for each output j
    SET LR1 CR5 ;;                  # restart the inner counter
k_loop:
    MULT.RC.VE LR0 LR1 0 LR0 CR15.L CR14.L ;
    ACC.ADD ;
    LD.MULT.CYCLIC LR2 CR0 LR0 ;    # the same rows, read once per output
    ADD LR2 LR2 CR4 ;
    INC LR1 1 ;
    BLT.S LR1 LR5 k_loop ;;
    ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR3 ; # output j
    INC LR6 1 ;
    BLT.S LR6 LR7 j_loop ;;
```

`xmem_reads` grows by a factor of `N_OUT`, the number of output channels.

## A15 — `SELECT` / `MASK.VV`, `CSEL` and a scalar `MUL` in the LR slot

Three smaller gaps.

```asm
# SELECT / MASK.VV: multiply by a 1.0 / 0.0 mask
    LD.MULT R0 LR2 CR2 ;;           # R0 <- keep mask, 1.0 data / 0.0 padding
    MULT.RC.VV LR0 R0 0 LR0 CR15.L CR14.L ;  # gate by multiplying
    ACC.ADD ;;

# CSEL: a branch diamond stands in for a conditional move
    BLT.S LR1 LR5 use_full ;;
    SET LR6 CR8 ;;                  # tail bound
    B joined ;;
use_full:
    SET LR6 CR7 ;;                  # full bound
joined:

# Scalar MUL: repeated ADD in place of cbound * P
    ADD LR6 LR6 LR7 ;;
    ADD LR6 LR6 LR7 ;;
```

The branch diamond costs several instruction words on each path, none of them
multiplying.

## The retired A16: the emulator-only accumulator store

A16 counted `ST.ACC`, a raw `R_ACC` dump in an emulator-only slot with no
hardware encoding. [ADR 0007](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0007-operational-aaq-and-nx8.md)
removed it: the AaQ stage stores its own row, so a raw dump is
`ACTIVATE.QUANTIZE identity` (the hardware instruction, quantized in the NX8
mode) or, in the FP32 mode, `ACTIVATE.EXACT identity` (emulator-only, an
exact FP32 row). The number A16 is not reused.

```asm
    ACTIVATE.QUANTIZE identity CR15.L CR14.L CR2 LR2 ;; # NX8: one row and its metadata
    ACTIVATE.EXACT identity CR15.L CR2 LR2 ;;           # FP32 mode only
```

## How runs count the idioms

Two mechanisms count idioms, at two levels of detail.

**Hit counts, on every run.** `ipu_emu/isa_alias_spec.py` declares the idioms
as a table, checked at import. Each run classifies every instruction word
against it and reports a count per idiom in `RunStats.alias_hits`, beside the
identity-multiply and lane counters. The table lists the most specific entry
first, and the first match wins, so a word that matches several entries counts
once, under the most specific.

A constraint can name an instruction in another slot of the same word, since a
word executes as a unit. That is the only difference between A1–A4 and A6:
each uses the same `MULT.RC.VE` with a `1.0` scalar or a declared ones vector,
and A1–A4 differ in the ACC instruction beside it. A6 wins for a nonzero
mask offset, for a mask-zero window narrower than the declared destination
span, and for a declared vector routed through `ACC.STRIDE`. A mask that
matches a narrower `valid_elements` declaration is ordinary padding and does
not by itself make a word A6. An unmasked move and a routing with identical
operands have the same signature, and count as A1–A4.

A scalar constraint names a supported scalar multiplicand. A match carries at
most one immediate constraint and one constraint on another slot;
`hardware_only` stands alone. The table rejects two entries with the same
match signature.

**The full profile, on request.** `ipu_emu.alias_profile.AliasProfile` runs a
detector per idiom (A1–A15) and records instruction sites, examples,
participating instruction words and cycles, active lanes and XMEM bytes. It
does not change the effective-MAC accounting. The debug how-to shows how to
turn it on.

## Add an idiom

1. To count it on every run, add an entry to the table in
   `ipu_emu/isa_alias_spec.py`, most specific entries first.
2. To profile it, write a detector: a class whose `observe(event)` yields
   `Match` objects. Register its factory and its `AliasDefinition` in
   `default_registry()` in `ipu_emu.alias_detectors`.
3. Add tests: a positive and a negative detector test, and the count it gives
   on a real kernel.

A detector subscribes to slots through its `slots` tuple; without one it
receives every event. A `cycle` event holds the completed instruction word and
its next pc. `Event.facts` holds the decoded operands and the execution
details `alias_observer` collects. The profiling engine has no per-idiom code,
so a new detector needs no change to emulator dispatch or report formatting,
unless it needs an execution fact nothing collects yet.

For an experiment, copy `default_registry()`, register the detector on the
copy, and pass it to `AliasProfile(registry)`. Each profile gets its own
detector state from the factories. A registry rejects a duplicate ID and a
match a detector emits under another detector's ID.
