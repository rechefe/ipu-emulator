# ISA alias catalogue

Operations this ISA does not have, and the instruction sequences kernels use in
their place.

Most entries trace to one fact: **every `ACC.*` and `AGG.*` op consumes
`MULT_RES`**. There is no path into the accumulator that does not cross the
multiplier, and the mult slot has no move, add, subtract, or pass-through form. So
a multiply against a constant `1.0` is the universal stand-in for all of them.

---

## Measuring these idioms

Two levels, and the light one needs no setup.

**Light — always on.** Every run classifies each instruction against the declared
table and reports per-alias hit counts in `RunStats.alias_hits`, alongside the
identity and lane counters. Nothing to enable.

**Full — opt in.** A separate system covering A1–A16, recording observed
instruction patterns and traffic. Effective-MAC accounting is unaffected by it.
Run any registered case with `--profile-aliases`, or `--alias-report FILE` to
enable profiling and export versioned JSON. For a harness or direct emulator
invocation:

```python
from ipu_emu.alias_profile import AliasProfile
from ipu_emu.ipu_state import IpuState

profile = AliasProfile(metadata={"experiment": "hardware-baseline"})
state = IpuState(alias_profile=profile)
# Alternatively: app.run(alias_profile=profile)
# Execute normally, then inspect profile.to_dict() / profile.to_json().
```

Each report contains all registered aliases, subtype/status counts, instruction
sites, bounded examples, participating instruction/cycle counts, elapsed spans,
active lanes, and actual XMEM bytes. Metrics on sequence occurrences are
**inclusive**: a hoisted constant load can participate in many occurrences, so
their costs must not be summed as exclusive totals. `overlap_instruction_groups`
and the unique profiled instruction/cycle totals identify shared evidence.
`unique_read_bytes` is the union of read addresses, irrespective of memory version.

---

## The catalogue

Each entry: the operation that does not exist, and the sequence standing in for it.

### A1 — `MOV.RC`: move a vector into the accumulator

Every other identity-multiply entry below is a special case of this one.

```asm
# Form 1 - ring vector x CR one
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;;  # ring <- src, one bundle ahead
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ;     # x 1.0, pure routing
    ACC.ADD.FIRST ;;                    # ACC.ADD to accumulate instead

# Form 2 - ring x a resident all-ones row in R0
    LDR_MULT_REG R0 LR2 CR2 ;;          # R0 <- 128 x 1.0, hoisted once
    MULT.RC.VV LR0 R0 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;

# Form 3 - single element, broadcast to all 128 lanes
    MULT.EE LR1 CR1 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
```

`MULT.VE` is the Ra-side variant of form 1, for a source already in `R0`/`R1`.
Form 2 additionally burns an `R0`/`R1` slot and an XMEM row on the ones vector.

### A2 — `ADD.VV` / `ADD.VS`: vector add

There is no vector add. Each addend is passed through the multiplier against `1.0`
and summed in the accumulator.

```asm
# Form 1 - vector + vector (residual add). Two passes, one accumulator.
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;;  # ring <- A
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ;     # A x 1.0 
    ACC.ADD.FIRST ;
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;;  # ring <- B, co-issued
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ;     # B x 1.0 
    ACC.ADD ;;

# Form 2 - vector + scalar
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ;     # vector x 1.0
    ACC.ADD.FIRST ;;
    MULT.EE LR1 CR1 0 LR0 CR15 ;        # scalar broadcast x 1.0
    ACC.ADD ;;
```

### A3 — `SUB.VV` / `NEG`: vector subtract

`ACC.SUB` exists, but its operand must still be staged through the multiplier.

```asm
# Subtract: stage the subtrahend, subtract in the acc slot
    MULT.EE LR1 CR1 0 LR0 CR15 ;     # broadcast x 1.0
    ACC.SUB ;;                       # e.g. softmax's x - max

# Negate only (no prior accumulator content)
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ;  # x 1.0
    ACC.SUB.FIRST ;;                 # R_ACC = -MULT_RES
```

Multiplying by a CR holding `-1.0` is an earlier form of the same alias, at the
same cost; only the sign moves from the multiplier to the acc slot.

### A4 — `AGG.SUM.RC` / `AGG.MAX.RC`: reduce a register directly

A reduction cannot read a vector register; it only reduces `MULT_RES`.

```asm
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ;  # x 1.0
    AGG.SUM.FIRST LR3 CR15 ;;        # AGG.SUM to accumulate across cycles
    # AGG.MAX.FIRST for max-reduce; ACC.MAX for an elementwise running max
```

`AGG.*` collapses lanes `0 .. valid_elements-1` to the single slot `LR[LR3] % 128`.

### A5 — `BCAST` / `SPLAT`: replicate one value across lanes

`MULT.EE` *is* the only broadcast path, so a pure splat costs a multiply.

```asm
# Form 1 - one Ra element -> all 128 lanes
    MULT.EE LR1 CR1 0 LR0 CR15 ;     # x 1.0
    ACC.ADD.FIRST ;;

# Form 2 - windowed splat: one partition-sized window -> every partition.
# Repeat per partition p, with rc_idx = (-partition_size * p) mod 512
# selecting the READ window and mask_offset = p selecting the WRITE window.
    SET LR4 CR3 ;;                   # CR3 = (-partition_size * p) mod 512
    MULT.RC.VE LR4 CR1 1 LR0 CR15 ;  # mask_offset = p = 1
    ACC.ADD ;;
    # ... unrolled for p = 0 .. partitions-1

# Form 3 - ACC.RESHAPE, limited to 8 lanes per call and still mult-gated
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ;  # x 1.0
    ACC.RESHAPE LRD0 LRD2 0 ;;       # MULT_RES[src[i]] -> R_ACC[dst[i]]
```

Form 3 reads *8 byte-elements* of an `LRDn` pair as source and destination lane
indices, so replicating one scalar across a 16-lane partition takes two calls and
still needs the identity multiply to populate `MULT_RES`. Form 2 costs one
multiply per partition.

### A6 — `GATHER` / `SCATTER` / `PERMUTE` / `ROTATE`

A masked `×1.0` where `rc_idx` picks the read window and `mask_offset` picks the
write window.

```asm
# Form 1 - gather/scatter
    MULT.RC.VE LR4 CR1 3 LR0 CR15 ;     # x 1.0, write window 3
    ACC.ADD ;;

# Form 2 - cross-lane rotate. The ring does NOT wrap at 128, so the row must be
# materialised twice, back to back, and read at a shifted offset.
    ACTIVATE.QUANTIZE identity CR15 ;
    STR_POST_AAQ_REG LR2 CR2 ;;         # drain R_ACC to scratch
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;;  # reload copy 1 at ring slot 0
    LDR_CYCLIC_MULT_REG LR2 CR0 LR5 ;;  # reload copy 2 at ring slot 128
    MULT.RC.VE LR6 CR1 0 LR0 CR15 ;     # LR6 = rotate amount
    ACC.MAX ;;

# Form 3 - ACC.RESHAPE: an arbitrary 8-lane permutation, MULT_RES -> R_ACC
    MULT.RC.VE LR0 CR1 0 LR0 CR15 ;     # x 1.0
    ACC.RESHAPE LRD0 LRD2 0 ;;
```

### A7 — `REDUCE.SEG`: segmented (partition-wise) reduce

Every `AGG` collapses to *one* scalar, so N partial sums require N passes — the
hand-built "primitive A":

```asm
# Drain the 128 lanes holding N partitions of partition_size
    ACTIVATE.QUANTIZE identity CR15 ;
    STR_POST_AAQ_REG LR2 CR2 ;;
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;;  # reload into the ring

# Then one pass per partition p, at rc_idx = partition_size * p
    SET LR4 CR3 ;;                      # CR3 = partition_size * 0
    MULT.RC.VE LR4 CR1 0 LR0 CR15 ;     # x 1.0
    ACC.ADD.FIRST ;;
    SET LR4 CR4 ;;                      # CR4 = partition_size * 1
    MULT.RC.VE LR4 CR1 0 LR0 CR15 ;     # x 1.0
    ACC.ADD ;;
    # ... repeated to p = N-1
```

`CR15.partition` already encodes the grouping, but it only feeds mask-shift math —
it does not move or reduce data.


### A8 — `MOV.ACC`: get `R_ACC` back to the multiply stage

`R_ACC` has no path back to the multiply-stage inputs. The only route is through
external memory — a register move executed through XMEM.

```asm
# Drain
    ACTIVATE.QUANTIZE identity CR15 ;   # 'identity' = this is a move
    STR_POST_AAQ_REG LR2 CR2 ;;

# Re-enter, either side
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;;  # -> R_CYCLIC
    LDR_MULT_REG R0 LR2 CR2 ;;          # -> R0 / R1
```

No identity multiply, but one `xmem_write` and one `xmem_read` per move, plus the
round-trip latency. It also **quantizes to INT8 on the way out**, so it is lossy as
a register move.

### A9 — `EXP`: only `exp2` exists

```asm
# Rebase by log2(e), then use the native exp2
    LDR_MULT_REG R0 LR2 CR2 ;;      # R0 <- 128 x log2(e), hoisted once
    MULT.RC.VV LR0 R0 0 LR0 CR15 ;  # REAL multiply, genuinely needed
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE exp2 CR15 ;
    STR_POST_AAQ_REG LR2 CR2 ;;
```

A real multiply — but it is work a native `exp` would not need, plus one
resident vector register and one XMEM row for the constant.

Full activation set (`activations.py`): `identity`, `relu`, `relu6`, `sigmoid`,
`tanh`, `gelu`, `softplus`, `elu`, `exp2`, `reciprocal`, `rsqrt`, `silu`,
`window`. No `exp`, `log`, `sqrt`, `abs`, `sign`, `clamp`, `pow`. Because
`reciprocal` and `rsqrt` exist natively, no Newton-Raphson or LUT divide chains are
needed anywhere.

### A10 — fractional scalar in a CR

In INT8 and wide vector modes, `MULT.RC.VE` interprets a CR scalar's low byte as a
signed integer, so fractional constants cannot use that scalar path.

```asm
# Wide FP32: materialise the constant as a full 128-lane row
    LDR_MULT_REG R0 LR2 CR2 ;;  # R0 <- 128 x 0.5
    MULT.RC.VV LR0 R0 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
```

Costs one XMEM row and one `R0`/`R1` slot per distinct fractional constant. In
narrow FP8, store the encoded byte for 0.5 in a writable CR and use
`MULT.RC.VE LR0 CR3 0 LR0 CR15` directly. In INT8, neither a scalar byte nor a
vector byte represents a fraction.

### A11 — `ACC.MUL`: scale the accumulator by `MULT_RES`

`ACC.ADD` always *adds* `MULT_RES` into `R_ACC`; `R_ACC[i] *= MULT_RES[i]` has no
encoding and no in-register alias. The only route is a full A8 round-trip:

```asm
    ACTIVATE.QUANTIZE identity CR15 ;
    STR_POST_AAQ_REG LR2 CR2 ;;         # drain R_ACC
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;;  # reload as ring data
    MULT.RC.VV LR0 R0 0 LR0 CR15 ;      # now multiply against R0
    ACC.ADD.FIRST ;;
```

Writing `ACC.ADD` in place of the round-trip computes `A+B`, not `A*B`.

### A12 — post-increment addressing / same-cycle load→mult forwarding

`MULT.RC.*` reads `R_CYCLIC` from the start-of-cycle snapshot, so it cannot consume
a row loaded in its own bundle (issue #157). The pointer bump therefore needs an
LR-only bundle of its own per inner-loop iteration, with the mult slot idle in it.

```asm
loop_pre:
    ADD LR1 LR1 CR1 ;;               # <-- ENTIRE BUNDLE, no multiply
loop:
    MULT.RC.VE LR0 LR1 0 LR0 CR15 ;  # real MAC
    ACC.ADD ;
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;
    ADD LR2 LR2 CR4 ;
    BLT LR1 LR5 loop_pre ;;
```

A kernel whose `AGG` `dest_slot` reads the snapshot escapes it — no pre-increment
bundle is needed.

### A13 — narrow / `valid_elements`-costed MULT

A kernel that zero-pads a narrow row to 128 lanes really does multiply all 128;
no encoding makes the narrow case cost less.

```asm
# Same bundle bills 128 lanes whether 16 are live or 128
    MULT.RC.VE LR0 LR1 0 LR0 CR15 ;       # CR15.valid_elements = 128 (default)
    ACC.ADD ;;
```

`valid_elements` does not gate multiply results: `_mult_mask_and_shift` gates them
with `R_MASK` and the partition-dependent shift. The lane counter intersects that
mask with lanes `0 .. valid_elements-1`, using the multiply's dstructure CR
(conventionally CR15), **for statistics only**. A zero value falls back to the full
mask as an undeclared width. `AGG` operations separately use `valid_elements` to
bound their reduction. Declaring a narrower width therefore changes the multiply
statistic, not execution or issue cost.

### A14 — multiple accumulators / `MULT.OUTER`

Only one `R_ACC` exists, so only one output channel can be in flight, and a kernel
re-streams its whole input block once per output channel.

```asm
j_loop:
    SET LR2 CR2 ;;                     # rewind data pointer EVERY j
k_loop:
    MULT.RC.VE LR0 LR1 0 LR0 CR15 ;
    ACC.ADD ;
    LDR_CYCLIC_MULT_REG LR2 CR0 LR0 ;  # re-read, N_OUT times over
    ...
```

Each re-stream repeats the reads, multiplying `xmem_reads` by `N_OUT`.

### A15 — `SELECT` / `MASK.VV`, `CSEL`, and scalar `MUL` in the LR slot

Three smaller gaps.

```asm
# Missing SELECT/MASK.VV - predication folded in multiplicatively
    LDR_MULT_REG R0 LR2 CR2 ;;      # R0 <- KEEP mask, 1.0 real / 0.0 pad
    MULT.RC.VV LR0 R0 0 LR0 CR15 ;  # gate by multiplying
    ACC.ADD ;;

# Missing CSEL - a branch diamond stands in for a conditional move
    BLT LR1 LR5 use_full ;;
    SET LR6 CR8 ;;                  # tail bound
    B joined ;;
use_full:
    SET LR6 CR7 ;;                  # full bound
joined:

# Missing scalar MUL in the LR slot - repeated ADD instead of cbound * P
    ADD LR6 LR6 LR7 ;;
    ADD LR6 LR6 LR7 ;;
```

The branch diamond costs several bundles per taken path, none of them multiplying.

### A16 — `STR_ACC_REG` is simulation-only

Flagged `"hardware": False` in `instruction_spec.py` — it has no hardware encoding.

```asm
# Simulation only
    STR_ACC_REG LR2 CR2 ;;

# Real-hardware equivalent
    ACTIVATE.QUANTIZE identity CR15 ;
    STR_POST_AAQ_REG LR2 CR2 ;;
```

The real form quantizes to INT8; `STR_ACC_REG` writes raw accumulator words. Cycle
counts from a kernel relying on it are not achievable on silicon. One live use
remains, at `fully_connected/fully_connected.asm:48`.

---

---

## Adding an alias

An alias reaches the counters two ways: the declarative table, and a profiler
detector.

### The table

The idioms above are declared in `ipu_common/isa_alias_spec.py`, validated at
import; each run reports per-alias hit counts. Entries are most-specific-first with
first-match-wins, so an idiom matching several entries is counted once, under the
most specific. A16 independently counts the simulation-only accumulator store.

Because a bundle is dispatched as a unit, a constraint can name a **co-issued
slot** — the only thing separating A1–A4 and A6, since all of them use an identical
`MULT.RC.VE` with a 1.0 scalar or a declared ONES vector. A1–A4 differ in the acc op
beside it. A6 takes precedence for nonzero mask slots, mask-zero windows narrower
than the declared destination span, and declared-vector routing through
`ACC.STRIDE`. A mask matching a narrower `valid_elements` declaration is ordinary
padding and does not by itself trigger A6. An unmasked move and routing with
identical operands share a signature, and match as A1–A4.

Scalar constraints must name a supported scalar multiplicand. Each match may carry
at most one immediate constraint and one co-slot constraint; `hardware_only` must
stand alone. Duplicate match signatures are rejected.

### A detector

The profiling engine contains no numbered alias cases. Implement a detector whose
`observe(event)` yields `Match` objects, then register its factory and
`AliasDefinition` in `default_registry()` in `ipu_emu.alias_detectors`. A detector
can subscribe to named execution slots through its `slots` tuple; omitting it
subscribes to all events. A `cycle` event contains the completed bundle and its
actual next PC, and `Event.facts` provides decoded operands plus the shared
execution provenance collected by `alias_observer`.

For an experiment, copy `default_registry()`, register the detector on that copy,
and pass it to `AliasProfile(registry)`. Factories create independent state per
profile; duplicate IDs and matches emitted under another detector's ID are
rejected. Add positive/negative detector tests and a real-kernel expectation
alongside the new entry. No emulator dispatch or report formatting changes are
needed unless the detector requires a genuinely new execution fact.
