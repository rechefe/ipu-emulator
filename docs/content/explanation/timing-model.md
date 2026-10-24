# Timing model

The IPU retires one instruction word per cycle, and every slot of the word
executes in that cycle. Within a word, a register read sees one of two values:

- the **cycle start** value: what the register held when the word began;
- the **forwarded** value: what an earlier slot of the same word wrote to it.

The **timing contract** fixes which of the two each read sees. It has four
rules ([ADR 0002](https://github.com/rechefe/ipu-emulator/blob/master/docs/adr/0002-timing-contract.md)):

1. The Control stage (`LR`, `COND`) reads LRs at cycle start. Every other
   stage reads LRs after the LR writes of the same word.
2. `MULT` reads `R0`, `R1`, `R_CYCLIC` (and their row metadata) and
   `R_MASK` at cycle start.
3. `MULT_RES` is forwarded to `ACC` (including `AGG` and `ACC.RESHAPE`).
   `R_ACC` is forwarded to `AAQ` (and to the emulator-only `AAQ_EXACT`).
4. The three LR sub-slots read cycle-start values.

A read that no rule names follows the [slot order](#slot-order).

`ipu_isa.timing` declares the contract as data, and every instruction in
`instruction_spec.py` declares the registers it reads and writes. `load_isa`
rejects a declaration that breaks the contract, and the emulator's timing
tests check that each handler reads what it declares.

## Every read, by slot

This table is generated from the declarations in `ipu_isa`. *Selected by*
names the operands that select the register; *implicit* is a read that no
operand names. `R` is the `R0`/`R1` pair.

--8<-- "generated/timing-contract-table.md"

The [instruction reference](../reference/isa/instructions.md) lists the same read points for
each instruction, under **Register reads**.

## Examples

Each example below assembles and runs as a test
(`docs/examples/test_timing_model.py`). The test sets up the state from the
`# given:` comments, runs the program to `HALT`, and checks every
`# expect:` comment. The examples run in the emulator-only FP32 mode, where an
element is its value and a row has no metadata; the timing is the same in
the NX8 mode. `XMEM[n]` is XMEM row `n` (128 elements); a `given` sets every
element of the row. `CR0` is the MULT constant 0 and `CR1` is 1 in every run.

### Rule 1: Control reads LRs at cycle start, other stages after the LR writes { #rule-1 }

In the first word, `INC` writes `LR0 = 1`. `BEQ` is in the Control stage, so it
compares the cycle-start `LR0` (0) with `CR0` (0) and branches to `taken`.
`LD.MULT` is in the `LOAD` stage, so its offset is the new `LR0` (1): it loads
row `CR2 + 1`, which holds 2s.

```asm
# given: CR2 = 16
# given: XMEM[16] = 1
# given: XMEM[17] = 2
    INC LR0 1 ; LD.MULT R0 LR0 CR2 ; BEQ LR0 CR0 taken ;;
    HALT ;;
taken:
    INC LR7 1 ;;
    HALT ;;
# expect: LR0 == 1
# expect: LR7 == 1
# expect: R0[0] == 2
```

The same rule gives `MULT`, `ACC`, `AAQ` and `AAQ_EXACT` their LR operands
after the LR writes: an `AGG` `dest_slot` or an `ACTIVATE.QUANTIZE`
`table_offset` sees an LR that the same word increments.

### Rule 2: MULT reads its data registers at cycle start { #rule-2 }

The second word loads 5s into `R0` and multiplies `R0` by `CR1` (1). `MULT`
reads `R0` at cycle start, so it multiplies the 2s that the first word loaded.
The 5s are in `R0` after the word, for the next word's `MULT`. `R_CYCLIC`,
`R_MASK` and the row metadata `LD.MULT` copies behave the same way: `LD.MULT.CYCLIC` or `LD.MASK` in the same word as
a `MULT` reaches only the next word's `MULT`.

```asm
# given: CR2 = 16
# given: CR3 = 17
# given: XMEM[16] = 2
# given: XMEM[17] = 5
    LD.MULT R0 LR0 CR2 ;;
    LD.MULT R0 LR0 CR3 ; MULT.VE LR0 CR1 0 LR0 CR15.L CR14.L ;;
    HALT ;;
# expect: MULT_RES[0] == 2
# expect: R0[0] == 5
```

### Rule 3: MULT_RES reaches ACC, and R_ACC reaches AAQ, in the same cycle { #rule-3 }

The second word multiplies, accumulates, activates and stores in one word.
`ACC.ADD.FIRST` reads this word's `MULT_RES` (3), and `ACTIVATE.QUANTIZE`
reads this word's `R_ACC` and writes 3 to XMEM row `CR3` (in the NX8 mode it
quantizes the row first and writes its metadata too).

```asm
# given: CR2 = 16
# given: CR3 = 18
# given: XMEM[16] = 3
    LD.MULT R0 LR0 CR2 ;;
    MULT.VE LR0 CR1 0 LR0 CR15.L CR14.L ; ACC.ADD.FIRST ; ACTIVATE.QUANTIZE identity CR15.L CR14.L CR3 LR0 ;;
    HALT ;;
# expect: MULT_RES[0] == 3
# expect: R_ACC[0] == 3
# expect: XMEM[18][0] == 3
```

### Rule 4: LR sub-slots read cycle-start values { #rule-4 }

The three LR sub-slots all read before any of them writes. In the second word,
`ADD` reads the cycle-start `LR0` (5), not the 6 that `INC` writes in the same
word.

```asm
    INC LR0 5 ;;
    INC LR0 1 ; ADD LR1 LR0 CR0 ;;
    HALT ;;
# expect: LR0 == 6
# expect: LR1 == 5
```

Two LR sub-slots of one word must not write the same LR; the emulator faults
on that word (F-3).

### Slot order { #slot-order }

The slots of a word execute in this order: `BREAK`, `LR`, `LOAD`, `MULT`,
`ACC`, `AAQ`, `AAQ_EXACT`, `COND`. A read that no rule names is forwarded
when every slot that writes the register runs earlier in the word, and is at
cycle start otherwise:

- `ACTIVATE.EXACT` reads `R_ACC` forwarded: only `ACC` writes it.
- `ACTIVATE.QUANTIZE` reads the AaQ LUT at cycle start: `LD.LUT`, in its own
  slot, writes it.
- `ACC.ADD`, `ACC.SUB`, `ACC.MAX` and the non-`FIRST` `AGG` forms read `R_ACC`
  at cycle start: their own slot writes it.
- CR reads are at cycle start: no slot writes a CR during a run.

## When a kernel needs an extra word

A value that a read sees only at cycle start reaches it one word later. The
kernel then spends a word, or arranges the program so the old value is the one
it needs:

- **Load, then multiply** (rule 2). A `MULT` cannot use data that the same
  word's `LOAD` fetches. Load one word ahead, or software-pipeline the loop so
  each word multiplies the previous word's load.
- **Branch on a counter** (rule 1). A branch compares the counter's value from
  before the word's `INC`. The loop below runs four times, not three, and
  ends with `LR0 = 4`. Compare against the limit minus one, or increment in an
  earlier word.
- **Chain LR arithmetic** (rule 4). An LR sub-slot cannot use another
  sub-slot's result from the same word. Put the dependent step in the next
  word.

```asm
# given: CR2 = 3
loop:
    INC LR0 1 ; BNE LR0 CR2 loop ;;
    HALT ;;
# expect: LR0 == 4
```

The [ISA idioms](isa-idioms.md) catalogue shows these patterns in
the kernels, including the pointer bump that needs an LR-only word
([A12](isa-idioms.md#a12-post-increment-addressing-a-load-feeds-the-next-words-multiply)).

## What the emulator does not model

The emulator executes one word per cycle and never stalls. It has no model of
the hardware pipeline's stage latencies or of the cache agent's backpressure,
so the stall-limit fault F-8 never fires
([OQ-28](../requirements/open-questions.md#emulator-and-isa)).

The hardware runs each word through six pipelined stages
([Architecture and pipeline](../requirements/architecture.md#pipeline)), and
IPU-ARCH-13 requires that pipelining not change any architectural result. The
timing contract is that architectural result: it states which values a word
sees, whatever the stage latencies. The
[Control Stage Specification](../specs/stage-control.md) describes the LR
forwarding from CTRL to the later stages.

The contract settles two rows of the requirements' open questions
([OQ-9 and OQ-10](../requirements/open-questions.md#conflicts-between-existing-documents)):
`MULT` does not see the same word's load (rule 2), and every stage after CTRL
reads LRs after the LR writes (rule 1).
