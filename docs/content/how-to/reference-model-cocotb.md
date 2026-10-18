# Use the reference model from cocotb

This page shows how to drive the emulator as a reference model from a cocotb
testbench, and compare the RTL with it in three ways: word by word (lockstep),
at the end of a run, or through a trace file read offline.

The reference model is
[`ipu_emu.reference.ReferenceModel`](../reference/reference-model-api.md#ReferenceModel). Its names and the
commit records its `step()` returns are the stable part of the emulator: they
change only with a version bump of [`API_VERSION`](../reference/reference-model-api.md#API_VERSION), and a
breaking change bumps the major number. The rest of `ipu_emu` can change in any release. The
[Reference model API](../reference/reference-model-api.md) page lists every
method and rule; each call below links to its entry there.

Every Python block on this page comes from
`docs/examples/test_reference_model_cocotb.py`.
`bazel test //docs:test_reference_model_cocotb` runs that file with a second
model as the DUT, so the code shown here runs as written.

## Step 1: put the model on the testbench's path

The model is pure Python. It needs `ipu_emu`, `ipu_isa` and numpy. It does not
need the assembler, a simulator or a build step.

- **Bazel testbench.** Depend on `//src/tools/ipu-emu:ipu_emu_lib`. It
  brings in `//src/tools/ipu-isa:ipu_isa_lib`.
- **Any other cocotb flow** (a Makefile, a CI script). No wheel is published.
  Check out this repository at a fixed commit and add its two source
  directories to `PYTHONPATH`, in the environment that runs the simulator:

  ```bash
  export PYTHONPATH="$IPU_EMULATOR/src/tools/ipu-emu/src:$IPU_EMULATOR/src/tools/ipu-isa/src:$PYTHONPATH"
  pip install numpy
  ```

  `$IPU_EMULATOR` is the checkout. Without the Bazel build flag, prototyping
  mode is off, which is what a reference model needs.

Assemble the program the RTL runs into a program file:

```bash
bazel run //src/tools/ipu-as:ipu-as -- assemble --input "$PWD/prog.asm" --output "$PWD/prog.bin"
```

## Step 2: give the model and the RTL the same starting state

[`reset(fmt)`](../reference/reference-model-api.md#ReferenceModel-reset) sets the element format
([`ElementFormat`](../reference/reference-model-api.md#ElementFormat)) and the reset values (`CR15` holds the
default dstructure). [`load_program`](../reference/reference-model-api.md#ReferenceModel-load_program) takes
the program file's bytes. Then write the registers
([`write_register`](../reference/reference-model-api.md#ReferenceModel-write_register)) and XMEM
([`write_xmem`](../reference/reference-model-api.md#ReferenceModel-write_xmem)) the test needs. Writes are allowed only while the model
is halted, which it is after a reset or a program load.

```python
--8<-- "docs/examples/test_reference_model_cocotb.py:imports"
--8<-- "docs/examples/test_reference_model_cocotb.py:prepare"
```

Write the same values into the RTL, over AXI or by backdoor. `CR0` and `CR1`
are constants and cannot be written. A register is named by its bank, as
`ipu_isa.registers` names them (`lr`, `cr`, `r`, `r_acc`, `post_aaq_reg`, ...),
and an `index` within the bank. `lr` and `cr` take and return unsigned 32-bit
ints; the other banks take and return bytes.

## Step 3: compare word by word (lockstep)

A lockstep comparison calls [`step()`](../reference/reference-model-api.md#ReferenceModel-step) once for each
word the RTL retires, and compares the model's commit record
([`CommitRecord.to_json()`](../reference/reference-model-api.md#CommitRecord-to_json)) with the one the RTL's
retire monitor builds.

Your testbench provides the monitor: a coroutine (`rtl_retire` below) that
awaits the next retired word and returns it as a dict with the commit record's
fields. Their JSON form is defined by
`src/tools/ipu-isa/src/ipu_isa/schemas/commit_record.v1.json`
([every field](../reference/reference-model-api.md#commit-record-schema)):

| Field | What it holds |
|---|---|
| `cycle` | The words retired since the last `reset()` or `load_program()`, before this one. |
| `pc_before`, `pc_after` | The word's PC, and the PC of the next word. |
| `slots` | Each non-`NOP` op: `{"slot", "mnemonic", "fields"}`, `fields` being the encoded operand values by operand name. |
| `reg_writes` | Each register the word wrote, at its end-of-cycle value: `{"register", "index", "value"}` for `lr` and `cr`, `{"register", "offset", "data"}` (hex bytes) for a vector register. |
| `xmem_writes` | Each XMEM write: `{"address", "data"}`, the data as hex bytes. |
| `outcome` | `retired`, `halt` (a `HALT` word) or `fault`. |
| `fault` | `{"id", "pc"}` when the word faulted (`"F-3"`, ...), else `null`. |
| `arith_events` | The arithmetic events the word raised. |

A write that leaves a value unchanged is still a write. A faulting word has
no slots and no writes: nothing of it retires.

The comparison loop stops at the first `halt` or `fault` record. It names the
word, its PC and the first field that differs:

```python
--8<-- "docs/examples/test_reference_model_cocotb.py:lockstep"
```

If your monitor cannot see a field, such as `slots`, pass `fields` without it.

The cocotb test prepares both sides and runs the loop, then checks
[`status`](../reference/reference-model-api.md#ReferenceModel-status) for [`RunState`](../reference/reference-model-api.md#RunState)
`HALTED`. `load_rtl` and `rtl_retire` are your testbench's coroutines:

```python
import cocotb

--8<-- "docs/examples/test_reference_model_cocotb.py:cocotb"
```

A `BREAK` slot never stops `step()`: the word retires, so the outcome
`break` does not occur. If the RTL stops at a breakpoint, resume it in the
monitor before it reports the word.

## Step 4: compare at the end of a run

When the RTL cannot report each word, run both to the end and compare the
state. [`run()`](../reference/reference-model-api.md#ReferenceModel-run) steps until a `HALT` word, a fault,
or the cycle limit, and returns a [`RunResult`](../reference/reference-model-api.md#RunResult): the final
`status` and the number of words retired (`cycles`). Then read the registers
([`read_register`](../reference/reference-model-api.md#ReferenceModel-read_register)) and XMEM
([`read_xmem`](../reference/reference-model-api.md#ReferenceModel-read_xmem)) to compare.
Compare that count with the words the RTL retired, not with its clock cycles:
the RTL can stall, the model does not.

```python
--8<-- "docs/examples/test_reference_model_cocotb.py:end-of-run"
```

`rtl_run`, `rtl_read_register` and `rtl_read_xmem` are your testbench's
coroutines: start the RTL and await its done signal, and read a register or
XMEM range by backdoor.

## Step 5: compare traces offline

A trace is a JSON Lines file: a header line
(`ipu_isa/schemas/trace.v1.json`), then one commit record per word. Pass a
[`TraceWriter`](../reference/reference-model-api.md#TraceWriter) to `run()` to write the model's trace as each
word retires. The header's fields come from
[`SCHEMA_VERSION`](../reference/reference-model-api.md#SCHEMA_VERSION), [`isa_digest()`](../reference/reference-model-api.md#isa_digest) and
[`EMULATOR_VERSION`](../reference/reference-model-api.md#EMULATOR_VERSION):

```python
--8<-- "docs/examples/test_reference_model_cocotb.py:trace-write"
```

In a trace, a record's `cycle` counts the records before it. The program
digest is the SHA-256 of the program file.

Have the RTL testbench write the same format, then compare the two files after
the simulation. The comparison reads plain JSON, so it needs nothing from the
emulator:

```python
--8<-- "docs/examples/test_reference_model_cocotb.py:trace-compare"
```

The conformance suite in `src/tools/ipu-isa/conformance/` holds a program,
starting state and expected trace for every hardware instruction, timing probe
and modelled fault. Its `README.md` says how a testbench runs each case; skip
the cases marked `emulator_only`.

## What the model does not model

Compare only what the model defines. It does not model:

- **Timing.** The model retires one word per cycle and never stalls. It has
  no pipeline latencies and no cache-agent backpressure, so compare words
  retired, not clock cycles ([Timing model](../explanation/timing-model.md)).
- **F-8, the word stall limit.** With no stall model it never fires
  (OQ-28 in [Open questions](../requirements/open-questions.md#emulator-and-isa)).
  Clearing its enable bit changes nothing.
- **F-7.** There is no table configuration, so no table is read-only or
  write-only ([Faults](../requirements/faults.md#in-the-emulator)).
- **The AaQ stage.** `ACTIVATE.QUANTIZE` is a placeholder: the AaQ spec's
  LUT, `LOAD` operation and block scale are not modelled, so an RTL that
  implements the spec differs from the model in AaQ results (OQ-20 and OQ-25
  in [Open questions](../requirements/open-questions.md#emulator-and-isa)).
- **The disabled behaviour of F-2 to F-6.** `set_fault_enable` refuses to
  clear their enable bits (OQ-2 in
  [Open questions](../requirements/open-questions.md#faults)).

The model also has behaviour the hardware does not. Skip it in an RTL test:
the wide formats (`FP32`, `INT32`), the `r_wide_debug` and
`r_cyclic_wide_debug` banks, and the emulator-only `acc_store` slot
(`ST.ACC`). In a commit record, a slot op or register write of these carries
`"tag": "emulator"`.
