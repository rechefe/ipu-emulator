# Reference model API

`ipu_emu.reference` is the emulator's stable interface for RTL verification
(API version `1.2.0`). A cocotb testbench resets the model, loads a program,
sets registers and XMEM, steps it one word at a time and compares each commit
record with what the RTL retired. The contract is the module's names
(`API_VERSION`, `ReferenceModel`, `RunResult`, `RunState`, `Stats`,
`Status`, `TraceWriter` and `read_trace`) and the commit records `step()`
returns. The rest of `ipu_emu` may
change between releases; this interface changes only with a version bump, and
a breaking change bumps the major number.

## A lockstep test

```python
from pathlib import Path

from ipu_emu.reference import ReferenceModel
from ipu_isa.formats import ElementFormat


async def test_lockstep(dut):
    model = ReferenceModel()
    model.reset(ElementFormat.INT8)
    model.load_program(Path("prog.bin").read_bytes())  # a program file, or its words as ints
    model.write_register("lr", 5, index=0)              # LR0 = 5
    model.write_xmem(0, bytes(range(128)))              # XMEM row 0

    while True:
        record = model.step()                           # one word's CommitRecord
        rtl = await rtl_retire(dut)                     # your testbench's own monitor
        assert record.to_json() == rtl
        if record.outcome in ("halt", "fault"):         # model.status names the fault
            break
```

`step()` returns the word's commit record, the same object a run writes to a
trace; its JSON form follows `ipu_isa/schemas/commit_record.v1.json`. Its
`outcome` is `retired`, `halt` or `fault`: the model never stops at a `BREAK`
slot, so `break` does not occur. A stepped record's `cycle` counts the words
since the last `reset()` or `load_program()`; in a trace `run()` writes, it
counts the trace's records, as the schema says.

To run without comparing each word, `run(max_cycles=None, *, trace=None)`
steps until a `HALT` word, a fault or the cycle limit. It returns a
`RunResult` with the final `status` and the words retired (`cycles`). Pass a
`TraceWriter` as `trace` to write each record as its word retires, and read
the file back with `read_trace`:

```python
from ipu_emu.reference import TraceWriter, read_trace

with TraceWriter("run.jsonl", model.trace_header()) as trace:
    model.run(trace=trace)
header, records = read_trace("run.jsonl")
```

## Methods

| Method | What it does |
|---|---|
| `reset(fmt)` | Puts registers and XMEM to their reset values (`CR15` holds the default dstructure) and the PC to 0, enables every fault with no sticky flag, and clears the arithmetic flags and the statistics, in element format `fmt`. The program stays. |
| `load_program(words)` | Loads a program file's bytes or its words as unsigned ints; the PC goes to 0. |
| `read_register(name, index=0)` | Register `index` of bank `name`, as `ipu_isa.registers` names them (`lr`: `LR0`-`LR15`, `cr`: `CR0`-`CR15`, `r`: `R0`-`R1`, ...). An unsigned int for `lr` and `cr`, the register's bytes for the others. |
| `write_register(name, value, index=0)` | The same, written; only while halted. |
| `read_xmem(address, size)` / `write_xmem(address, data)` | XMEM bytes; writes only while halted. |
| `set_fault_enable(fault, enabled)` | A fault's enable bit. F-2 to F-6 cannot be cleared (OQ-2). |
| `step()` | One word. A `BREAK` slot never stops it: the word retires. A step has no cycle limit. |
| `run(max_cycles=None, *, trace=None)` | Steps until `HALT`, a fault, or `max_cycles` words (`None` keeps the model's limit, 0 disables it). |
| `stats()` | A `Stats` copy of the run statistics since the last `reset()` or `load_program()` (1.1.0): `cycles`, `mult_active_cycles`, `acc_active_cycles`, `xmem_reads`, `xmem_writes`, `mult_lane_ops`, `mult_identity_lane_ops`, `lane_occupancy` and `effective_mac_utilization` (`(mult_lane_ops - mult_identity_lane_ops) / (cycles * 128)`). They depend on operand values, so no commit record carries them. |
| `fault_enabled(fault)` | Whether a fault's enable bit is set (1.2.0). |
| `fault_flags()` | The sticky fault flags since the last `reset()`, as a `frozenset` of `FaultId`: every fault that happened, whether it halted or, disabled, did not (1.2.0). |
| `arith_flags()` | The sticky arithmetic-event flags since the last `reset()`, as a `frozenset` of `ipu_isa.faults.ArithmeticEvent` (1.2.0). |
| `trace_header()` | The `trace.v1.json` header of the loaded program, for a `TraceWriter` (1.2.0). |
| `status` | A `Status`: `state` (`RunState.RUNNING`, `HALTED` or `FAULTED`), `pc` and, when faulted, `fault`. |

`r_wide_debug` and `r_cyclic_wide_debug` are emulator-only banks: they hold
the operands of the wide formats (FP32, INT32), which the hardware does not
have.

## Rules

- The model is `HALTED` after a reset, a program load or a `HALT` word, and
  `RUNNING` once stepped. After a `HALT` word or a fault, `status.pc` is that
  word's PC.
- The host accesses the IPU refuses with an AXI error raise
  `ConfigurationError`: writing a register, XMEM or a program while running;
  writing `CR0` or `CR1` (constants in hardware); a bank, index or range that
  does not exist; a value that does not fit.
- After a fault, `step()`, `run()`, `load_program()` and the register and
  XMEM writes raise `ConfigurationError` until `reset()` (IPU-FLT-7). After a
  `HALT` word, `step()` and `run()` raise it until `reset()` or
  `load_program()`.
- No method returns an emulator object: values are ints, floats, bytes, the
  dataclasses above and `CommitRecord`.
- Version 1.1.0 added `Stats` and `stats()`; 1.2.0 added the flag and enable reads, `trace_header()`, and `TraceWriter` and `read_trace` to the contract. Nothing earlier changed.
- The contract test, `src/tools/ipu-emu-py/test/test_reference_api_contract.py`,
  pins every public name, signature and record field under its version; a
  strict mypy test (`//src/tools/ipu-emu-py:mypy_reference`) types the module.

## Conformance suite

`src/tools/ipu-isa/conformance/` holds the ISA's expected behaviour as data:
for each case a program (`program.asm`, `program.bin`), its starting state
(`setup.json`, schema `setup.schema.json`) and the trace of its run
(`expected.trace.jsonl`). An RTL testbench runs each case as the lockstep test
above does: reset in the case's format, load `program.bin`, zero every
register and XMEM byte (`CR0` and `CR1` aside), write the registers and XMEM
regions of `setup.json`, set its fault enables, step until
`HALT`, a fault or `max_cycles`, and compare each commit record with the
expected one. It skips the cases marked `emulator_only`. The suite's
`README.md` has the steps; `bazel test //src/tools/ipu-emu-py:test_conformance`
runs every case through this API.
