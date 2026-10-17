# Reference model API

`ipu_emu.reference` is the emulator's stable interface for RTL verification
(API version `1.0.0`). A cocotb testbench resets the model, loads a program,
sets registers and XMEM, steps it one word at a time and compares each commit
record with what the RTL retired. Only the names below are the contract:
`API_VERSION`, `ReferenceModel`, `RunResult`, `RunState` and `Status`. The rest
of `ipu_emu` may change between releases; this module changes only with a
version bump, and a breaking change bumps the major number.

## A lockstep test

```python
from ipu_emu.reference import ReferenceModel
from ipu_isa.formats import ElementFormat

model = ReferenceModel()
model.reset(ElementFormat.INT8)
model.load_program(open("prog.bin", "rb").read())   # ipu-as output, or a list of int words
model.write_register("lr", 5, index=0)               # LR0 = 5
model.write_xmem(0, bytes(range(128)))               # XMEM row 0

while True:
    record = model.step()                            # one word's CommitRecord
    rtl = await rtl_retire(dut)                      # your testbench's own monitor
    assert record.to_json() == rtl
    if record.outcome in ("halt", "fault"):          # model.status says which fault
        break
```

`step()` returns the word's commit record, the same object a run writes to a
trace; its JSON form follows `ipu_isa/schemas/commit_record.v1.json`. Its
`outcome` is `retired`, `break` (the word's `BREAK` fired; the word still
retired), `halt` or `fault`. To run without comparing each word, `run(max_cycles=None, trace=None)` steps until a
`HALT` word, a fault or the cycle limit, and returns a `RunResult` with the
final `status` and the words retired (`cycles`). Pass a `TraceWriter`
(`ipu_emu.trace`) as `trace` to write every record.

## Methods

| Method | What it does |
|---|---|
| `reset(fmt)` | Clears registers, XMEM, the PC, the fault controls (every fault enabled, no sticky flag), the arithmetic flags and the statistics, in element format `fmt`. The program stays. |
| `load_program(words)` | Loads a program file's bytes or its words as integers; the PC goes to 0. |
| `read_register(name, index=0)` | An `int` for `lr` and `cr` (register `index`), the element's bytes for a vector register (element `index`). Names are `ipu_isa.registers`'. |
| `write_register(name, value, index=0)` | The same, written; only while halted. |
| `read_xmem(address, size)` / `write_xmem(address, data)` | XMEM bytes; writes only while halted. |
| `set_fault_enable(fault, enabled)` | A fault's enable bit. Only F-1 and F-9 may be cleared. |
| `step()` | One word. A `BREAK` slot never stops it: the word retires. |
| `run(max_cycles=None, *, trace=None)` | Steps until `HALT`, a fault, or `max_cycles` words (`None` keeps the model's limit, 0 disables it). |
| `status` | A `Status`: `state` (`RunState.RUNNING`, `HALTED` or `FAULTED`), `pc` and, when faulted, `fault`. |

## Rules

- The model is `HALTED` after a reset, a program load or a `HALT` word, and
  `RUNNING` once stepped. After a `HALT` word or a fault, `status.pc` is that
  word's PC.
- Writing a register or XMEM while running, writing `CR0` or `CR1` (constants
  in hardware), a register or range that does not exist, and a value that does
  not fit raise `ConfigurationError`.
- After a fault, `step()`, `run()` and every write raise `ConfigurationError`
  until `reset()` (IPU-FLT-7). After a `HALT` word, `step()` and `run()` do the
  same until `reset()` or `load_program()`.
- No method returns an emulator object: values are `int`, `bytes`, the
  dataclasses above and `CommitRecord`.
- The contract test, `src/tools/ipu-emu-py/test/test_reference_api_contract.py`,
  pins every public name and signature; a strict mypy test
  (`//src/tools/ipu-emu-py:mypy_reference`) types the module.
