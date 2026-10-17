# Reference model API

`ipu_emu.reference` is the emulator's stable interface for RTL verification.
A cocotb testbench resets the model, loads a program, sets registers and
XMEM, steps it one word at a time and compares each commit record with what
the RTL retired. [How-to: use the reference model from
cocotb](../how-to/reference-model-cocotb.md) walks through a testbench.

--8<-- "generated/reference-api/version.md"

The [API](#api), [schema](#commit-record-schema) and [fault](#faults) sections
of this page are generated from the code and the schema files, so they always
match the version above.

## Compatibility

The contract is:

- the names in `ipu_emu.reference.__all__`, with their signatures, dataclass
  fields and enum values;
- the commit records `step()` returns (`ipu_emu.commit`), field by field;
- the commit record and trace header schemas, which are language-neutral JSON
  Schema files in `ipu_isa/schemas/`.

Everything else in `ipu_emu` is internal and can change in any release. The
other symbols on this page (`TraceWriter`, the schema helpers, the argument
types) are listed because the contract hands them out or takes them, not as
a separate promise.

`API_VERSION` is a semantic version:

- **Major** (breaking): a public name removed or renamed, a parameter removed
  or made required, a field removed, renamed or retyped, or a meaning changed
  (a method that now refuses what it accepted, a field that now counts
  something else).
- **Minor**: an addition that leaves every existing call working, such as a
  new method or a new optional parameter. Version 1.1.0 added `Stats` and
  `stats()`; 1.2.0 added the flag and enable reads (`fault_enabled()`,
  `fault_flags()`, `arith_flags()`), `trace_header()`, and `TraceWriter` and
  `read_trace` to `__all__`; 1.3.0 added `debug_run()`,
  `attach_alias_profile()` and `alias_profile()`. Nothing earlier changed.

`debug_run()` is the one escape hatch for tools such as the debugger: its
callback receives the emulator's internal state, and that argument's type,
like its `control` argument's, is outside the contract. The method's
signature and what it returns follow the version.

Each schema carries its own version, at the end of its `$id`. An optional
field added is a minor bump; a field removed or renamed, or a meaning
changed, is a major one. A trace header names the commit record schema's
version (`schema_version`).

Two tests hold the promise. `src/tools/ipu-emu-py/test/test_reference_api_contract.py`
pins every public name, signature and record field under the version that
introduced it, so a surface change fails until a new version's entry matches.
`src/tools/ipu-emu-py/test/test_commit.py` pins each schema file's bytes to
its version. A strict mypy test (`//src/tools/ipu-emu-py:mypy_reference`)
types the module.

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
trace. Its `outcome` is `retired`, `halt` or `fault`: the model never stops at
a `BREAK` slot, so `break` does not occur. A stepped record's `cycle` counts
the words since the last `reset()` or `load_program()`; in a trace `run()`
writes, it counts the trace's records, as the schema says.

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
  dataclasses below and `CommitRecord`.
- Registers are named by bank, as `ipu_isa.registers` names them (`lr`:
  `LR0`-`LR15`, `cr`: `CR0`-`CR15`, `r`: `R0`-`R1`, ...), and `index` picks
  the register within the bank. `r_wide_debug` and `r_cyclic_wide_debug` are
  emulator-only banks: they hold the operands of the wide formats (FP32,
  INT32), which the hardware does not have.

## API

--8<-- "generated/reference-api/api.md"

## Commit record and trace schema { #commit-record-schema }

A commit record's JSON form (`CommitRecord.to_json()`) is a document of the
commit record schema. A trace is a JSON Lines file: a header line of the trace
header schema, then one commit record per word. Download the schema files to
validate either with any JSON Schema (draft 2020-12) validator.

--8<-- "generated/reference-api/schema.md"

## Faults

A faulting word's commit record has `outcome` `fault` and names the fault in
`fault.id`; `status.fault` is the same fault as a `FaultId`. These are the
IDs of the [fault list](../requirements/faults.md#fault-list); the
[emulator section of Faults](../requirements/faults.md#in-the-emulator) says
what the model checks for each.

--8<-- "generated/reference-api/faults.md"

## Conformance suite

`src/tools/ipu-isa/conformance/` holds the ISA's expected behaviour as data:
for each case a program (`program.asm`, `program.bin`), its starting state
(`setup.json`, schema `setup.schema.json`) and the trace of its run
(`expected.trace.jsonl`). No case contains Python, so another implementation
(the RTL under cocotb, another model) needs only these files and the schemas.

To run the suite against another implementation, for each case:

1. Skip it if `setup.json` has `"emulator_only": true`.
2. Reset in the case's `format` and load `program.bin`.
3. Zero every register and XMEM byte (`CR0` and `CR1` aside), then write the
   registers and XMEM regions of `setup.json` and set its fault enables.
4. Run until `HALT`, a fault or `max_cycles`, recording a commit record per
   word.
5. Compare the header's `schema_version`, `isa_digest`, `format` and
   `program_digest` with the expected trace's, then each record, field by
   field.

The suite's `README.md` has the details of each step.
`bazel test //src/tools/ipu-emu-py:test_conformance` runs every case through
this API.
