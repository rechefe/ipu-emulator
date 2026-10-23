# How-to: use the reference model from a cocotb testbench

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-13 · Size S · Depends on: E09-06, E04 (stable reference-model API, commit record, JSON schema, conformance suite) · Behaviour change: none

## Goal

An RTL engineer can drive the emulator as a reference model from a cocotb
testbench and compare the RTL against it, cycle by cycle or at the end of a
run, after reading one page.

## Background

- Q10: the reference-model consumer is a Python API driven by a cocotb-like
  testbench, with an optional trace writer.
- Q11, Q12: a small, versioned module (`reset`, `load_program`, registers by
  ISA name, XMEM access, `step`, `run`); `step()` returns a commit record (PC
  before and after, retired op per slot, register writes, XMEM writes,
  break/halt). Breaking changes bump the major version.
- Q47: the element format is passed at `reset()`.
- Q60: the commit record and trace are a versioned JSON schema in `ipu_isa`,
  with a conformance suite stored as data.
- Q59, OQ-28: F-8 is not modelled; the page must say so.
- No page documents any of this today.

## Implementation

1. Write `how-to/reference-model-cocotb.md` with
   `anthropic-skills:technical-writing`:
   - What the reference model is, and which API is stable (link E09-16).
   - Set up: depend on the reference-model package from the testbench (Bazel
     target, or how to put it on `PYTHONPATH` for a non-Bazel cocotb flow).
     Open question: is a wheel published for non-Bazel RTL repos?
   - Lock-step comparison: one cocotb coroutine that calls `step()` per
     retired word and compares the commit record with the RTL's retire
     monitor.
   - End-of-run comparison: `run()`, then compare registers and XMEM.
   - Trace comparison: write a JSON trace, compare it offline.
   - What the model does not model: timing stalls, F-8, the AaQ placeholder
     (link OQ-20, OQ-28).
2. The example testbench is a real file in the repo, run by a test against a
   fake DUT (a second instance of the model), and included with
   `pymdownx.snippets`. Coordinate with E04's conformance-suite issue, which
   may already have such a file.

## Tests

- [ ] The included example runs as a `py_test` (no simulator needed; the DUT is
      the model itself or a stub).
- [ ] `bazel test //docs:test_doc_commands` passes.

## Acceptance Criteria

- [ ] An RTL engineer can write a lock-step comparison from this page alone.
- [ ] The example code is tested.
- [ ] Unmodelled behaviour is listed with OQ links.
