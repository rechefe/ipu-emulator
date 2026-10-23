# Commit record and trace: versioned JSON schema in `ipu_isa`

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-29 · Size M · Depends on: E04-11, E04-19 · Behaviour change: none (new output only)

## Goal

Define what one retired cycle looks like, in a form any language can read (Q12, Q60):

- a **commit record** per cycle: PC before and after, the op retired in each slot, register writes, XMEM writes, and the outcome (retired, break, halt, fault);
- a **trace**: a header plus a sequence of commit records.

The contract is a versioned JSON Schema stored in `ipu_isa`. The Python `CommitRecord` is a thin layer over it. A cocotb testbench, a future ETISS-based model (Q56) or a docs trace viewer (E08, Q55) reads the same files.

## Background

- Nothing like this exists. Debuggers and the alias profiler read emulator internals directly.
- After E04-05 and E04-11, every architectural write is observable through public events: `RegFile.add_observer` and `XMem.add_observer` (ranged writes), and `CycleObserver.issued/retired` with `IssueFacts`.
- Register names come from `ipu_isa` (D2: `instruction_spec.py`, registers included, is the single source of truth).

## Implementation

1. Add `ipu_isa/schemas/commit_record.v1.json` and `ipu_isa/schemas/trace.v1.json` (JSON Schema 2020-12). Sketch:
   ```json
   {
     "cycle": 1234,
     "pc_before": 17, "pc_after": 18,
     "slots": [{"slot": "mult", "mnemonic": "MULT.RC.VV", "fields": {"rc_idx": 3, "ra": 0}}],
     "reg_writes": [{"register": "LR", "index": 3, "value": "0x00000004"},
                    {"register": "MULT_RES", "offset": 0, "data": "<hex, little-endian>"}],
     "xmem_writes": [{"address": 4096, "data": "<hex>"}],
     "outcome": "retired",
     "fault": null,
     "arith_events": []
   }
   ```
   Trace header: `schema_version`, `isa_digest` (hash of the ISA definition from E03), `format`, `program_digest`, `emulator_version`. Traces are JSON Lines: header line, then one record per line.
2. Rules that keep the contract language-neutral: integers as JSON numbers up to 32 bits, wider data as lowercase hex strings, little-endian; register names exactly as in `ipu_isa`; NOP slots omitted; ops tagged EMULATOR (Q17, Q39) carry `"tag": "emulator"` so a hardware consumer can filter them.
3. Versioning (Q11): semantic version in the schema `$id`. Adding an optional field is a minor bump; removing or renaming a field, or changing a meaning, is a major bump.
4. `ipu_emu/commit.py`: frozen `CommitRecord` dataclass with `to_json()`/`from_json()`, and `CommitCollector`, a `CycleObserver` plus register/XMEM observer that builds one record per cycle.
5. `ipu_emu/trace.py`: `TraceWriter(path, header)` and `read_trace(path)`.

## Tests

- [ ] Every record produced over the full emulator test suite validates against the schema (sampled to keep the run under 10 s).
- [ ] Round trip: `CommitRecord.from_json(r.to_json()) == r`.
- [ ] Register and XMEM writes in the record equal the state difference between cycles, for a set of programs covering every slot.
- [ ] A fault cycle has `outcome == "fault"`, the fault ID and PC, and no writes (after E04-20).
- [ ] A schema snapshot test fails when the schema changes without a version bump.

## Acceptance Criteria

- [ ] The schema lives in `ipu_isa` and has no Python-specific types.
- [ ] Collecting records costs nothing when no collector is attached (E04-02 benchmark unchanged).

## Open question

- Validating against JSON Schema in tests needs the `jsonschema` package as a test-only dependency (runtime stays pure Python + numpy, Q63). Accept it, or write a small hand-rolled validator?
