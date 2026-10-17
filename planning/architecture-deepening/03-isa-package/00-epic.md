# Epic E03: ISA package (`ipu_isa`)

ID E03 · Folder `03-isa-package` · Runs after E02 (needs E02-01 to E02-06) · Feeds E04, E05, E06, E08, E09

## Goal

Make the ISA one deep module: `ipu_isa`, a pure-data package that holds every fact about the instruction set exactly once and derives everything else (D2, Q13, Q31). Authors keep editing nested dicts in `instruction_spec.py` (Q30); a strict schema check and one loader turn them into a frozen, typed view that every consumer reads. The package also owns the binary codec (Q69), the hardware/emulator/assembler tags (Q17), the element formats (Q47), the declared timing contract (Q32) and the fault list (Q52, Q59). Adding an instruction then takes one ISA entry, one execute handler and one test, proven by a CI test (Q16).

## Background

Verified at HEAD 694a8a8. Details and line references are in each issue.

- **Not pure data.** `ipu_common` imports `ipu_as` inside two factories (`instruction_spec.py:1326`, `registers.py:318-333`), holds activation math, the emulator's alias catalogue and an SVG generator, and mutates other modules at import (`union_layout.py:268-280`).
- **Parallel tables.** Operand-type facts live in six tables across three packages (`VALID_OPERAND_TYPES`, `get_operand_type_bits`, `OPERAND_TYPE_MAP`, `_TYPE_FIELD_SUFFIX`, `_OPERAND_TYPE_TO_SV_TYPEDEF`, `OPERAND_TYPE_DETAILS`); slot facts in five or more (`SLOT_COUNT`/`SLOT_METADATA`/`COMPOUND_LAYOUT_SLOT_ORDER`, `_SLOT_FIELD_PREFIX`, `_SLOT_META`, `slot_to_class_name`, `_slot_to_inst` and the legend, `debug_pipeline` labels). Drift already exists.
- **Weak checks.** `validate_instruction_spec` ignores unknown keys and does not check `read`-flag legality, doc operand names or the syntax line. 30 syntax lines and 28 examples use commas the grammar rejects.
- **String-key consumers.** Consumers walk `INSTRUCTION_SPEC[slot][name]["operands"]` directly in about a dozen modules.
- **Codec in the assembler.** The emulator decodes programs through `ipu_as.compound_inst` (`ipu_emu/execute.py:30`), and decoded field names come from Python class names.
- **No hardware view.** The SV package contains the emulator-only ACC_STORE and BREAK slots (`IPU_COMPOUND_INST_WIDTH = 186`) although a comment says they are excluded, and its header changes on every commit.
- **Timing, formats and faults are implicit.** Slot order is call order in `ipu.py`; implicit register reads are undeclared; element format is `DType` plus three wide-mode flags in the emulator; the fault list exists only in `docs/content/requirements/faults.md`.
- **Inconsistent mnemonics.** Some use underscores (`LDR_MULT_REG`), branch signedness is not in the name, `BKPT` is called "conditional" but halts unconditionally.

## Decisions applied

| Decision | Applied in |
|---|---|
| D2 (one source; one edit propagates) | whole epic; E03-24 proves it |
| D3 (refactors output-identical) | every refactor issue cites E02 baselines; behaviour changes are E03-19, E03-21, E03-22 |
| D6 (better classification; `acc_store` not hardware) | E03-07, E03-22 |
| Q13, Q31 (`ipu_isa` = ISA only; move activations, alias spec, SVG out; no import of `ipu_as`) | E03-01, E03-02 |
| Q15 (derive `syntax`) | E03-04 |
| Q16 ("one edit" CI test) | E03-24 |
| Q17, Q39, Q41 (tags; BREAK is EMULATOR; emulator slots after hardware slots; emulator binary keeps 186 bits) | E03-07, E03-22, E03-23 |
| Q30 (dict authoring, strict schema, read-only typed view) | E03-03, E03-09, E03-10, E03-11 |
| Q32 (timing contract; data only here) | E03-14; emulator changes in E04-06, E04-12, E04-25 to E04-27 |
| Q33, Q45, Q46 (mnemonic standard, rename script, SV rename table) | E03-19 |
| Q40 (`BKPT` becomes hardware `HALT`; RTL control spec) | E03-19 (name), E03-20 (semantics and docs) |
| Q42 (`--target hardware`) | E05-15, using E03-07 and E03-16 |
| Q47 (element-format enum, FP32/INT32 EMULATOR) | E03-13 |
| Q52, Q53, Q59 (faults F-1 to F-9, F-8 not modelled, arithmetic events) | E03-15 |
| Q62 (one-way deps, import-graph test) | E03-02, E03-18 |
| Q69 (`ipu_isa` owns the binary codec; `src/tools/ipu-isa`) | E03-01, E03-16, E03-17, E03-18 |
| L32 (SV header hash of body) | E03-21 |
| L33 (geometry in the ISA package; apps migrate later) | E03-12 |
| L34 (incremental migration via adapters and `StrEnum`) | E03-06, E03-09 to E03-11 |

## Scope

- Package rename and moves; error root `IpuError`.
- Strict schema, derived syntax, operand-type and slot registries, tags, immutable layout, typed view and consumer migration.
- Geometry, ISA constants, element formats, timing contract data, fault list data.
- Binary codec (word, slot, disassembly) in `ipu_isa`.
- Mnemonic rename with script and SV rename table; `HALT`; SV package header and hardware-only content; RTL-facing docs.
- The "one edit" CI test.

## Out of scope

- Emulator behaviour: slot-order consumption, cycle view, timing flips, format codecs, fault model (E04).
- Assembler adoption of the slot codec, explicit field names in `get_fields`, `--target hardware`, located errors (E05).
- Moving `gen_docs`, `gen_codegen`, `gen_vscode` into `ipu_gen` (E08; E03-01 creates the package with the SVG generators).
- Kernel and app migration of `LANES`/`ROW_BYTES`/`ELEM_BYTES` copies (E06, E07).
- Numeric semantics of formats (bias, specials, rounding) and the AaQ stage (E04, Q35).

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? | SV `*_WIDTH` |
|---|---|---|---|---|---|---|
| E03-01 | [01-rename-to-ipu-isa.md](01-rename-to-ipu-isa.md) | Rename `ipu_common` to `ipu_isa` and move non-ISA modules out | M | E02-01 to E02-04 | none | unchanged |
| E03-02 | [02-isa-purity.md](02-isa-purity.md) | `ipu_isa` imports nothing from the assembler or the emulator | S | E03-01, E01 | none | unchanged |
| E03-03 | [03-strict-schema-check.md](03-strict-schema-check.md) | Strict schema check for the spec dicts | S | E03-01 | none | unchanged |
| E03-04 | [04-derived-syntax.md](04-derived-syntax.md) | Derive `syntax`; comma-free examples | S | E03-03 | docs text only | unchanged |
| E03-05 | [05-operand-type-registry.md](05-operand-type-registry.md) | Operand-type registry replaces six parallel tables | M | E03-02, E03-03 | none | unchanged |
| E03-06 | [06-slot-registry.md](06-slot-registry.md) | Slot registry replaces five parallel slot tables | M | E03-02, E03-03 | none | unchanged |
| E03-07 | [07-hardware-emulator-tags.md](07-hardware-emulator-tags.md) | HARDWARE / EMULATOR / ASSEMBLER tags and the hardware word | S | E03-06 | none | unchanged |
| E03-08 | [08-immutable-derived-layout.md](08-immutable-derived-layout.md) | Derived layout computed once, immutable, no import-time `setattr` | S | E03-05 | none | unchanged |
| E03-09 | [09-typed-isa-view.md](09-typed-isa-view.md) | Read-only typed ISA view from one loader | M | E03-03, 05, 06, 07, 08 | none | unchanged |
| E03-10 | [10-emulator-reads-typed-view.md](10-emulator-reads-typed-view.md) | Emulator reads the ISA through the typed view | M | E03-09 | none | unchanged |
| E03-11 | [11-assembler-and-tools-read-typed-view.md](11-assembler-and-tools-read-typed-view.md) | Assembler, generators and fixtures read the typed view | M | E03-09 | none | unchanged |
| E03-12 | [12-geometry-and-isa-constants.md](12-geometry-and-isa-constants.md) | Geometry and ISA constants live in `ipu_isa` | S | E03-05 | none | unchanged |
| E03-13 | [13-element-formats.md](13-element-formats.md) | Element-format enum with derived sizes and accumulator type | M | E03-07, E03-12 | none (data) | unchanged |
| E03-14 | [14-timing-contract-data.md](14-timing-contract-data.md) | Declare the timing contract as data | M | E03-09, E02-05, E02-06 | none (data) | unchanged |
| E03-15 | [15-fault-list-data.md](15-fault-list-data.md) | Fault list F-1 to F-9 and arithmetic events as data | S | E03-07, E03-09 | none (data) | unchanged |
| E03-16 | [16-word-codec.md](16-word-codec.md) | Instruction-word codec: field layout, decode, encode, program bytes | M | E03-05, 06, 07, 08 | none | unchanged |
| E03-17 | [17-slot-codec.md](17-slot-codec.md) | Slot codec: encode and decode one instruction | M | E03-09, E03-16 | none | unchanged |
| E03-18 | [18-disassembler.md](18-disassembler.md) | Disassembler in `ipu_isa`; emulator sources stop importing the assembler | S | E03-17 | none | unchanged |
| E03-19 | [19-mnemonic-standard-rename.md](19-mnemonic-standard-rename.md) | Mnemonic standard: rename script, schema rule, SV rename table | M | E03-04, E02-01 to E02-04 | yes: names only, binaries identical | unchanged; enum/member **names** change |
| E03-20 | [20-halt-hardware-instruction.md](20-halt-hardware-instruction.md) | `HALT` is a hardware COND instruction | S | E03-07, E03-19 | docs only | unchanged |
| E03-21 | [21-sv-header-content-hash.md](21-sv-header-content-hash.md) | SV package header: hash of the body | S | E02-03 | yes: header line | unchanged |
| E03-22 | [22-sv-package-hardware-only.md](22-sv-package-hardware-only.md) | SV package generated from the HARDWARE set only | S | E03-07, E03-11 | **yes: wire format** | **186 → 155; 4 removed** |
| E03-23 | [23-rtl-docs-hardware-only.md](23-rtl-docs-hardware-only.md) | RTL-facing docs show the hardware word | S | E03-22 | docs only | unchanged |
| E03-24 | [24-one-edit-ci-test.md](24-one-edit-ci-test.md) | "One edit" CI test with a dummy instruction | M | E03-11, 14, 15, 17, 18 | none (test) | unchanged |

## Suggested order

```
E02-01..06
    │
    ▼
   01 ──▶ 02 ──┬──▶ 05 ──┬──▶ 08 ──┐
    │          │         └──▶ 12 ──┼──▶ 13
    └──▶ 03 ───┼──▶ 06 ──▶ 07 ─────┤
         │     │                   ▼
         │     │                  09 ──┬──▶ 10
         │     │                       ├──▶ 11 ──▶ 22 ──▶ 23
         │     │                       ├──▶ 14
         │     │                       └──▶ 15
         │     └── 05,06,07,08 ──▶ 16 ──▶ 17 (+09) ──▶ 18
         └──▶ 04 ──▶ 19 ──▶ 20
                                   11,14,15,17,18 ──▶ 24
   E02-03 ──▶ 21   (any time)
```

- E03-01 lands first, when few PRs are open (large mechanical diff).
- E04 can start E04-06 after E03-14, E04-13 after E03-13, E04-19 after E03-15, E04-18 after E03-18.
- E05 can start E05-02 after E03-17, E05-15 after E03-07 and E03-16.
- Land E03-19 (names) and E03-22 (widths) close together, so the downstream RTL repo takes one breaking update.

## Cross-epic assumptions

- **E01** provides the import-graph test mechanism (E03-02 registers `ipu_isa`'s rule) and the code-conventions page.
- **E02** baselines exist before E03-01; every E03 refactor proves itself with them.
- **E04** consumes `SLOT_EXECUTION_ORDER`, the timing data and `KNOWN_DEVIATIONS` (E03-14), `ElementFormat` (E03-13), `FAULTS` (E03-15), `IpuError` (E03-01) and the codec (E03-16 to E03-18). E04-03 and E05-04 subclass `ipu_isa.errors.IpuError`.
- **E05** consumes the slot codec (E03-17), field-name layout (E03-16) and tags (E03-07); E05-16 removes hard-coded mnemonic text from `ipu_as` after E03-19's script rewrites it once.
- **E08** adopts `src/tools/ipu-gen/` created in E03-01 as the `ipu_gen` package and moves the generators into it; generators already take an `Isa` argument (E03-11).
- **E09-05**: action A-1 (`BKPT` → `END`) is updated to `HALT` by E03-20.

## Epic acceptance criteria

- [ ] `ipu_isa` imports no other repo package; no import-time mutation; no placeholder globals.
- [ ] Every operand-type, slot, format, fault and timing fact is written once; consumers read the typed view.
- [ ] The emulator and assembler share one binary codec in `ipu_isa`; `ipu_emu` sources import nothing from `ipu_as`.
- [ ] The SV package is generated from the HARDWARE set, with its header a hash of its body; the only `*_WIDTH` changes are the ones E03-22 announces.
- [ ] The "one edit" test passes with an empty list of manual edits once E04-14 and E05-02 have landed.
- [ ] `//src/tools/ipu-isa:mypy` is `mypy_test(name = "mypy", lib = ":ipu_isa_lib", strict = True)` and passes (Q65, E01-11; E03-09 makes it strict).
