# Architecture deepening — issue drafts

This directory holds the plan for restructuring the IPU emulator repository: 200 small, independently mergeable issues in 10 epics. The issues came out of an architecture review and a decision session with the repository owner. Each file is a ready-to-file GitHub issue draft, in the same style as [`../riscv-host-integration/`](../riscv-host-integration/README.md).

- **Decisions:**
  - [DECISION-LOG.md](DECISION-LOG.md) records every decision from the session, with question IDs (Q1–Q83, L1–L43).
  - [IMPLEMENTATION-DECISIONS.md](IMPLEMENTATION-DECISIONS.md) gives the defaults chosen for questions the drafts left open.
  - [`docs/adr/`](../../docs/adr/) holds the load-bearing decisions.
- **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md). Every issue uses its terms.

## Epics

| # | Folder | Title | Issues |
|---|---|---|---|
| 00 | [`00-verified-fixes`](00-verified-fixes/00-epic.md) | Epic E00: Verified fixes | 12 |
| 01 | [`01-foundation`](01-foundation/00-epic.md) | Epic E01: Foundation — pinned toolchain, lockfile, fast CI, lint, conventions | 15 |
| 02 | [`02-safety-nets`](02-safety-nets/00-epic.md) | Epic E02: Safety nets before the refactors | 9 |
| 03 | [`03-isa-package`](03-isa-package/00-epic.md) | Epic E03: ISA package (`ipu_isa`) | 24 |
| 04 | [`04-emulator-core`](04-emulator-core/00-epic.md) | Epic E04: Emulator core — one cycle module, format codecs, faults and a stable reference-model API | 31 |
| 05 | [`05-assembler`](05-assembler/00-epic.md) | Epic: Assembler as a deep module — slot codec, assembly session, located errors, strict templates | 16 |
| 06 | [`06-kernel-framework`](06-kernel-framework/00-epic.md) | Epic: Kernel framework (`ipu_harness`): a kernel is an `.asm`, a `kernel.toml` and an optional reference | 21 |
| 07 | [`07-kernel-migration`](07-kernel-migration/00-epic.md) | Epic: Kernel migration onto `ipu_harness`, family by family | 25 |
| 08 | [`08-debug-and-tools`](08-debug-and-tools/00-epic.md) | Epic: Debugger, generators and editor as deep modules with thin adapters | 27 |
| 09 | [`09-docs`](09-docs/00-epic.md) | Epic E09: Documentation restructure, drift tests and open questions | 20 |
## Order

```
E00 ──▶ E01 ──▶ E02 ──▶ E03 ──┬──▶ E04 ──┬──▶ E06 ──▶ E07
                              └──▶ E05 ──┘
E08 and E09 run alongside, as soon as the issues they depend on land.
```

- **E00:** verified bug fixes that need no design.
- **E01:** tooling: pinned toolchain, lockfile, fast CI, ruff, code conventions.
- **E02:** golden harnesses. Every later refactor must leave their output identical.
- **E03:** builds the ISA package. E04 (emulator) and E05 (assembler) then move onto it.
- **E06 and E07:** E06 replaces the kernel harness with declarations, and E07 migrates the kernels family by family.

## Rules every issue follows

- **Refactors leave outputs identical:** binaries, SV package, test outputs, proven by the E02 harnesses. A behaviour change gets its own issue, marked `Behaviour change: yes`.
- **Wire format:** an issue that changes a `*_WIDTH` localparam in the generated SV package says so explicitly.
- **Tests stay green:** `bazel test //...` passes after every issue.

## Decision record

The authoritative list is [DECISION-LOG.md](DECISION-LOG.md). In short:
- **Source of truth:** `ipu_isa` holds everything about the ISA. Authors write dicts, checked by a strict schema; consumers read a typed view.
- **Timing contract:** Control reads LRs at cycle start; all other stages read them post-LR-write.
- **Element formats:** INT8 and E1–E7 on hardware, FP32 and INT32 in the emulator only. They replace wide mode, and instructions are written once.
- **Reference-model API:** stable, with a language-neutral JSON commit record and trace, for cocotb.
- **Faults:** F-1 to F-9 per the requirements; arithmetic events set sticky flags.
- **Mnemonics:** `UNIT.OP[.VARIANT]`, no underscores, `BKPT` → `HALT`, signed branches `.S`.
- **Kernels:** a kernel is `.asm` + `kernel.toml` + an optional PyTorch reference, and the harness owns XMEM placement.
- **Packages:** `ipu_isa`, `ipu_as`, `ipu_emu`, `ipu_debug`, `ipu_gen`, `ipu_harness`, with kernels in a top-level `kernels/`. Dependencies go one way only.
- **Out of scope:** AaQ (placeholder) and ETISS. Both are tracked in `docs/content/requirements/open-questions.md`.
