# Decision log (grilling session, 2026-10-14/15)

## Round 1
- D1 Scope: all 9 clusters. Order: verified bug fixes + toolchain + lockfile → golden safety nets → structural refactors (01/03/04 first, then 05/06, then 02/07/09). Deliverable: issue drafts in `planning/architecture-deepening/`.
- D2 Source of truth: `instruction_spec.py` is THE single source of truth (registers too). Guiding goal: make changes as easy as possible from this one file (one edit there propagates everywhere).
- D3 Refactor PRs must be output-identical (golden-proven); behaviour changes go in separate, labelled PRs.
- D4 The user decides hardware-contract questions directly.
- D5 Compatibility: expose a PARTIAL, FIXED, NON-BREAKING emulator API so it can serve as a reference model for the (RTL) simulator. Everything else may change freely (no shims needed).
- D6 SV package: keep current format; widths are always derived from sources. `acc_store` is not real hardware. Wanted: a better classification in `instruction_spec.py` that marks purely emulated logic (so SV/hardware outputs exclude it).
- D7 Prototyping mode stays ("there for a reason").
- D8 Wide-vector mode is needed, but rethink how it is implemented end to end.
- D9 JS stays out of Bazel; the extension is its own project.

## Round 2
- Q10 Reference-model consumer: Python API driven by cocotb-like testbench (+ optional trace writer).
- Q11 Stable API: small versioned module (reset/load_program/regs by spec name/XMEM/step/run); names from the ISA; contract tests; breaking = major bump. Everything else internal.
- Q12 step() returns a commit record (pc before/after, per-slot retired op, reg writes, XMEM writes, break/halt).
- Q13 One ISA package; rename `ipu_common` -> `ipu_isa`, strictly ISA definitions.
- Q14 -> see Q30.
- Q15 Derive `syntax` from mnemonic + operands.
- Q16 "One edit" acceptance test: ISA package + execute handler + test; everything else derived; CI dummy-instruction test.
- Q17 Tag every slot/opcode HARDWARE / EMULATOR / ASSEMBLER; SV + RTL docs from HARDWARE only.
- Q18 User: it is a problem that anything outside the Control stage uses snapshot LR values -> see Q32.
- Q19 RESHAPE reads this cycle's MULT_RES. Q20 MULT reads R_MASK at start of cycle. Q21 AAQ reads this cycle's R_ACC.
- Q22 Branch signedness must be clear from the name -> Q33.
- Q23 AGG docs: only valid_elements used.
- Q24 FP8 is configurable (DType E1..E7) -> Q34.
- Q25 AaQ: placeholder implementation; leave the whole AaQ stage alone for now.
- Q26 FP32 accumulator overflow = open question (currently raises OverflowError) -> new "Open issues" docs page. User wants to rethink the whole docs site (except specs & requirements) and rewrite docs with their `anthropic-skills:technical-writing` skill.
- Q27 reciprocal/rsqrt(0): placeholder (AaQ) -> out of scope.
- Q28 Errors must also qualify what would be an error on real hardware -> Q38.
- Q29 Yes to all L1-L43 defaults.
- User: if too much at once, break the work into steps after the session.

## Round 3
- Q30 Keep nested-dict authoring; add strict schema check at import; consumers use a read-only typed view from one loader.
- Q31 `ipu_isa` = ISA only (instructions, registers, operand types/enums, slots+tags, timing contract, derived encoding, geometry). Move out: activations.py -> ipu_emu, isa_alias_spec.py -> ipu_emu, SVG generators -> docs tooling. No import of ipu_as.
- Q32 Timing contract: (1) Control stage (LR, COND) reads LRs at cycle start; all other stages read post-LR-write (fix 3 attn_v_* kernels in a behaviour PR); (2) R0/R1/R_CYCLIC/R_MASK read by MULT at cycle start; (3) MULT_RES -> ACC/AGG/RESHAPE and R_ACC -> AAQ seen same cycle; (4) LR sub-slots read cycle-start values.
- Q33 (b) dotted signedness: BLT.S/BGE.S (BGT.S/BLE.S; .U reserved). ALSO: define a better, consistent mnemonic standard (some mnemonics use underscores, some don't) -> Q45.
- Q34 Rename fully_connected fp8_e4m3/e5m2 -> fp8_e4/e5; FP8 special-value convention goes to Open issues. Encoder bug fixed regardless.
- Q35 AaQ entirely out of scope; all AaQ items (incl. crashes) recorded as open issues.
- Q36 "Open issues" docs page: yes.
- Q37 Docs restructure by reader need (Getting started / How-to / Reference / Explanation / Specs+requirements / Open issues), skeleton early, pages rewritten as each area lands. Include debugging. ALSO: integrate the interactive debugger from branch `claude/relaxed-galileo-4luoj7` into the docs site, and let a reader upload or write a (small) kernel and run it themselves.
- Q38 IllegalProgram IS a HardwareFault: classes = HardwareFault (illegal program: LR write conflict, XMEM out of range, undefined opcode, mask race) and SimulationLimit (cycle limit, wide checks, running off the end). FP32 overflow open issue.
- Q39 break slot (BREAK, BREAK.IFEQ) = EMULATOR.
- Q40 (a) BKPT -> hardware HALT, added to RTL control spec.
- Q41 Keep emulator-only slots in the emulator binary (186 bits); emulator-only slots ordered after hardware slots.
- Q42 Assembler `--target hardware`: rejects EMULATOR instructions, emits hardware-only words. Default emulator.
- Q43 Wide mode = "run at full FP32". The configuration is specified in `ipu_isa`; instruction logic written once, no `if wide: else:`.
- Q44 Kernels are NOT wide-only; wide is only how they are simulated today. Must be able to simulate any kernel with FP8 too.

## Round 4
- Q45 Mnemonic standard accepted: Control stage plain RISC-style (variants via dot suffix), datapath UNIT.OP[.VARIANT], no underscores. Mapping: LDR_MULT_REG->LD.MULT, LDR_CYCLIC_MULT_REG->LD.MULT.CYCLIC, LDR_MULT_MASK_REG->LD.MASK, STR_POST_AAQ_REG->ST.AAQ, STR_ACC_REG->ST.ACC, INCR_MOD_POW2->INC.MODP2, BLT/BGE/BGT/BLE->.S, BKPT->HALT; rest unchanged. Script rename of .asm + docs; binaries unchanged.
- Q46 SV enum member renames accepted (names-only breaking); downstream pins a commit, inform with rename table.
- Q47 One element-format enum in ipu_isa: INT8, E1..E7 (hardware, runtime-selected), FP32, INT32 (EMULATOR). Derives element bytes, row bytes, accumulator type, registers. Replaces dtype + 3 wide flags; reference API takes format at reset(). All implementations written so they are unaffected by format.
- Q48 Handlers: decode lanes via format codec -> compute -> encode with format rounding/saturation. No format branches. Replaces X2.
- Q49 CR scalars decoded in active format (low element-width bits). Behaviour PR after audit. Must be documented.
- Q50 Yes. Wide = XMEM 4x wider; 128 elements per vector, only width changes. Kernels format-independent; layouts pack per format; per-format goldens; pilot family (projections) in FP32, INT8, E4.
- Q51 The "interactive debugger" = artifact "IPU Cycle Inspector" (https://claude.ai/artifact/4sTeRLzw27gc3z8JmhhMVB), a trace-replay viewer (per-cycle JSON traces recorded from the Python emulator, 14 kernels).

## Round 5
- Q52 Emulator faults = requirements F-1..F-9 with IDs; per-fault enable (halt + record code/PC vs sticky flag), all enabled at reset. User asked about F-8 (word stall limit). User: SimulationLimit will exist in HW as a cycle counter (= F-9).
- Q53 Accumulator overflow / NaN = arithmetic events per proposed requirement: IEEE result + per-event sticky flag, never halt.
- Q54 One open-questions page: existing docs/content/requirements/open-questions.md, new "Emulator and ISA" section, continuing OQ numbers.
- Q55 Cycle Inspector into docs: phase 1 (in program) recorded traces generated by Bazel via reference-model trace writer, curated kernels, regenerated each docs build; phase 2 (follow-up) Pyodide live mode, paste/upload small kernels.
- Q56 ETISS out of scope; keep a future migration in mind.
- Q57 LayerNorm will change completely (AaQ placeholder) -> my choice: single reference matching what the kernel computes (no eps); revisit with AaQ.
- Q58 Rewrite 9 plans with decisions, split into small issue drafts under planning/architecture-deepening/ + README (order, deps), commit + push to claude/tender-pascal-5fsay7. No GH issues / PR until asked.

## Round 6
- Q59 F-8 listed as "not modelled" (no stall model). Drop SimulationLimit. Two error kinds: hardware faults F-1..F-9 (ID + enable bit) and configuration errors (setup time only).
- Q60 Reference contract language-neutral: commit record + trace = versioned JSON schema in ipu_isa; conformance suite (programs + expected traces) stored as data; Python API is a thin layer. Keeps ETISS migration open.

## Round 7 (code structure & efficiency)
- Q61 Priorities: (a) dev loop / test+CI time, (b) emulator speed, (d) memory, (c) assembler. PLUS: code must be readable, maintainable, built on correct design patterns.
- Q62 One responsibility per package, one-way deps, enforced by Bazel deps + import-graph test, no cycle-dodging lazy imports.
- Q63 Pure Python + numpy; speed via codecs/vectorisation/paged XMEM/single cycle; ETISS later for step-change.
- Q64 Benchmark targets; track numbers (cycles/s, assemble time, peak memory). (User said "track numbers" - gate not explicitly confirmed; treat as tracking, gate optional.)
- Q65 ruff (lint+format) everywhere as Bazel test; strict typing on ipu_isa + reference API first, then spread.

## Round 8
- Q66 Written "Code conventions" docs page linked from CLAUDE.md (deep modules, ~600-line files / ~60-line functions via ruff complexity rules, type hints, Google docstrings, no flag arguments, no private cross-module access, tests via public interfaces, two error kinds). User: "we really need that - there's tons of code lines right now."
- Q67 Handlers split into one module per slot written against format codecs; Ipu/cycle = orchestration; TUI split per pane over DebugSession.
- Q68 GLOSSARY.md: row, instruction word, instruction, pc, ...; code renamed per area as refactored.
- Q69 Packages: ipu_isa (incl. binary codec), ipu_as, ipu_emu, ipu_debug, ipu_gen, ipu_harness, ipu_kernels; one dir each under src/tools/; import-graph test; wrappers removed. User: "we need to split to more packages".
- Q70 Kernel tests stay in kernel folders. User asks for a critical review of the code each kernel requires (agent running).

## Round 9
- Q71 CI 5-10 min. Code-review guideline: PR author states they ran all tests locally. Nightly pipelines for the heavy tier. Perf numbers tracked (no gate).
- Q72 Faster-without-changing-results encouraged (plan compilation cached by PC in the faithful path). User unsure what prototyping mode is (explained: opt-in fast path, 1.1-1.3x today).
- Q73 enormous tier out of default; nightly + --config=full.
- Q74 kernel.toml direction OK; references via PyTorch.
- Q74b PyTorch reference (test-only dep): users should be able to test vs PyTorch; KEEP option to test vs the user's own data.
- Q75b User wants the full picture of the kernel test framework: writing + testing a new kernel must be easy, clear, little code beyond the assembly (and maybe a reference).
- Q76 family registry.toml for shape variants (replaces L12).
- Q77/Q77b user does not see why the 16 MemoryApp kernels are special -> explain.
- Q78 fully_connected goldens from reference per format; delete .bin.
- Q79 design notes only in .asm/template header; shown on generated kernel docs page.
- User: must run /mattpocock-skills:teach (kernels) and /mattpocock-skills:setup-matt-pocock-skills themselves (disable-model-invocation).

## Round 10 (kernel authoring)
- Q80 Kernel = `<name>.asm` + `kernel.toml` (`[kernel]` op/formats, `[params]` domain, `[inputs]`/`[outputs]` shapes, `[abi]` name->CR, `[cases]` incl. sweeps and user files) + optional PyTorch `reference.py` (or shared reference by op). Run: `bazel test //kernels/<family>/<name>` (all cases x formats), `bazel run ... -- --case X --format e4 --debug`. Framework: validate params, place regions, pack per format, program CRs, assemble with case params as template values (`{{ cr.x }}`), run to HALT, unpack, PyTorch reference rounded to format, compare with format tolerance tier, report cycles/MAC util, optional trace. Escape hatches: pack.py (single-tensor transforms), test_*.py, stages hook.
- Q81 Automatic XMEM placement by default; optional `row =` pin.
- Q82 Cases may name .npy inputs and expected outputs (relative to kernel folder) or `--input x=path.npy` on the command line; compare to file if given, else reference.
- Q83 Delete MemoryApp and the whole-image path entirely; framework places/packs every region; pack.py only transforms one declared tensor; raw-image debugging via the debugger XMEM view.
- CODEOWNERS: deferred to the end of the program (after package split). Candidate owners from history: rechefe (admin), YuvalHarary (yuval93813; faults owner), DavidShenzi (david_sh), yarden-carmi (vscode/apps). Unassigned collaborators: ZevProg, alonab01, ranfeb02-sys, davidlevy1206-alt.
- Setup (mattpocock skills): GitHub Issues tracker; default triage labels; single-context GLOSSARY.md + docs/adr/. Labels must be created by the user with gh (not available in cloud session).
