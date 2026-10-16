# Implementation decisions

The issue drafts left some questions open. To keep implementation moving, these defaults apply. Each one can be overridden in PR review; an override becomes a follow-up issue.

The user decided the program-level questions in the grilling session. They are recorded in [README.md](README.md#decision-record) and in `docs/adr/`.

## Repository layout and packaging

| Question | Default | Raised in |
|---|---|---|
| Where do kernels live? | Top-level `kernels/<family>/<name>/` (one Bazel package per kernel: `//kernels/<family>/<name>`). The Python support code is `ipu_harness` under `src/tools/ipu-harness/`. Kernels need no Python package. | E06 epic, E06-01, E06-17 |
| Are mnemonic aliases kept after the rename? | No. All in-repo sources are renamed in the same change, and the old names are rejected with a hint that names the new mnemonic. | E03-19, E05-16 |
| Can an EMULATOR instruction sit inside a HARDWARE slot? | No. The schema check rejects it. `HALT` is a hardware instruction, so no case needs it. | E03-07 |
| Which slots warn as "simulation-only" after the tags land? | Only ACC_STORE. `SlotInfo.debug_only` marks it, alongside its EMULATOR tag. The assembler warning, `emulator_only_instructions` (kernel tests expect `DebugOnlyInstructionWarning` from it), the generated docs note, the editor's `hardware` flag, the alias spec's `hardware_only` and the debugger's `*` labels read `debug_only`, so BREAK (also EMULATOR) changes none of them. The hardware views (`word_bits(Tag.HARDWARE)`, E03-22 SV, E03-23 RTL docs, E05-15 `--target hardware`) read the tag. E03-23 and E05-15 decide whether the user-facing notes move to the tag. | E03-07, E03-23, E05-15 |
| Hardware binaries | `--target hardware` writes `<name>.hw.bin` with no header. | E05-15 |
| `check --json` schema files | `vscode-ipu-asm/schema/` | E05-08 |
| Template parameters | Declared in `kernel.toml [params]`. A standalone template may use an `ipu:params` header comment. | E05-09 |
| `--abi` input | A `kernel.toml` path. Only its `[abi]` table is read. | E05-13 |
| How are the dependencies between packages enforced? | One table, `tools/import_graph.toml`, lists the first-party packages each package may import. `//tools:test_import_graph` reads every source under `src/tools/*/src/<package>/` and fails on any other first-party import, including function-level imports and literal `importlib.import_module` calls. A package with sources and no row fails the test, and so does an import of an `ipu_*` package whose `:py_sources` are not in the test's data. Rows are per top-level package; "nothing imports X except Y" is expressed by listing X only in Y's row. Later issues add rows (E06-01, E08-03, E08-13) or delete the edges marked temporary (`ipu_emu` → `ipu_as`, planned for E03-18, went in E03-16 once the word codec removed its last use; `ipu_as` → `ipu_gen` in E08-14). E06-01 extended the schema: dotted subpackage keys (the longest key wins; `ipu_kernels.reference` may import only `ipu_isa`, stricter than "not `ipu_emu`" because the framework and the rest of `ipu_kernels` reach the emulator) and a `[deferred]` table of modules a key may import only inside a function (`torch` from `ipu_harness`). `ipu_harness` never imports `ipu_kernels`, not even inside a function (Q62); E06-09 and E06-16 load kernel-side modules with `importlib.import_module` on a name read from registry data, which is plugin loading, not an edge (the test sees only literal names). No issue adds a separate check. `//src/tools/ipu-isa:test_purity` stays as the run-time check that `ipu_isa` imports with only its own library on the path. | E03-02, E03-18, E06-01, E08-03, E08-13 |

## ISA and emulator

| Question | Default | Raised in |
|---|---|---|
| XMEM row count per format | The byte address space stays the same (512 MiB, allocated lazily). Addressable rows are that space divided by the row width. The hardware row limit is an open question in `requirements/open-questions.md`. | E03-13 |
| IMEM depth | Stays a constant in `ipu_isa`. Making it a run-time parameter is an open question. | E03-12 |
| What `HALT` does | Enters `HALTED` and reports kernel done. Fetching past the program end is fault F-2. | E03-20 |
| CR1 as an element scalar | CR0 and CR1 are hardware constants for 0 and 1. As an element scalar, CR1 reads as the encoding of 1.0 in the active format, so `MULT.VE … CR1` stays an identity multiply. As an LR or integer operand it is the integer 1. This is documented in the ISA reference. | E04-28 |
| Disabled faults | Only F-1 (the slot runs as `NOP`) and F-9 (the run continues) may be disabled. Disabling any other fault is a configuration error, until OQ-2 is answered. | E04-22 |
| Does a MULT product overflow set the accumulator overflow flag? | Yes. It sets the same arithmetic-event flag. | E04-23 |
| `pad_mode` that needs float lanes under an integer format | Configuration error. | E04-19 |
| R_CYCLIC alignment check | Fault F-4. | E04-19 |
| F-6 / F-7 | F-6 covers addresses beyond the XMEM address space. F-7 is never raised until tables exist. | E04-19 |
| Matmul benchmark kernel | `matmul_720x240_x128`, the largest standard case. Overridden for E04-02: it uses `matmul_384x192_x128`, because 720x240 takes about 22 s a run and `bench_emulator -- --quick` must finish in 30 s. E01-14's `kernel/*` keeps 720x240. | E04-02, E01-14 |
| Where the element codec lives | `ipu_isa.formats`, so the harness packs data without importing the emulator. | E04-13 |
| JSON schema validation | `jsonschema` as a test-only dependency. | E04-29 |
| Conformance data | `src/tools/ipu-isa/conformance/` (programs plus expected traces). | E04-31 |
| Unknown `IPU_EMU_PROTOTYPING` value | Raises `ValueError`. | E00-04 |

## Kernels and harness

| Question | Default | Raised in |
|---|---|---|
| LR presets | `[abi]` accepts LR targets (`i = "LR3"`) as well as CR targets. | E06-06 |
| `kernel.toml` readers and loader scope | `ipu_harness.manifest.load_manifest` is the reader; the one other is `ipu-as assemble --abi`, which reads only the `[abi]` table (E05-13; `ipu_as` cannot import `ipu_harness`), and a test allows exactly these two. The loader checks names, domains and paths; E06-03 evaluates shapes; E06-10 checks tolerance tier names and adds the per-kernel table form (the tier spellings differ: `exact`/`accum`/... here, `EXACT`/`FP8_ULP` in E06-10 and the epic; E06-10 settles them); E06-08 adds an input's `range` key; E06-11 adds `[benchmark]`. A registry entry's `params` are fixed values (the param's only choice, and its value in every template case) and its `formats` a subset; errors name the entry's source. `BYTE_UNIT_KERNELS` is keyed `family/name`. | E06-02 |
| `plan_case` details | `PlanError` is an `ipu_emu.errors.ConfigurationError` (Q59), and so is `ManifestError`. `plan_case(..., case=CaseDecl)` takes the case itself, so a sweep's expanded cases pass their base. `CasePlan.derived` holds every `[derived]` value; a value that does not evaluate is an error, never skipped. The shape grammar (`ipu_harness.expressions`, checked at load and evaluated per case) adds `/` (exact; whole result or inside `ceil`) and unary minus to the spec's list; `ceil` takes 1 argument, `min`/`max` at least 2. `check_params` returns the complete params and the formats whose element scalars accept them (E06-16). Exactness of element scalars uses the emulator's public FP8 codec until E04-13. The "later stages take only a `CasePlan`" test lives in E06-04, E06-05, E06-07 and E06-11; a source scan keeps the checks out of every module but the entry points. | E06-03 |
| Region placement details | `place(plan, row_counts=None)`: a tensor packing (E06-05) gives no count for takes `default_rows` (one row per 128 elements of the last dimension per index of the others, times the accumulator stride for a `store = "accumulator"` output); a given count is the rows as stored, stride included. `row_counts` naming no tensor, or below 1, is a `ValueError` (a caller bug, not configuration). `Placement` also carries `capacity` (`ipu_isa.formats.addressable_rows`). Placement only reserves rows; the runner writes a scratch region's `fill` (E06-11). The `projections` layout guard uses today's XMEM shapes with `default_rows`; E06-05/E07-02 re-run it with packing's counts. | E06-04 |
| `isa.json` number and slot-fill data | `numberPattern` is `ipu_as.ipu_token.NUMBER_PATTERN`, a hand-written regex (in syntax Python `re` and JS `RegExp` read alike) held to `int(v, 0)` by `test_isa_data` on every string of up to 4 characters over the literal alphabet, 20000 seeded longer ones and every corpus token; "derived" means that test. `slotFill` adds `kinds` (mnemonic -> slot kind: the slot of its real form in the generator's view, a pseudo-instruction's the slot it expands into; the same slot `CompoundInst` fills) to the spec's shape. `slotOrder` stays beside `slotFill.order` so version-0 readers keep working. The editor applies `slotFill` only for `rule: "own-kind"`; any other rule, or no `schemaVersion`, keeps the version-0 rules. Operand bounds stay JSON numbers (exact while ranges stay within 2^53). | E08-21 |
| Extension agreement and waits | Both directions are checked by `test/verdicts.js` (the editor's verdict, `test/verdict.js`, on the corpus, `mangle.mutations` of it and `test/corpus/negative/`) and `//vscode-ipu-asm:check_verdicts`, which judges each text with `ipu_as.diagnostics.check` (what `ipu-as check --json` prints) and fails on a disagreement not in `test/corpus/negative/allowed-gaps.json`, or a listed gap that no longer occurs; `agreement.js` stays the TextMate tokenizer test. The negative corpus also enters the `check --json` and CLI goldens (6 entries added, none changed) and stays out of the asm digests (`asm_fixtures`, `asm_corpus.NOT_PROGRAMS`). The fixed sleeps become signals: `ipuAsm.check` resolves when its check is done, runs resolve once sent, the kernel API's `terminalIdle` is the idle test a run uses, `kernels.ready` for the checkout scan, and "no late empty line" asks the on-type formatter for its edits at the cursor (`vscode.executeFormatOnTypeProvider`). | E08-27 |
| Derived values | A `[derived]` table of arithmetic expressions over params. A `module:fn` hook is the escape hatch. | E06-06 |
| Tolerance tiers | `exact`, `accum`, `transcendental`, `fp8_e<n>`. The values are set by the projections pilot (E07-03). | E06-10 |
| Benchmarks | `[benchmark]` table in `kernel.toml`, run by the runner. `benchmark.py` files are deleted. | E06-11 |
| Reading `registry.toml` from Bazel | A generated `registry.bzl`, kept fresh by a staleness test. | E06-17 |
| MobileViT packed programs | Spread across families by operation. | E07-23 |

## Tooling, CI and docs

| Question | Default | Raised in |
|---|---|---|
| Ruff line length | 100 | E01-09 |
| Type checker | mypy (strict for `ipu_isa` and the reference API) | E01-11 |
| Benchmark history | A JSON artifact uploaded by the nightly job, plus a `benchmark-data` branch | E01-14 |
| Codecov | Dropped. The nightly coverage run uploads an lcov artifact. | E01-06 |
| Weekly `test_full_*` stacks | They run in the nightly heavy tier. | E01-06 |
| Release log | Committed. A master-only workflow refreshes it, and the docs build is offline (L29). | E01-07 |
| giscus IDs | Committed in `mkdocs.yml`. They are public. | E00-06 |
| Alias-profile targets | Moved to the heavy tier. Each default run profiles once per case. | E01-04 |
| URLs that change when pages move | `mkdocs-redirects` | E09-06 |
| Kernel pages | Under Reference | E09 |
| LaTeX PDF guide | Kept as is | E09 |
| CLI reference | Generated by a small in-house generator. No new mkdocs plugins beyond redirects. | E09-15 |
| Formatter | Normalises mnemonics and register names to upper case | E08-24 |
| Cycle Inspector fonts | System font stack (no external fonts on the docs site) | E08-17 |
| Inspector demos | The vector-tour demos are kept as curated conformance programs | E08-16 |
| SV generator in `ipu_gen` | `ipu_gen.sv_package` reads `ipu_isa` only (names from `SlotInfo.opcode_enum` and `OperandType.sv_typedef`). `OPERAND_ENUM_ORDER` pins the operand enums' declaration order, wire format, checked at import. No `--status-file` (E03-21 removed stamping). `test_sv_package_golden` and `update_sv_golden` keep their `ipu-as-py` labels; `test_gen_codegen` and `test_sv_package_compiles` move to `//src/tools/ipu-gen`. `//docs:verify_sv_package` is an alias to `ipu-gen verify-sv-package` (E08-15). | E08-13 |
| Docs pages and layout SVGs in `ipu_gen` | `ipu_gen.docs_pages` and `ipu_gen.layout_svg` (E08-14). The SVGs read `LAYOUT_ORDER`, `SLOTS` and `SLOT_UNIONS` (labels from `asm_opcode_class` and `token_class`); E05-03 moves them onto `WORD_LAYOUT`. `ipu_gen` imports `ipu_as` for `Inst.description()` until E05-02. The word diagram keeps its title `CompoundInst Layout` so the pages stay byte-identical; `operand-types.md` names `ipu_gen.docs_pages` as its generator (the one changed baseline line). `//src/tools/ipu-gen:gen_union_layout_svg` became `ipu-gen union-layout-svg`. | E08-14 |
| No `*_wrapper.py` shims | rules_python 0.40 has no `py_binary(main_module = ...)`, so a kept label is an `alias`: `//vscode-ipu-asm:gen_vscode` and `//docs:verify_sv_package` point at `//src/tools/ipu-gen:vscode_data` / `:verify_sv_package` (the `ipu-gen` CLI with its subcommand fixed in `args`), `//vscode-ipu-asm:dump_parser_tokens` at a `py_binary` in `ipu-as-py`; `//docs:mkdocs` is a `py_console_script_binary`. The LRD pair indices are `ipu_isa.registers.lrd_pair`, which the emulator and `ipu_gen.vscode_data.REGISTER_FACTS` both use. | E08-15 |
| CODEOWNERS | Written last, from the candidates in E09-20 | E09-20 |
