# Implementation decisions

The issue drafts left some questions open. To keep implementation moving, these defaults apply. Each one can be overridden in PR review; an override becomes a follow-up issue.

The user decided the program-level questions in the grilling session. They are recorded in [README.md](README.md#decision-record) and in `docs/adr/`.

## Repository layout and packaging

| Question | Default | Raised in |
|---|---|---|
| Where do kernels live? | Top-level `kernels/<family>/<name>/` (Bazel labels `//kernels/<family>:<name>`). The Python support code is `ipu_harness` under `src/tools/ipu-harness/`. Kernels need no Python package. | E06 epic, E06-01, E06-17 |
| Are mnemonic aliases kept after the rename? | No. All in-repo sources are renamed in the same change, and the old names are rejected with a hint that names the new mnemonic. | E03-19, E05-16 |
| Can an EMULATOR instruction sit inside a HARDWARE slot? | No. The schema check rejects it. `HALT` is a hardware instruction, so no case needs it. | E03-07 |
| Hardware binaries | `--target hardware` writes `<name>.hw.bin` with no header. | E05-15 |
| `check --json` schema files | `vscode-ipu-asm/schema/` | E05-08 |
| Template parameters | Declared in `kernel.toml [params]`. A standalone template may use an `ipu:params` header comment. | E05-09 |
| `--abi` input | A `kernel.toml` path. Only its `[abi]` table is read. | E05-13 |

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
| CODEOWNERS | Written last, from the candidates in E09-20 | E09-20 |
