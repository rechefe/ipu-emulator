# Build ETISS and the `IPU` plugin under Bazel; CI wiring

Part of the [ETISS integration epic](00-epic.md). Depends on #1.

## Goal

Make `bazel build //src/tools/ipu-etiss/...` produce ETISS, the `IPU`
architecture plugin, and a runner binary — locally and in CI — without any
manual `cmake` invocation, while keeping the existing Python-only targets
untouched.

## Implementation

- **ETISS as a Bazel dependency.** Add `rules_foreign_cc` and `rules_cc` to
  `MODULE.bazel`; fetch ETISS with `http_archive` at the pinned commit
  (`74451e0`) and build it with the `cmake()` rule (`ETISS_BUILD_DEFAULTSUB=ON`
  so the bundled TCC JIT and integrated plugins are included; RISC-V archs may
  be disabled for build time until issue 7). Export headers, `libETISS.so`,
  `include/jit/**` (needed by the JIT at run time) and `bare_etiss_processor`.
- **Boost.** v1: system packages (`libboost-{system,filesystem,program-options}-dev`)
  installed by CI (`.github/workflows/ci.yml`) and documented in `setup.md`.
  Evaluate BCR `boost.*` modules for a hermetic build and record the outcome.
- **Plugin.** New package `src/tools/ipu-etiss/` with:
  - `arch/` — `IPUArch.{h,cpp}`, `IPUArchLib.cpp`, `IPUFuncs.{c,h}`, generated
    files from #3, and a small `CMakeLists.txt` using `FIND_PACKAGE(ETISS)` +
    `ETISSPluginArch(IPU)`; built by a second `cmake()` target that depends on
    the ETISS target (this is the RV32IMACFD out-of-tree pattern and keeps the
    JIT header install step correct).
  - `runner/` — `ipu_etiss_run.cpp`: loads `libIPU.so` via `etiss::loadLibrary`,
    sets up `SimpleMemSystem` (IMEM at `0x0`, XMEM at `0x1000_0000`), applies
    `ipu.*` config keys, runs `CPUCore::execute`, and writes `xmem.bin`,
    `regs.json`, `stats.json`. Command line mirrors `run_test` inputs.
- **Bazel macro** `ipu_etiss_test(name, asm, …)` in `asm_rules.bzl` (next to
  `assemble_asm`) that assembles a program and runs it on the runner — used
  by #4/#5 tests.
- Keep `bazel test //...` working on machines without Boost: tag the ETISS
  targets `etiss` and skip them with `--build_tag_filters=-etiss` in a
  documented fallback, but CI builds them.

## Tests

- [ ] `bazel build //src/tools/ipu-etiss/...` succeeds from a clean cache.
- [ ] `bazel run //src/tools/ipu-etiss:ipu_etiss_run -- --help` prints usage.
- [ ] The #1 NOP/branch program runs through the runner as a `sh_test`.
- [ ] CI job builds the ETISS targets and runs the test above.

## Acceptance Criteria

- [ ] ETISS is pinned and fetched by Bazel; no submodules, no manual steps.
- [ ] Plugin and runner build under Bazel and load the plugin at run time.
- [ ] CI is green with the new targets included.
