# Benchmarks tracked nightly with stored history (no gate)

Part of [Epic E01: Foundation](00-epic.md) · ID E01-14 · Size M · Depends on: E01-06 · Behaviour change: none

## Goal

The numbers the user cares about (emulator cycles per second, assemble time, peak memory, import time, test-suite time) are measured the same way every night on master and kept as a history, so every refactor PR can say "faster" or "slower" with data and a slow drift is visible. Nothing fails because a number moved.

## Background

Verified at 694a8a8:

- Kernel benchmarks exist but measure the kernel, not the toolchain. `kernel_registry/benchmarking.py` runs a package's `benchmark.py` configs and writes cycle counts, stage occupancy and alias hits to a `results.md` beside it (7 files in the tree). They report IPU cycles, not host time or memory.
- Nothing measures host-side speed. Ad-hoc measurements from the architecture review (plan 02, plan 04): `IpuState()` about 0.27 s and about 550 MB RSS (XMEM allocates 512 MiB, xmem.py:37); about 58 µs per narrow cycle; assembling every `.asm` in the repository takes seconds, dominated by the parser.
- The full test suite took about 1 800 s wall on 4 cores (E01-04); nothing records that over time.
- E04-02 plans emulator-specific benchmarks (`IpuState` construction, matmul cycles/s, prototyping cycles/s) and depends on "E01 benchmark tracking" for the format and the store. This issue provides both and the cross-cutting measurements.

## Decisions applied

- Q64, Q71: benchmark targets; track the numbers (cycles/s, assemble time, peak memory); no gate.
- Q61: priorities are dev loop and test time, then emulator speed, then memory, then the assembler. The benchmark set follows that order.

## Implementation

1. Result format: one JSON list per run in the `customSmallerIsBetter` / `customBiggerIsBetter` shape that `benchmark-action/github-action-benchmark` reads:
   ```json
   [{"name": "assemble/all_kernels", "unit": "s", "value": 4.21, "extra": "files=137 words=..."}]
   ```
   Two files per run: `bench-smaller.json` (times, memory) and `bench-bigger.json` (rates).
2. `//tools/bench:bench`, a `py_binary` that runs registered benchmark modules and writes the two files. A benchmark module exposes `def run(quick: bool) -> list[Result]`. Each measurement runs in a fresh subprocess so imports and memory are not shared.
3. First measurements (this issue):
   - `import/ipu_as`, `import/ipu_emu`, `import/ipu_apps`: wall time of `python -c "import X"` in a fresh interpreter, best of 5 (s).
   - `assemble/all_kernels`: assemble every kernel `.asm` once with the library API (s), and `assemble/words_per_s` (bigger is better).
   - `kernel/<k>/cycles_per_s` and `kernel/<k>/peak_rss_mb` for two fixed kernel cases: one small (`identity` default) and one large (a matmul or projection default case; the same kernel E04-02 picks, so the numbers line up). Peak RSS from `resource.getrusage(RUSAGE_SELF).ru_maxrss` in the subprocess.
   - E04-02 adds the emulator entries (`IpuState` construction, prototyping cycles/s) as another module.
4. Nightly job `benchmarks` in `nightly.yml` (E01-06), separate from the test job so test load does not disturb timings: bootstrap, `bazel run -c opt //tools/bench:bench -- --out $RUNNER_TEMP`, then `github-action-benchmark` (pinned by SHA) twice (smaller and bigger) with `auto-push: true` to a `benchmark-data` branch, `fail-on-alert: false`, `comment-on-alert: false`, `alert-threshold: "120%"` (alerts only shown in the job summary).
5. Record the full-suite wall time from the nightly `full` job as `ci/full_suite_wall_s`, and the default-tier wall time of the latest master CI run (`gh run list --workflow ci.yml --branch master --limit 1 --json ...`) as `ci/default_tier_wall_s`, so the Q71 5-10 minute target has a history.
6. `bazel run //tools/bench:bench -- --quick` for local use; the conventions page (E01-12) says a PR that claims a speed change pastes before/after numbers from it.
7. A smoke test `//tools/bench:bench_smoke_test` runs every benchmark module with `quick=True` and a tiny budget, so the modules cannot rot.

## Tests

- [ ] `bazel run //tools/bench:bench -- --quick --out /tmp/b` finishes in under 60 s and writes two valid files.
- [ ] `bench_smoke_test` passes in the default tier.
- [ ] A `workflow_dispatch` run of the nightly workflow pushes a data point to `benchmark-data`; a second run appends a second point.

## Acceptance Criteria

- [ ] Cycles/s, assemble time, peak memory, import time and CI wall time are recorded nightly with history.
- [ ] No workflow fails because a number changed.
- [ ] E04-02 can add its measurements as one module without changing the runner.

Open question: where should the history live? This issue proposes a `benchmark-data` branch written by `github-action-benchmark` (it also renders a chart page there). Alternatives: a JSON file committed to master by the nightly job, or workflow artifacts only (90-day limit). E09 can surface the history in the docs site later.

## Risks

- Shared GitHub runners have noisy timings (often ±10 %). Report best of several runs, keep the threshold advisory, and read trends rather than single points.
