# Emulator benchmarks for the Q71 targets

Part of [Epic E04: Emulator core](00-epic.md) · ID E04-02 · Size S · Depends on: E01 (benchmark tracking), E04-01 · Behaviour change: none

## Goal

Measure the emulator before the refactors start, so every later E04 PR can state its effect on speed and memory. Q71 sets three tracked targets (tracked, not gated):

| Target | Value |
|---|---|
| `IpuState()` construction | < 5 ms and < 5 MB RSS delta |
| Matmul benchmark kernel | >= 50k cycles/s |
| Emulator unit tests | < 30 s wall time |

## Background

- Plan 02 reports `IpuState()` at about 0.27 s and 550 MB RSS, because `XMem.__init__` allocates `bytearray(1 << 29)` (xmem.py:37). It reports the emulator suite at about 57.6 s.
- Plan 02 reports about 58 µs per narrow cycle, dominated by per-lane Python loops in MULT and ACC handlers.
- E01 provides the benchmark tracking mechanism (Q64, Q71). This issue adds the emulator's entries to it.

## Implementation

1. Add `src/tools/ipu-emu-py/bench/bench_emulator.py` with three measurements:
   - `ipu_state_construct`: `timeit(IpuState, number=20)`, plus `tracemalloc` peak and `resource.getrusage().ru_maxrss` delta.
   - `matmul_cycles_per_second`: run a fixed matmul kernel (one of the existing projection or matmul kernels, assembled once at setup) to `HALT` with `run_until_complete`; report best of 5 cycles/s. Run it for INT8 and for the format the kernel uses today (wide FP32). Add E4 once E04-17 lands.
   - `prototyping_cycles_per_second`: the same kernel with `--define=ipu_proto=1`.
2. Register the script as a Bazel `py_binary` target `//src/tools/ipu-emu-py:bench_emulator` that writes JSON in the format E01 tracks.
3. Record the suite wall time with `bazel test //src/tools/ipu-emu-py/... --nocache_test_results` in the same report.
4. Add a short "Performance" section to the PR template guidance used by E04 issues: E04-05, E04-08, E04-15 and E04-16 paste before/after numbers.

## Tests

- [ ] `bazel run //src/tools/ipu-emu-py:bench_emulator -- --quick` runs in under 30 s and emits valid JSON.
- [ ] A smoke test runs the benchmark with a tiny cycle budget so the script cannot rot.

## Acceptance Criteria

- [ ] Baseline numbers for all three targets are recorded in the PR and in E01's tracking store.
- [ ] The benchmark is not part of the default `bazel test //...` run (it is a `bazel run` target plus a smoke test).

## Open question

- Which existing kernel is "the matmul benchmark"? Proposal: the largest projection matmul that runs in under 10 s today. The user picks.
