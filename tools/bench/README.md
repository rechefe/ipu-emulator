# Benchmarks

Host-side numbers for the toolchain: how fast it imports, assembles and
emulates, and how much memory it takes (E01-14, decisions Q61, Q64, Q71). They
are tracked nightly as a history. No test or workflow fails because a number
moved; read trends, not single points (shared runners vary by about 10 %).

## Run locally

```bash
bazel run //tools/bench:bench -- --quick                  # about 30 s; files in bench-results/
bazel run //tools/bench:bench -- --quick --only kernels   # one module
bazel run -c opt //tools/bench:bench -- --out /tmp/bench  # what the nightly job runs
```

A PR that claims a speed change pastes the `--quick` table from before and
after the change (see the code-conventions page).

## What is measured

| Name | Unit | Better | How |
|---|---|---|---|
| `import/ipu_as`, `import/ipu_emu`, `import/ipu_apps` | s | lower | `python -c "import <module>"` in a fresh interpreter, startup included, best of 5 (2 with `--quick`). |
| `assemble/all_kernels` | s | lower | Every kernel `.asm` assembled once with `ipu_as.lark_tree.assemble`, parser construction included. |
| `assemble/words_per_s` | words/s | higher | Instruction words of that run per second. |
| `assemble/render_s`, `assemble/parse_s`, `assemble/encode_s` | s | lower | Every program in the assembled-binary digest manifest, split by pipeline stage, best of 3 (1 with `--quick`); parser built and tables warm first. Also `bazel run //src/tools/ipu-as-py:bench_assemble`. |
| `kernel/<k>/cycles_per_s` | cycles/s | higher | The `default` case of `identity` (small) and `matmul_720x240_x128` (large), run and checked as the tests do; only the emulated run is timed. |
| `kernel/<k>/peak_rss_mb` | MB | lower | `ru_maxrss` of the process that ran the case. |
| `ci/full_suite_wall_s` | s | lower | Nightly only: the coverage step of the nightly `full` job. |
| `ci/default_tier_wall_s` | s | lower | Nightly only: the test step of the latest successful `ci.yml` run on master. |

Each measurement runs in a fresh interpreter (`ipu_bench.isolated`), so
imports and memory are not shared between measurements.

## Output

Two files in the `customSmallerIsBetter` and `customBiggerIsBetter` formats of
[github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark):

- `bench-smaller.json`: times and memory.
- `bench-bigger.json`: rates.

```json
[{"name": "assemble/all_kernels", "unit": "s", "value": 4.62, "extra": "files=89 words=4499"}]
```

## History

The `benchmarks` job of `.github/workflows/nightly.yml` runs after the `full`
job (so test load does not disturb the timings), uploads the two files as the
`benchmarks` artifact, and appends them to the `benchmark-data` branch, which
also holds the chart page (`dev/bench/index.html`). An alert above 120 % of
the previous value is shown in the job summary only.

## Add a benchmark

Add a module to `ipu_bench/benchmarks/` that defines
`run(quick: bool) -> list[Result]`. The runner finds it; nothing else changes.
Take each measurement in a fresh interpreter with
`ipu_bench.isolated.call("ipu_bench.benchmarks.<module>:<function>", **kwargs)`,
and measure as little as still exercises the code when
`ipu_bench.result.tiny_budget()` is true: `//tools/bench:bench_smoke_test`
runs every module that way in the default tier. A `Result` name is
`<area>/<measurement>` and stays stable: renaming one starts a new history.
