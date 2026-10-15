# Nightly workflow: the heavy tier, with coverage, in one pass

Part of [Epic E01: Foundation](00-epic.md) · ID E01-06 · Size M · Depends on: E01-04, E01-05 · Behaviour change: none (code); coverage moves from every push to nightly and starts producing real data

## Goal

Every night, master runs every non-manual test (`--config=full`) once, under coverage instrumentation, from a clean cache. A failure is reported where people see it. Coverage numbers are real, not an empty file uploaded with `|| true`.

## Background

Verified at 694a8a8:

- There is no scheduled workflow. The heavy tier defined in E01-04 (size `enormous`: the 90 alias-profile targets, the per-kernel debugger sweep from E00-09) would otherwise never run in CI.
- Coverage today runs on every push (`ci.yml:40-66`), but `MODULE.bazel:10-17` does not set `configure_coverage_tool = True` on the Python toolchain, so `bazel coverage` has no Python coverage tool. The report copy and HTML steps end in `|| true` (ci.yml:44, 51), and Codecov runs with `fail_ci_if_error: false`.
- `.bazelrc:15-18` holds the coverage flags; E00-10 removes the LLVM-only one.
- Measured cold full run (E01-04 background): about 1 800 s wall on 4 cores. Coverage instrumentation adds overhead on top; the job needs a timeout well under GitHub's 6-hour limit.
- The `manual` mobilevit stacks (`test_full_*`, `ipu_app.bzl:179-183`) take hours each and stay out of `//...` even with `--config=full`.

## Decisions applied

- Q71: nightly pipelines for the heavy tier; no double run.
- Q73: `enormous` runs nightly and with `--config=full`.

## Implementation

1. `MODULE.bazel`: `python.toolchain(..., configure_coverage_tool = True)`.
2. `.bazelrc`:
   ```
   coverage --combined_report=lcov
   coverage --instrument_test_targets=false
   coverage --instrumentation_filter=^//src/tools[/:],^//docs[/:],^//tools[/:]
   ```
   (Instrument libraries, not test files.)
3. New `.github/workflows/nightly.yml`:
   ```yaml
   on:
     schedule:
       - cron: "17 2 * * *"      # 02:17 UTC, off the hour
     workflow_dispatch:
   concurrency:
     group: nightly
     cancel-in-progress: false
   permissions:
     contents: read
     issues: write
   ```
   Job `full` (`timeout-minutes: 240`): checkout master, `./tools/bootstrap.sh`, repository cache only (no disk cache, so tests really run and non-hermetic tests surface), then one command:
   ```bash
   bazel coverage --config=full //...
   ```
   This runs every non-manual test once and writes the combined lcov report (`bazel-out/_coverage/_coverage_report.dat`).
4. Publish coverage: `lcov --summary` into the job summary (`$GITHUB_STEP_SUMMARY`), and upload the `.dat` file and a `genhtml` report as an artifact (30-day retention). No step ends in `|| true`; an empty or missing report fails the job.
5. Failure reporting: on failure, create or update one open issue titled "Nightly failure" (label `nightly`) with the run link and the list of failed targets (`gh issue` with the job's token). On success, close it if open.
6. Leave Codecov out. Open question: keep uploading to Codecov from the nightly job, or rely on the artifact and job summary?
7. The benchmarks job (E01-14) is added to this workflow by that issue.

## Tests

- [ ] `workflow_dispatch` run on the PR branch completes; the job summary shows a non-zero line coverage for `ipu_emu`, `ipu_as` and `ipu_common`.
- [ ] The run includes the `test_alias_profile_kernels_*` and `test_debug_launch_kernels_*` targets (visible in the test summary).
- [ ] A dispatch run with a deliberately failing heavy-tier test opens the "Nightly failure" issue; the next green run closes it.

## Acceptance Criteria

- [ ] One nightly run executes every non-manual test exactly once.
- [ ] Coverage is collected only in the nightly run and is non-empty.
- [ ] A nightly failure is visible as an open issue until fixed.

Open question: should the `manual` `test_full_*` mobilevit stacks run on a weekly schedule? They take hours each; this issue leaves them manual.

## Risks

- Coverage instrumentation can change timing-sensitive tests (none known; the emulator has no wall-clock behaviour). If a test only fails under coverage, give it a `no-coverage` tag, exclude that tag in the coverage run (`--test_tag_filters=-docs,-no-coverage`), and run it in a plain `bazel test --config=full` step instead, naming it in the PR.
