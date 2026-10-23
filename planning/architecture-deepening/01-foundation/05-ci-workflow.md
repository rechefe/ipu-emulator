# CI workflow: PR + master + dispatch, one test pass, Bazel cache

Part of [Epic E01: Foundation](00-epic.md) · ID E01-05 · Size M · Depends on: E01-01, E01-02, E01-04 · Behaviour change: none (code); CI runs once per change and no longer produces coverage on every push

## Goal

Every pull request and every push to master runs the default tier once, with a warm Bazel cache, and reports in 5-10 minutes once the emulator speed-ups of E04 land. Failures are visible; nothing hides behind `|| true`.

## Background

Verified at 694a8a8 (`.github/workflows/ci.yml`):

- Triggers: `push` to `**` and `pull_request` to `**` (ci.yml:3-7). A branch with an open PR runs the whole workflow twice per push.
- Steps: `bazel build //... --build_tag_filters=-docs` (ci.yml:34-35), then `bazel test //... --test_output=errors` (ci.yml:37-38), then `bazel coverage //...` (ci.yml:40-44). Coverage reruns every test under instrumentation, so the suite runs twice per workflow, four times per PR push.
- `MODULE.bazel:10-17` configures no Python coverage tool (`python.toolchain(configure_coverage_tool = True)` is absent), so the lcov report is likely empty. `cp ... || true` (ci.yml:44) and `genhtml ... || true` (ci.yml:51) hide that, and Codecov runs with `fail_ci_if_error: false` (ci.yml:61-66).
- No cache: every run fetches the Python toolchain, the wheels and rebuilds every action.
- `--test_output=errors` is already in `.bazelrc:21`; the flag in ci.yml:38 is a duplicate.
- Measured full-suite cost (E01-04): about 5 750 test-seconds on a cold cache; about 2 330 without the alias-profile targets. On a 4-vCPU `ubuntu-latest` runner that is about 10 minutes of test time for the default tier on a cold cache. Bazel caches test results by input digest, so with a disk cache a PR that touches one kernel reruns only that kernel's targets.
- `vscode-extension.yml:25-39` already runs the extension's JS tests on PRs that touch `ipu_as`, `ipu_common`, `ipu_emu`, kernel `.asm` files and the kernel registry, with JS kept out of Bazel (D9).

## Decisions applied

- Q71: CI 5-10 minutes; nightly heavy tier (E01-06); no double run.
- Q73: the default tier only; `--config=ci` (E01-04).
- D9: JS stays out of Bazel; the extension workflow's path triggers cover assembler and ISA changes.

## Implementation

1. Triggers and concurrency:
   ```yaml
   on:
     pull_request:
     push:
       branches: [master]
     workflow_dispatch:
   concurrency:
     group: ci-${{ github.ref }}
     cancel-in-progress: ${{ github.ref != 'refs/heads/master' }}
   ```
2. Steps: checkout, `./tools/bootstrap.sh`, restore cache, `bazel test --config=ci //...`, upload test logs on failure. Delete the separate build step (`bazel test` builds what it tests), the coverage, lcov, genhtml, artifact and Codecov steps (coverage moves to the nightly run, E01-06).
3. Cache: `actions/cache` (pinned by commit SHA) for the Bazel repository cache and a disk cache:
   ```
   build:ci --repository_cache=~/.cache/bazel-repo
   build:ci --disk_cache=~/.cache/bazel-disk
   ```
   Key: `bazel-${{ runner.os }}-${{ hashFiles('.bazelversion', 'MODULE.bazel', 'MODULE.bazel.lock', 'requirements_lock.txt') }}-${{ github.sha }}`, restore keys from the same prefix without the SHA, so each run starts from the newest compatible cache. Save on master pushes only, so PRs share master's cache and do not evict each other (GitHub's 10 GB per-repository limit). Trim the disk cache before saving (`find ~/.cache/bazel-disk -atime +7 -delete`).
4. On failure, upload `bazel-testlogs/**/test.log` and `test.xml` as an artifact (14-day retention).
5. Print the critical path and the slowest 20 targets at the end of every run (`--test_summary=detailed` is already set; add a step that sorts `bazel-testlogs/**/test.xml` durations), so CI time regressions are visible in the log.
6. `vscode-extension.yml`: keep its PR path filters; add `requirements_lock.txt` and `tools/bootstrap.sh` (both change the build it runs). Add a comment that E03's package rename (`ipu_common` to `ipu_isa`) and the Q69 package split must update these paths.
7. Make the CI job a required status check on master (repository setting; the user does this, note it in the PR).

## Tests

- [ ] A draft PR shows exactly one CI run per push.
- [ ] A second push with no source change finishes with every test "cached" and in under 3 minutes.
- [ ] A push that touches one kernel's `cases.py` reruns only that kernel's targets and suites that depend on `ipu_apps_lib` (expect most of `ipu-apps` until E06 splits the library; record the count).
- [ ] A deliberately failing test fails the job and uploads its log.

## Acceptance Criteria

- [ ] CI triggers on pull requests, master pushes and manual dispatch only.
- [ ] One `bazel test --config=ci //...` invocation per run; no coverage, no `|| true`.
- [ ] Cache hits are visible in the log on the second run.
- [ ] Cold-cache wall time is recorded in the PR. The 5-10 minute target is tracked in the benchmarks (E01-14); reaching it on a cold cache depends on E04's emulator speed-ups.

## Risks

- A branch without a PR gets no CI. `workflow_dispatch` covers it.
- A poisoned cache entry could hide a failure. Test results are keyed by input digests, so this needs a non-hermetic test. The nightly run (E01-06) uses no disk cache and catches it.
