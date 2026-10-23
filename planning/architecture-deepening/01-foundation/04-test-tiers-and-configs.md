# Test tiers: `enormous` leaves the default run; `--config=ci` and `--config=full`

Part of [Epic E01: Foundation](00-epic.md) · ID E01-04 · Size S · Depends on: E00-09 · Behaviour change: yes (`bazel test //...` no longer runs `enormous` tests; `--config=full` does)

## Goal

`bazel test //...` runs the default tier: everything a change must pass before review, in minutes. The heavy tier (size `enormous`) runs with `--config=full` and every night (E01-06). CI and developers use the same named configs, so "what CI runs" is one line in `.bazelrc`.

## Background

Measured at 694a8a8 (full `bazel test //...`, 4 cores, cold cache): 221 test targets, about 5 750 test-seconds, 1 800 s wall time.

- The 90 `test_alias_profile_kernels_<kernel>` targets (`ipu-apps/BUILD.bazel:113`, macro `ipu_per_kernel_test`, `ipu_app.bzl:194-213`) take about 3 420 test-seconds, 60 % of the total. Each one reruns every case of a kernel twice (`test_alias_profile_kernels.py:24-28`: a baseline run and a profiled run), so it repeats work the kernel's own target already does, and checks profiler consistency on top. The slowest are `proj_qkv_240_p4` (345 s), `proj_qkv_144_p4` (258 s) and `proj_ffn1_240_p4` (231 s).
- Other long targets: `test_seam_proj_qkv_to_scores_km_repack` (217 s), `attention_test` (147 s), `test_seam_qkv_matmul_scores_km_64x48` (106 s), `proj_qkv_240_p4` (96 s).
- Today only `test_debug_launch` (`ipu-apps/BUILD.bazel:100-105`) and the `manual` `test_full_*` mobilevit stacks (`ipu_app.bzl:179-183`) are `enormous`. E00-09 makes `test_debug_launch` small, so after it `enormous` means "heavy tier" only.
- `.bazelrc` has no `ci` or `full` config. `ci.yml:35` passes `--build_tag_filters=-docs` to `bazel build` only, so `bazel test //...` (ci.yml:38) still builds `//docs:build_docs` (tagged `docs`, `no-sandbox`) and the networked release-log genrule.

## Decisions applied

- Q73: the `enormous` tier is out of the default run; it runs nightly and with `--config=full`.
- Q71: CI target 5-10 minutes; nightly pipelines for the heavy tier.

## Implementation

1. `.bazelrc`:
   ```
   # Default tier: everything but the heavy (enormous) tests.
   test --test_size_filters=-enormous

   # CI: default tier, no docs targets.
   build:ci --build_tag_filters=-docs
   test:ci --test_tag_filters=-docs
   test:ci --build_tests_only

   # Full: every non-manual test, including the heavy tier.
   test:full --test_size_filters=
   build:full --build_tag_filters=-docs
   test:full --test_tag_filters=-docs
   ```
   Verify that an empty `--test_size_filters=` clears the default filter on Bazel 8.3.0; if it does not, list all sizes explicitly (`small,medium,large,enormous`).
2. Move `test_alias_profile_kernels` to the heavy tier: pass `size = "enormous"` at `ipu-apps/BUILD.bazel:113`.
3. Add a "Test tiers" section to `README.md` and to the code-conventions page (E01-12): what each tier holds, how to run it (`bazel test //...`, `bazel test --config=full //...`), and the rule for putting a new test in the heavy tier (it repeats coverage another default-tier test gives, or it takes more than about 2 minutes on CI).
4. `CLAUDE.md` Build & Test block: add `bazel test --config=full //...  # include the heavy (nightly) tier`.
5. `.claude/hooks/session-start.sh:64`: warm with `bazel build --config=ci //...`.

*As built:* the session-start hook warms with `bazel fetch --config=ci //...` rather than `bazel build`, to stay inside the session's disk allowance (`.claude/hooks/session-start.sh`). The tests still build on first use.

## Tests

- [ ] `bazel test //... --nobuild` lists no `test_alias_profile_kernels_*` and no `test_full_*`; `bazel test --config=full //... --nobuild` lists the alias-profile targets and still no `manual` target.
- [ ] `bazel test --config=ci //...` builds no `docs`-tagged target (`--explain` or the build event log).
- [ ] Record wall time of `bazel test --config=ci //...` on a cold cache before and after in the PR.

## Acceptance Criteria

- [ ] The default tier excludes `enormous` tests; `--config=full` includes them.
- [ ] CI, the hook and the docs use the named configs instead of repeating flags.
- [ ] The PR is labelled as a behaviour change (test selection).

Open question: the alias-profile targets are the proposal for the heavy tier because they are 60 % of test time and repeat each kernel's cases. Confirm, or name other targets (for example the `test_seam_*` suites) that should move too.

## Risks

- A regression that only the heavy tier catches lands on master and is found the next night. The nightly workflow (E01-06) reports failures; the review guideline (E01-13) asks authors to run `--config=full` locally.
