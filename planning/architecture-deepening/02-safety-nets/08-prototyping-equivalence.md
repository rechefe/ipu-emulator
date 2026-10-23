# Keep and extend prototyping-vs-faithful equivalence

Part of [Epic E02: Safety nets before the refactors](00-epic.md) · ID E02-08 · Size S · Depends on: E02-05, E02-06, E02-07 · Behaviour change: none

## Goal

Prototyping mode stays (D7), and E04 turns it into an adapter on a single cycle module (E04-07, E04-08). Before that, make the equivalence evidence cover more than three hand-written programs: every timing probe from E02-05/E02-06 in lockstep, and every kernel case at output level. Record the one known divergence as an explicit expected failure, so E04 removes it on purpose.

## Background

- `test_prototyping_equivalence.py` runs `PROGRAMS` (3 programs, `test/test_prototyping_equivalence.py:36-85`) through faithful and prototyping engines in lockstep and compares every register, XMEM, snapshots, cycle outcomes and stats after every cycle (`:176-190`). It also covers activation fast paths, cached scalar reads, LR conflicts and plan invalidation (`:228-359`). It only runs in a `--define=ipu_proto=1` build.
- In a flagged build, prototyping is the default mode (`docs/content/prototyping-mode.md`, "Enabling it"), so `bazel test --define=ipu_proto=1 //...` runs every kernel test in prototyping mode.
- CI runs only the default build (`.github/workflows/ci.yml`, "Run all tests"). Nothing runs the flagged build regularly.
- Known divergence: prototyping executes LR sub-slots one at a time (`ipu_emu/ipu.py:2244-2245`), faithful resolves all three first (`ipu.py:1186-1206`). If a sub-slot read raises, prototyping has already written the earlier sub-slot. Every LR-slot read is cycle-start today, so no program observes it.

## Implementation

1. Move the E02-05 and E02-06 probe programs into a shared module (`test/timing_probes.py`) as data. Both the timing tests and the lockstep test import it, and `test_prototyping_equivalence.py` parametrizes over it in addition to `PROGRAMS`.
2. Add `test_lr_subslot_exception_partial_state_matches` to the lockstep test, marked `xfail(strict=True, reason="E04-07 closes the LR sub-slot divergence")`. When E04-07 lands, the strict xfail turns into a failure and the PR removes the marker.
3. Make sure the E02-07 baseline check runs unchanged in a flagged build, so each kernel case's `output_sha256`, `state_sha256` and `cycles` must match the faithful baseline in prototyping mode. Fields that legitimately differ between modes (none are expected; `stats` counters are not in the record) are listed in `baseline.py`, not skipped silently.
4. Tag the flagged run for E01's nightly heavy tier: document `bazel test --define=ipu_proto=1 //...` as the nightly command in the E01 CI workflow (E01 owns the job; this issue provides the command and the expectation that it is green).

*As built:* the nightly `proto` job runs `bazel test --config=full --config=proto --test_size_filters=enormous //...`, the heavy tier only. The default tier already runs with the flag on every pull request (ci.yml `proto` job), so running it again nightly would repeat that.

## Tests

- [ ] `bazel test --define=ipu_proto=1 //src/tools/ipu-emu-py:test_prototyping_equivalence` passes, with the timing probes included and one strict xfail.
- [ ] `bazel test --define=ipu_proto=1 //src/tools/ipu-apps/...` passes against the E02-07 baselines recorded in the default build.
- [ ] Default build: `bazel test //...` unchanged.

## Acceptance Criteria

- [ ] Every timing probe runs in lockstep in both modes.
- [ ] Every kernel case is proven output-identical between modes by the same baseline file.
- [ ] The LR sub-slot divergence is an explicit strict xfail that names E04-07.
- [ ] The nightly tier runs the flagged build.

## Risks

- A kernel that differs between modes today would block this issue. If one is found, record it as a strict xfail with the case id and open an E04 issue; do not regenerate the baseline in prototyping mode.
