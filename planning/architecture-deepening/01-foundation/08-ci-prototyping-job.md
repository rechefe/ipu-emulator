# CI: run the default tier in prototyping mode (`--define=ipu_proto=1`)

Part of [Epic E01: Foundation](00-epic.md) · ID E01-08 · Size S · Depends on: E01-05 · Behaviour change: none

## Goal

Every pull request also runs the default tier with the prototyping fast paths built in, so the bit-identity guarantee of prototyping mode is checked on every change instead of never.

## Background

Verified at 694a8a8:

- `BUILD.bazel:5-12` defines `config_setting(name = "prototyping_on", values = {"define": "ipu_proto=1"})`. With it, `ipu-emu-py` generates `_build_flags.py` with `PROTOTYPING = True`, and every `IpuState` defaults to the fast paths (`prototyping.py:22-29`).
- No workflow passes `--define=ipu_proto=1`. `CLAUDE.md` lists the command, but CI never runs it.
- In a default build, 19 test cases in `test_prototyping_equivalence.py` skip (`_REQUIRES_PROTO_BUILD`, test_prototyping_equivalence.py:28-30). They are the tests that prove prototyping mode is bit-identical to the faithful path (registers, XMEM, exceptions, snapshots). Today they never run in CI.
- With the flag, every kernel test also runs through the fast paths, which checks them against the kernels' goldens.

## Decisions applied

- D7: prototyping mode stays, so it must be tested.
- Q72: faster-without-changing-results paths are encouraged; prototyping is one of them and needs this check until E04-08 makes it an adapter on the single cycle module.
- Q71: no double run of the same configuration. This is a different build configuration, run as a parallel job.

## Implementation

1. `.bazelrc`:
   ```
   # Prototyping build: the emulator's opt-in fast paths.
   build:proto --define=ipu_proto=1
   ```
   Keep `--define=ipu_proto=1` working; `--config=proto` is the short form. (Plan 08 question 7 recommended keeping `--define` for now rather than migrating to a Starlark `bool_flag`.)
2. `ci.yml`: a second job `proto`, in parallel with the main job, with the same bootstrap and cache steps and:
   ```bash
   bazel test --config=ci --config=proto //...
   ```
   Give it its own cache key suffix (`-proto`), because every emulator action differs.
3. Check in the job that the prototyping tests ran: fail if `test_prototyping_equivalence` reports any skipped case (parse its `test.xml`).
4. `CLAUDE.md` and `docs/content/prototyping-mode.md`: mention `--config=proto` next to the `--define` form, and that CI runs the default tier in both modes.
5. Make the `proto` job a required status check (repository setting; note in the PR).

## Tests

- [ ] The `proto` job is green on the PR, and its log shows 0 skipped cases in `test_prototyping_equivalence`.
- [ ] A scratch commit that breaks a fast path (for example skipping a slot that is not a NOP) fails the `proto` job and not the main job.

## Acceptance Criteria

- [ ] Every PR runs the default tier with and without the prototyping build.
- [ ] The prototyping equivalence tests run in CI.

## Risks

- The job may expose existing failures under the flag. Triage them before making the job required; any fix is its own issue.
- It doubles CI compute (not wall time). If runner minutes become a concern, restrict the job to `//src/tools/ipu-emu-py/... //src/tools/ipu-apps/...`, which is nearly everything today.
