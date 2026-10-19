# Epic E02: Safety nets before the refactors

ID E02 · Folder `02-safety-nets` · Runs after E00 and E01, before E03-E07 · Every later refactor issue cites these harnesses as its proof

## Goal

Pin today's observable behaviour in fast, reviewable Bazel tests, before any structural refactor starts. Decision D3 says refactor PRs must be output-identical and behaviour changes go in separate, labelled PRs. That rule needs a mechanical proof. After this epic, a refactor PR proves it is output-identical by leaving every baseline file untouched, and a behaviour PR shows its effect as a diff of one baseline file.

The harnesses cover the five outputs that later epics touch:

| Output | Touched by | Harness |
|---|---|---|
| Assembled binaries (`.bin`, `.mem`) of every `.asm` | E03 rename, E05 codec/session, E07 templates | E02-01 |
| Decoded-word field names, disassembly text, enum descriptors | E03 codec, E05 field-name table, E08 debugger | E02-02 |
| Generated SV package and its `*_WIDTH` localparams | E03 registries, tags, rename; E08 `ipu_gen` | E02-03 |
| Other generated artifacts (`isa.json`, TextMate grammar, generated docs pages) | E03, E05, E08, E09 | E02-04 |
| Register timing (which value a read sees within one cycle) | E04 cycle module, view, timing flips | E02-05, E02-06 |
| Kernel case results and the CR/LR state each harness programs | E04, E06, E07 | E02-07 |
| Prototyping mode equals faithful mode | E04 single cycle, plan cache | E02-08 |

## Background

Verified at HEAD 694a8a8.

- There are 104 `.asm` files outside `bazel-*`: 89 kernels under `src/tools/ipu-apps/src/ipu_apps/kernels/`, 12 MobileViT programs under `src/tools/ipu-apps/test/mobilevit/`, and 3 VS Code fixtures under `vscode-ipu-asm/test/`. No test compares any assembled output with a recorded value. `assemble_<kernel>` genrules exist (`ipu_app.bzl:38-41`), but only `bazel build //...` exercises them.
- Some programs are only correct with a render context: `conv_universal_wide384` and `depthwise_conv_stride2_*` render per run through `kernels/convolutions/app.py` `render_asm`, and the four `asm_packed_output_linear_*` programs need `chunk_widths` (`test/mobilevit/test_full_layer_l4_packed.py:281`).
- Decoded-word field names come from Python class names (`ipu_as/compound_inst.py:145-178`). The emulator rebuilds the same strings with its own tables (`ipu_emu/ipu.py:213-244`). Nothing pins them.
- The SV package has only structural tests (`ipu-as-py/test/test_gen_codegen.py`) and a slang elaboration test (`test_sv_package_compiles.py`). The localparam values are not pinned, although CLAUDE.md requires every `*_WIDTH` to stay identical unless a PR calls out the change. Its header stamps the commit hash (`docs/BUILD.bazel:44-51`), so a byte compare against the generated file fails on every commit.
- The register-timing behaviour each handler implements is pinned only in places (`test_execute.py:1559` `test_agg_and_activate_quantize_same_cycle_sees_live_r_acc`, `:1886` `test_reshape_mult_res_reads_from_snapshot`). Plan 01's forwarding table lists 14 consumer/register rows; most have no direct test. Decision Q32 changes 4 of them in E04.
- Kernel case checks compare against references with tolerances (`kernel_registry/case_support.py:47` `check_close`, `rtol=1e-4`). A refactor that changes an output inside the tolerance passes silently.
- `test_prototyping_equivalence.py` runs 3 programs (plus wide variants) in lockstep in a `--define=ipu_proto=1` build. No kernel runs in that mode in CI (`.github/workflows/ci.yml` runs `bazel test //...` only).

## Decisions applied

| Decision | Applied as |
|---|---|
| D3 | Every harness fails on any change to its baseline. A refactor PR must not touch a baseline file. A behaviour PR regenerates only the baseline it changes and says why (E02-09). |
| D6 | The SV golden (E02-03) pins every `*_WIDTH` localparam by name, so a wire-format change is visible in the test failure. |
| D7 | Prototyping mode stays. E02-08 keeps the lockstep test and adds kernel-level evidence that the two modes produce identical results. |
| Q32 | E02-05 and E02-06 pin today's timing, including the 4 rows Q32 changes. Each changed row is one named test, so each E04 behaviour PR (E04-25, E04-26, E04-27) flips only its own tests. |
| Q70 | Kernel baselines (E02-07) live in each kernel's folder, next to its `test.py`. |
| Q71 | Harnesses add no new emulator runs to the default tier: E02-07 hooks into the kernel tests that already run. Assembly-level harnesses are sharded to stay inside the 5-10 minute CI budget (E01). |
| Q73 | Lockstep runs of real kernels (E02-08) go to the nightly heavy tier, not the default tier. |

## Scope

- Baseline digests of every assembled program, including programs that need a render context.
- Snapshots of decoded field names, disassembly text and enum descriptors.
- SV package golden with an explicit localparam table, plus a script that diffs localparams against a base commit.
- Snapshots of the other generated artifacts.
- Timing characterization tests for every row of the plan 01 forwarding table.
- Per-case kernel digests (binary run, CR/LR state after setup, output, cycle count).
- Prototyping-vs-faithful equivalence at kernel level.
- One update command and a written review rule for baselines.

## Out of scope

- Changing any behaviour. Every issue here is "Behaviour change: none".
- Tolerances and references (E06). E02 pins today's bits, not correctness.
- The docs asm-block test (E09-01).
- CI job layout and the nightly tier itself (E01). E02 adds targets and tags; E01 decides where they run.

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|---|---|---|---|---|---|
| E02-01 | [01-asm-binary-digests.md](01-asm-binary-digests.md) | Baseline digests of every assembled program | S | E01 (toolchain pin) | none |
| E02-02 | [02-decode-snapshots.md](02-decode-snapshots.md) | Snapshot decoded field names, disassembly text and enum descriptors | S | E02-01 | none |
| E02-03 | [03-sv-package-golden.md](03-sv-package-golden.md) | SV package golden with a pinned localparam table | S | E01 (toolchain pin) | none |
| E02-04 | [04-generated-artifact-snapshots.md](04-generated-artifact-snapshots.md) | Snapshot `isa.json`, the TextMate grammar and the generated docs pages | S | E01 (toolchain pin) | none |
| E02-05 | [05-timing-lr-views.md](05-timing-lr-views.md) | Timing characterization: which LR value each consumer reads | S | E01 | none |
| E02-06 | [06-timing-datapath-views.md](06-timing-datapath-views.md) | Timing characterization: datapath registers and end-of-word control | S | E01 | none |
| E02-07 | [07-kernel-case-digests.md](07-kernel-case-digests.md) | Per-case kernel digests: binary, CR/LR after setup, output, cycles | M | E02-01 | none |
| E02-08 | [08-prototyping-equivalence.md](08-prototyping-equivalence.md) | Keep and extend prototyping-vs-faithful equivalence | S | E02-05, E02-06, E02-07 | none |
| E02-09 | [09-baseline-workflow.md](09-baseline-workflow.md) | One update command and a review rule for baselines | S | E02-01 to E02-08, E01 (code-conventions page) | none |

## Suggested order

```
E01 ──┬──▶ 01 ──┬──▶ 02 ─────────────────┐
      │         └──▶ 07 ──┐              │
      ├──▶ 03             │              │
      ├──▶ 04             ├──▶ 08 ──▶ 09 ◀┘   (09 also needs 03, 04)
      ├──▶ 05 ────────────┤
      └──▶ 06 ────────────┘
```

E02-01, E02-03, E02-04, E02-05 and E02-06 can be worked in parallel. E03 may start once E02-01 to E02-04 are merged. E04 may start once all of E02 is merged. E09 collects the update commands in one place.

## Cross-epic assumptions

- E01 pins the toolchain and lockfile, so digests are stable across machines.
- E01 provides a nightly heavy tier (Q73) and a code-conventions page that E02-09 links into.
- E03, E04, E05, E06 and E07 cite E02 harnesses as the proof for every refactor issue, and update a baseline only in an issue marked "Behaviour change: yes".
- E03's mnemonic rename (Q45) changes `.asm` text but not binaries. E02-01 keys its manifest by path and digest, so the rename leaves the binary digests unchanged. E02-02's disassembly snapshot changes in that PR; that is the intended, reviewed diff.
