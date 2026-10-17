# One update command and a review rule for baselines

Part of [Epic E02: Safety nets before the refactors](00-epic.md) · ID E02-09 · Size S · Depends on: E02-01 to E02-08, E01 (code-conventions page, review guideline) · Behaviour change: none

## Goal

Make the D3 rule ("refactor PRs are output-identical; behaviour changes go in separate, labelled PRs") checkable by a reviewer in one look. There is one command that regenerates every baseline, one list of baseline files, and one written rule for when a PR may touch them.

## Background

E02-01 to E02-07 add these baselines and update targets:

| Baseline | Location | Update target |
|---|---|---|
| Assembled-program digests | `src/tools/ipu-apps/test/baselines/asm_digests.json` | `//src/tools/ipu-apps:update_asm_digests` |
| Field names, emulator field maps, enum descriptors | `src/tools/ipu-as-py/test/baselines/*.json` | `//src/tools/ipu-as-py:update_decode_snapshots` |
| Disassembly text | `src/tools/ipu-apps/test/baselines/disassembly/` | `//src/tools/ipu-apps:update_disassembly_snapshots` |
| SV package golden | `src/tools/ipu-as-py/test/golden/ipu_instr_pkg.sv` (+ localparam dict in the test) | `//src/tools/ipu-as-py:update_sv_golden` |
| Generated artifacts | `src/tools/ipu-as-py/test/baselines/generated/` | `//src/tools/ipu-as-py:update_generated_artifacts` |
| Kernel case records | `<kernel folder>/baseline.json` | `//src/tools/ipu-apps:update_kernel_baselines` |

The timing tests (E02-05, E02-06) have no update target: their expected values are code, changed by hand in the behaviour PR that changes the rule.

## Implementation

1. Add `tools/update_baselines.sh` that runs every update target above (the kernel one optional with `--with-kernels`, because it takes minutes) and then prints `git status --short` for the baseline paths.
2. Add a `## Baselines` section to the code-conventions page E01 creates:
   - the table above;
   - the rule: a PR labelled refactor must not modify any baseline file; a PR labelled behaviour change regenerates only the baselines it changes, and its description says why each one changed;
   - the localparam rule from CLAUDE.md, with `tools/sv_localparam_diff.sh <base>` (E02-03).
3. Add two checkbox lines to the PR template (or to the review guideline E01 writes, if there is no template): "No baseline file changed" / "Baselines changed: <list> because <reason>".
4. Add a short pointer in CLAUDE.md's Build & Test section: `tools/update_baselines.sh` and the rule in one sentence.

## Tests

- [ ] `tools/update_baselines.sh` on a clean base commit leaves `git status` clean.
- [ ] Each update target is reachable from the script (a test parses the script and the BUILD files and checks the lists match).

## Acceptance Criteria

- [ ] One command regenerates every non-timing baseline.
- [ ] The rule is written on the code-conventions page and linked from CLAUDE.md and the PR checklist.

## Risks

- A contributor regenerates baselines to make a refactor pass. The PR checklist and review rule are the guard; CODEOWNERS for the baseline paths (E09, after the package split) strengthens it.
