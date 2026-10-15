# SV package header: hash of the body instead of the commit

Part of [Epic E03: ISA package](00-epic.md) · ID E03-21 · Size S · Depends on: E02-03 · Behaviour change: yes, header line only: the vendored file no longer changes on every commit · `*_WIDTH` localparams: **unchanged** (header comment only)

## Goal

Decision L32: the vendored `ipu_instr_pkg.sv` stamps a hash of its own body instead of the source commit. The file then changes only when its content changes, a downstream diff is meaningful, and the downstream repo can still verify that its copy is unmodified.

## Background

- Header line 2 today: `// Source: rechefe/ipu-emulator @ <commit>` (`bazel-bin/docs/ipu_instr_pkg.sv:2`), rendered from `source_commit` (`ipu_as/gen_codegen.py:379-391`).
- The commit comes from `bazel-out/stable-status.txt` (`STABLE_GIT_COMMIT`, `tools/workspace_status.sh`), and the genrule is stamped (`docs/BUILD.bazel:43-51`, `stamp = 1`), so the action reruns on every commit. `generate_sv_package` falls back to `workspace_source_commit()` (`gen_codegen.py:429-447`), which shells out to git.
- The compound comment in the template says "slots MSB → LSB: cond, lr×N, xmem, mult, acc, aaq, break" (`ipu_instr_pkg.sv.j2`); there is no `xmem` slot and `store`/`acc_store` are missing. The real order is `LAYOUT_ORDER` (E03-06).

## Implementation

1. Render the body (everything after the header block), compute `sha256` over its UTF-8 bytes, and emit `// Content: sha256:<hex>` as header line 2. Keep line 1 ("Auto-generated from instruction_spec.py — do not edit.").
2. Add `def verify_sv_package(text: str) -> bool` and a `bazel run //docs:verify_sv_package -- <file>` target, so the downstream repo (or a reviewer) can check a vendored copy.
3. Remove stamping: drop `stamp = 1`, the `stable-status.txt` argument, `source_commit_from_status`, `workspace_source_commit` and `UNKNOWN_SOURCE_COMMIT`.
4. Generate the compound-layout comment from `LAYOUT_ORDER` (comment-only change).
5. Regenerate the E02-03 golden (header and comment lines change; the localparam dict is not edited).

## Tests

- [ ] `test_sv_package_golden` passes with the regenerated golden; `test_localparams_match_table` is untouched.
- [ ] `verify_sv_package` accepts the generated file and rejects it after a one-character body edit.
- [ ] Rendering twice gives identical bytes; the genrule output is identical across two commits that do not touch the ISA (cache hit).
- [ ] `test_sv_package_compiles` passes.

## Acceptance Criteria

- [ ] The vendored file's bytes depend only on its content.
- [ ] Every `*_WIDTH` localparam is unchanged (`tools/sv_localparam_diff.sh <base>` prints nothing).
- [ ] The PR description tells the downstream RTL repo that only header and comment lines changed.
