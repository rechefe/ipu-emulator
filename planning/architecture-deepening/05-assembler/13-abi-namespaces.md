# `cr` / `lr` ABI namespaces in the render context

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-13 · Size S · Depends on: E05-09 · Behaviour change: none (additive)

## Goal

A kernel names its registers by role, `{{ cr.data_base }}` and `{{ lr.data_ptr }}`, instead of a block of `{% set X = "crN" %}` lines that the harness must mirror by number (Q80). The assembler only accepts a name → register mapping and exposes it as read-only template namespaces. Who supplies the mapping is E06's job: the harness reads `kernel.toml [abi]`, and the CLI reads it from a file.

## Background

- Kernels declare their calling convention as `{% set X = "crN" %}` lines in the `.asm`, and harnesses repeat the numbers in `set_cr(<int>, …)` calls. The plan-05 inventory found 553 such annotations, 49 files with CR maps in comments, and visible drift. One example: `softmax_rows.asm:32` says `CR7 = ROW_BYTES (512)`, while `softmax/app.py:166` sets `set_cr(7, 1)`.
- Q80: a kernel is `<name>.asm` + `kernel.toml`, with `[abi]` mapping name → CR. Case params become template values. E06 owns `kernel.toml`.
- CLAUDE.md: any `CR0`–`CR15` is a valid operand. `CR15` conventionally holds dstructure configuration but is not blocked.

## Implementation

1. `render(..., namespaces={"cr": {"data_base": "CR2", ...}, "lr": {"data_ptr": "LR0", ...}})` (the parameter already exists from E05-09):
   - each namespace is exposed as a read-only object whose attribute access returns the register token;
   - an unknown name raises `TemplateError("unknown ABI name cr.data_bse")`, whatever the `Undefined` policy;
   - values must match `^(CR|LR)\d+$` against the ISA register count from `ipu_isa`, else `TemplateError` at render start;
   - two names mapping to the same register are allowed. Duplicate-register policy is E06's validation.
2. Reserve the names `cr` and `lr`: a declared param or define with either name is an error.
3. `AssemblySession(..., abi: Mapping[str, Mapping[str, str]] | None)` forwards it as `namespaces`.
4. CLI: `--abi FILE` reads a small JSON or TOML file `{cr = {name = "CR2"}, lr = {…}}`. Open question: should `--abi` instead take a `kernel.toml` path and read its `[abi]` table (E06 format)? Whichever E06 decides, the session interface stays a mapping.
5. Document the namespaces in the assembly-syntax reference (the E09 page rewrite picks this up).

## Tests

- [ ] `{{ cr.data_base }}` renders `CR2`, and the program assembles to the same word as a literal `CR2`.
- [ ] An unknown ABI name raises with the template line.
- [ ] A bad value (`CR16`, `R0`) raises at render start.
- [ ] A define named `cr` is rejected.
- [ ] E02 digests unchanged (no source uses the namespaces yet).
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] The assembler accepts an ABI mapping through the session, the CLI and `render`, with no dependency on `ipu_harness`.
- [ ] E06-06 / E06-07 pass `kernel.toml [abi]` as `abi=` (coordination confirmed in those issues).
