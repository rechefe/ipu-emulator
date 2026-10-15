# Library raises `AssemblerError`; only the CLI prints and exits

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-06 · Size S · Depends on: E05-05 · Behaviour change: yes (in-process callers of `lark_tree.parse` / `assemble*` get an `AssemblerError` instead of a printed line plus `SystemExit(1)`; the CLI output and exit code are unchanged)

## Goal

Library code never prints and never calls `exit` (Q28). An in-process caller (the kernel registry, a harness, a test) gets a typed, located exception it can report or assert on. Only `ipu_as/cli.py` turns errors into text and exit codes.

## Background

- `lark_tree.parse` prints `Error parsing code: …` and calls `exit(1)` (`lark_tree.py:182-183`). `lark_tree.assemble` does the same with `Error assembling code: …` (`lark_tree.py:192-193`).
- `kernel_registry/cases.py:65` and the render-per-run convolution harnesses (`conv_universal_wide384/app.py:333`, `depthwise_conv_stride2_{narrow,128,16}/app.py`) call `assemble_to_bin_file` in-process. A bad kernel raises `SystemExit`, which escapes the `except (ValueError, OSError, …)` list in `kernel_registry/runner.py:62`, so the runner never reports which kernel failed.
- The CLI (`cli.py:22-30`) delegates to those functions and relies on their exit.

## Implementation

1. Delete the `print` + `exit(1)` from `lark_tree.parse` and `lark_tree.assemble`. Errors propagate as `ParseError`, `EncodeError` and the other E05-04 types.
2. In `cli.py assemble` and `disassemble`, catch `AssemblerError` and print exactly today's text: `Error parsing code: {e}` for `ParseError` and `Error assembling code: {e}` for `EncodeError` and its subclasses. Then `sys.exit(1)`. Today a template error is not caught at all: `template.render` runs outside the `try` (`lark_tree.py:174-175`), so the user gets a jinja2 traceback and exit code 1. After this PR it prints `Error rendering template: {e}` and exits 1. That is the only CLI-visible change, and the reason this issue is labelled a behaviour change.
3. Any other exception in the CLI is a bug. Let it raise, so a traceback shows.
4. Callers in `ipu-apps`: no code change is needed, since `AssemblerError` is a `ValueError` and `runner.py:62` already catches `ValueError`. E06 owns the runner's error policy (catch `IpuError`) and makes that explicit.

## Tests

- [ ] `lark_tree.assemble("bogus")` raises `ParseError`, not `SystemExit`.
- [ ] A CLI test (`click.testing.CliRunner`) on a bad source prints the same first line as the base commit and exits 1.
- [ ] `run_case` on a kernel with a broken `.asm` reports `error: …` through `runner.py`'s handler (test with a temporary kernel source).
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `grep -n "exit(\|print(" src/tools/ipu-as*/src/ipu_as/{lark_tree,session,encode}.py` is empty (excluding `__main__` demos, which should be deleted).
- [ ] CLI stdout, stderr and exit code are unchanged for the base-commit error corpus (`test_diagnostics` sources).
