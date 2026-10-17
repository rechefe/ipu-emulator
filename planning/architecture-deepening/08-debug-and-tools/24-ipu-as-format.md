# `ipu-as format` and the extension's format provider

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-24 · Size M · Depends on: E08-22 · Behaviour change: yes (formatting is done by the checkout's assembler; the JS formatter remains only as the fallback without a checkout)

## Goal

The assembly formatter moves to Python as `ipu-as format` (L36), next to the parser it must agree with. It checks its own output assembles to the same binary. The extension's format provider calls it, and the CLI makes the formatter usable from CI and pre-commit hooks.

## Background

- The formatter exists only in JS: `language.js:550-720` (`format(text, {indent, blankLine, fromLine, toLine})`), used by `editing.js:266-290` for document and range formatting.
- `check_rewrites.py` (Bazel `//vscode-ipu-asm:check_rewrites`) judges JS rewrites by assembling the original and the rewritten text to identical binaries.
- CLAUDE.md conventions: mnemonics and register names upper case, operands separated by spaces. A formatter is where these can be applied automatically. Whether to change case is a style decision, so it is opt-in (see the open question).

## Implementation

1. `ipu_as/formatter.py`:
   - `format_source(text, *, indent="    ", blank_line=False, line_range=None) -> str`;
   - it works on the source (not the rendered text) and keeps Jinja regions and comments verbatim;
   - it uses `language_service` tokens (E08-22) for structure.

   Port the JS rules one to one first: same output on the corpus.

   *As built:* the formatter does not use `language_service` tokens. Those tokens come from a render, and a file being edited may not render (a template without its defines) or lex, but must still format. It scans the source the way `language.js` does, with comments and Jinja tags masked. The cost is that this scan finds a tag's end by the first closing delimiter. A source with `}}`, `%}` or `#}` inside a string in a tag is read differently by Jinja, and the safety check (step 2) refuses to format it.
2. Safety check inside `format_source`: if the source has no Jinja markers, assemble before and after and require identical words; otherwise compare token streams. On mismatch, raise `FormatError` and leave the text unchanged.
3. CLI: `ipu-as format [--check] [--range L1:L2] [--indent N] --input FILE|-`. Output goes to stdout, or the file is rewritten with `--in-place`. `--check` exits 1 if the file would change. *As built:* a refused format (step 2) or an input that is not UTF-8 exits 3, in every mode.
4. Extension: when the checker supports `format` (probe via `--json-versions` or `format --help`), the format provider calls it with the range. Otherwise it uses the JS formatter. *As built:* the probe is `format --help`. The provider also falls back to the JS rules in four cases: on the `bazel run` path (about 8 s a run, against an instant answer from the same rules), for a checker command of another shape, after a format that took over 10 s, and for format on type (a process per keystroke).
5. Add `bazel test //src/tools/ipu-as-py:test_format_corpus`: formatting every kernel `.asm` is idempotent and binary-identical. It runs in report mode and is not a gate on kernel style.

## Tests

- [ ] Python output equals JS output on `test/corpus` and on every kernel source. Recorded from JS once, then the JS fixtures are the expected values.
- [ ] Idempotent: `format(format(x)) == format(x)`.
- [ ] Binary-identical for every kernel without Jinja. Token-identical for templated ones.
- [ ] `--check` exits 1 on an unformatted file and 0 on a formatted one.
- [ ] Extension CI workflow green; `bazel test //...` green.

## Acceptance Criteria

- [ ] `ipu-as format` exists, and the extension uses it when available.

## Open question

- Should `ipu-as format` also normalise mnemonic and register case to the CLAUDE.md convention (upper case)? This is binary-identical, but it is a style change across many files. The E07 per-family migration may want it as `--upper-case`. Not decided. The default here is to preserve case.
