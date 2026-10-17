# Switch templates to `StrictUndefined`

Part of [Epic: Assembler as a deep module](00-epic.md) · ID E05-11 · Size S · Depends on: E05-09, E05-12 · Behaviour change: yes (a template that names an undefined variable now fails instead of rendering an empty string)

## Goal

A misspelled or missing template value is an error, not an empty string spliced into assembly (L10). Combined with declared parameters, this makes the only way to give a template a value the declared, checked one.

## Background

- `template._ENVIRONMENT = SandboxedEnvironment()` (`ipu_as/template.py:38`) uses the default `Undefined`. `{{ typo }}` renders as `""`, and the assembly may still parse. For example, `BEQ LR0 LR1 {{ lable }}` becomes `BEQ LR0 LR1`, which is then an operand-count error far from the cause.
- Some templates test definedness on purpose (`is defined` / `is undefined`) or use `| default(...)`. Those must be rewritten as declared params with defaults first (E05-12 does this for the known files).

## Implementation

1. Inventory: a one-off script (kept as a test helper) renders every `.asm` and `.asm.j2` under `src/` with the defines each caller uses. It uses the registry's case defines, the E05-12 Bazel `defines`, and the mobilevit test contexts. With `StrictUndefined`, every failure is listed. Attach the list to the PR.
2. Fix each failure by declaring the param (with its current effective default) or by fixing a real bug. A real bug is a behaviour change: list it separately in the PR, with the digest change it causes.
3. Set `undefined=jinja2.StrictUndefined` on the environment in `render`.
4. Add a lint test that fails on `| default(` and on `is defined` / `is undefined` in any `.asm` or `.asm.j2` under `src/`. The declared-param default replaces both.

## Tests

- [ ] `render("{{ x }}")` raises `TemplateError` naming `x` and the line.
- [ ] The inventory script renders every source with no failure.
- [ ] Lint: a planted `| default(` fails the lint test.
- [ ] E02 digests unchanged except for the bugs listed in the PR. Each such change has its own commit, and the digest manifest is updated with the reason.
- [ ] `bazel test //...` green.

## Acceptance Criteria

- [ ] `StrictUndefined` is in effect for every render.
- [ ] No `| default(` and no definedness tests remain in kernel sources.
