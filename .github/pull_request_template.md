<!--
Fill in every section. What reviewers check, and what CI already checks:
docs/content/code-review.md (https://rechefe.github.io/ipu-emulator/code-review/).
Code rules: docs/content/code-conventions.md.
-->

## What and why
<!-- One paragraph. Link the issue (Closes #N). -->

## Kind of change
- [ ] Refactor: no output changes. Proof: <!-- golden test names, or "full suite + digests" -->
- [ ] Behaviour change (label `behaviour change`): <!-- what changes, for whom; one sentence of justification per golden update -->
- [ ] Docs / build / CI only

## Baselines
<!-- Rule: docs/content/code-conventions.md, "Baselines". Regenerate with tools/update_baselines.sh. Tick one. -->
- [ ] No baseline file changed
- [ ] Baselines changed: <!-- each file --> because <!-- one sentence per file -->

## Checks I ran locally
<!-- Tick what you ran. If a run was not possible, say which part and why under "Notes for reviewers". -->
- [ ] `bazel test --config=full //...` passed (all tests, including the nightly tier)
- [ ] `bazel test --config=proto //...` passed (if the emulator changed)
- [ ] `bazel test --test_tag_filters=lint //...` passed
- [ ] ISA change: `bazel test //src/tools/ipu-as-py:test_sv_package_golden` passed and `tools/sv_localparam_diff.sh <base-ref>` prints nothing (the `*_WIDTH` localparams of `ipu_instr_pkg.sv` are identical to the base), or the change is called out below as breaking

## Notes for reviewers
<!-- Anything surprising: performance numbers, golden updates, follow-ups, SV renames or width changes for the downstream RTL repository. -->
