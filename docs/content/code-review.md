# Code review

Every pull request uses the template in `.github/pull_request_template.md`, which GitHub fills in when the PR is opened. The rules code is checked against are on the [Code conventions](code-conventions.md) page.

## Author

1. Keep a PR small: one issue per PR.
2. Fill in every section of the template.
3. Mark the kind of change. A refactor changes no output; name the golden tests (or the full suite with digests) that prove it. A behaviour change is its own PR with the `behaviour change` label.
4. Treat each golden update as a behaviour change. Give one sentence of justification per updated golden. The baseline rule and the one command that regenerates every baseline (`tools/update_baselines.sh`) are in [Code conventions, Baselines](code-conventions.md#baselines).
5. Run all tests locally, including the heavy tier, and say so in the template:

    ```bash
    bazel test --config=full //...                  # every test, including the nightly tier
    bazel test --config=proto //...                 # if the emulator changed
    bazel test --test_tag_filters=lint //...        # ruff lint and format
    ```

    If a run was not possible, say which part did not run and why. CI runs only the default tier; the heavy tier runs nightly, so a break there shows up only the next morning unless the author ran it.

6. After an ISA change, check the generated SystemVerilog package as `CLAUDE.md` describes ("Generated SystemVerilog Package"). The `*_WIDTH` localparams stay identical to the base, or the PR calls the change out as breaking.
7. When `//:glossary_adoption_test` reports fewer uses of an avoided term than its baseline, run `bazel run //:glossary_baseline` and commit `tools/glossary_baseline.json`. When it reports more, use the glossary term it names; a raised count in the baseline needs a reason in the PR.

## Reviewer

Check the PR against the [Code conventions](code-conventions.md) and the issue's acceptance criteria. Leave what CI checks to CI: a formatting or lint remark is not a review comment.

| CI checks (not review comments) | The reviewer checks |
|---|---|
| Formatting and lint (ruff, the thresholds in `ruff.toml`) | The rules marked **Review** on the conventions page |
| Type checks of packages with a `mypy_test` | Test independence: expected values come from an independent source, not from the code under test |
| The default test tier, including the glossary check on the docs (`//:glossary_adoption_test`) | Names in code against `GLOSSARY.md`, and any raised count in `tools/glossary_baseline.json` |
| The default tier in prototyping mode (`--config=proto`) | The refactor-versus-behaviour split: a refactor PR changes no output and says how that is proven; a behaviour change is labelled and separate |
| Docs drift tests (`asm` blocks assemble, commands are real, links resolve) | The author's statement that all tests ran locally, and a reason for any part that did not |
| | One sentence of justification per golden update |

## Merging

- The required checks are green.
- One approval.
- Squash or rebase, as the repository is set up.
- A behaviour change that touches the SystemVerilog package (`ipu_instr_pkg.sv`): inform the downstream RTL owners. List renamed fields in a rename table and call out width changes as breaking.

## Agents

An agent that writes a PR description uses the `mattpocock-skills:pr` skill and fills in the template. An agent that reviews a branch or PR uses the `mattpocock-skills:code-review` skill, with this page and the conventions page as its standards.
