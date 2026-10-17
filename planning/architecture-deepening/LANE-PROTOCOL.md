# Lane protocol

The program runs in parallel **lanes**. Each lane is one Claude Code session working one independent set of issues on its own branch. The coordinating session merges finished lanes into `claude/tender-pascal-5fsay7`. Every lane follows this protocol.

## Setup

- Work on your lane branch, created from `claude/tender-pascal-5fsay7`. Do not push to `claude/tender-pascal-5fsay7` itself.
- Read the issue files for your lane, their epic `00-epic.md`, [IMPLEMENTATION-DECISIONS.md](IMPLEMENTATION-DECISIONS.md), [DECISION-LOG.md](DECISION-LOG.md), [`GLOSSARY.md`](../../GLOSSARY.md) and [`docs/adr/`](../../docs/adr/). The decision log is the user's authoritative decisions. `CLAUDE.md` rules apply.
- Touch only files your issues need. Other lanes are editing other parts of the repo at the same time.

## Per issue (or per batch)

**Batches.** Size-S issues that are related may be batched, 3–5 at a time. One implementation pass makes one commit per issue, and one review covers the whole batch against every issue in it. Size-M and size-L issues each get their own loop.

1. **Implement.** Before editing, record `base = $(git rev-parse HEAD)`. Implement the issue completely: Implementation, Tests and Acceptance Criteria. Run the Bazel test targets of every package you touch. Never run `bazel clean`; GitHub CI runs the full suite on every push. Commit with the message `<ISSUE-ID>: <issue title>`, a body explaining what changed, and the trailer lines this session's instructions require.
2. **Review.** Run a code review of `base..HEAD` in a fresh subagent:
   - **If the `mattpocock-skills:code-review` skill is available,** invoke it with fixed point `base`. Use the issue file(s) as the spec, and `CLAUDE.md`, `GLOSSARY.md`, `docs/adr/` and the two decision files as the standards.
   - **If it is not available,** follow the [two-axis review](#two-axis-review) below.
3. **Approval.** The change is approved only when there are no **blocking** findings. A blocking finding is one of:
   - a real bug
   - an unmet spec requirement or acceptance criterion
   - failing tests
   - a clear violation of a documented standard or decision

   Style nits and judgement-call smells are non-blocking. Never block on work the issue explicitly defers to another issue.
4. **Fix and repeat.** Fix the blocking findings, commit them as `<ISSUE-ID>: address review`, and review again. Stop after 4 rounds. If findings are still open, record them in your final report.
5. **Push** the lane branch after each approved issue or batch: `git push -u origin <lane-branch>`. On network errors, retry up to 4 times with backoff.

## Finish

When every issue in the lane is done, report back to the coordinating session with `send_message` (session `session_018TB1hKL3qUMDiwud2qDZ4v`). Give the lane branch, the final commit, each issue's status (approved / not approved, with open findings) and anything another lane must know.

## Two-axis review

This is adapted from mattpocock-skills `code-review`. Run it in a subagent so the review context stays separate from the implementation.

1. **Pin the change.** Find the fixed point with `git rev-parse <base>`, then take `git diff <base>...HEAD` and `git log <base>..HEAD --oneline`. Fail if the diff is empty.
2. **Standards axis.** This runs in its own subagent. Its sources are `CLAUDE.md`, `GLOSSARY.md`, `docs/adr/*`, the code-conventions page if it exists, and the decision files. It reports two things:
   - **Violations:** each place the diff breaks a documented standard, citing the file and the rule.
   - **Smells:** any of the baseline smells below, always reported as a judgement call. A documented standard overrides a smell. Skip anything tooling already enforces.

   Smell baseline (Fowler):

   | Smell | What to look for |
   |---|---|
   | Mysterious Name | A name that does not reveal what the thing is. |
   | Duplicated Code | The same logic shape appears twice. |
   | Feature Envy | Code that reaches into another object's data more than its own. |
   | Data Clumps | The same few fields travel together everywhere. |
   | Primitive Obsession | A primitive standing in for a domain concept. |
   | Repeated Switches | The same switch on the same type appears in more than one place. |
   | Shotgun Surgery | One change forces scattered edits. |
   | Divergent Change | One module changes for unrelated reasons. |
   | Speculative Generality | Hooks or parameters nothing needs. |
   | Message Chains | Long `a.b().c().d()` navigation. |
   | Middle Man | A thing that only delegates. |
   | Refused Bequest | A subclass that ignores what it inherits. |

3. **Spec axis.** This runs in its own subagent, in parallel with the standards axis. Its source is the issue file(s). It reports three things, each quoting the spec line:
   - requirements that are missing or only partly done
   - behaviour nobody asked for (scope creep)
   - requirements that look implemented but are wrong
4. **Aggregate.** Report under `## Standards` and `## Spec` headings, without merging or re-ranking the two axes. Then classify each finding as blocking or non-blocking under the approval rule above.
