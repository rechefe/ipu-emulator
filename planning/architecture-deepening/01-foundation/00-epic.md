# Epic E01: Foundation — pinned toolchain, lockfile, fast CI, lint, conventions

ID E01 · Folder `01-foundation` · Runs after E00, before E02 · Every later epic builds on it

## Goal

Make the development loop reproducible and fast before the refactors start:

- **Same tools everywhere.** One pinned bootstrap script for CI, Docker and the agent hook; one committed Python lockfile; libraries carry runtime dependencies only.
- **CI that answers in minutes.** Pull requests and master run the default test tier once, with a Bazel cache, plus a prototyping-mode job. The heavy tier, coverage and benchmarks run nightly. Target: 5-10 minutes per PR (Q71), reached fully once E04's emulator speed-ups land.
- **Written conventions, checked where possible.** Ruff lint and format as Bazel tests over the whole repository; type checking scaffolded, strict first on `ipu_isa` and the reference API; a code-conventions page linked from `CLAUDE.md`; a PR template and review guideline in which the author states that all tests ran locally; a check that docs adopt `GLOSSARY.md` terms.
- **Numbers over time.** Benchmarks (cycles/s, assemble time, peak memory, import time, CI wall time) tracked nightly with history, without a gate.

## Background

Verified at 694a8a8 (details and line references in each issue):

- **Toolchain.** `.bazelversion` pins Bazel 8.3.0, but bazelisk is downloaded from `releases/latest` without a checksum in three workflows and the Dockerfile. `uv` is installed unpinned in four places, to a `PATH` entry (`~/.cargo/bin`) that current installers no longer use.
- **Dependencies.** A `local = True` repository rule runs `uv pip compile` against live PyPI on every fetch (`tools/python_deps.bzl`), so versions float. A line parser (`tools/pyproject_deps.bzl`) folds `dev` and `docs` extras into library deps: the assembler library depends on pytest, pyslang and three MkDocs packages. The two `uv.lock` files are unused and stale.
- **CI.** `ci.yml` runs on every push to every branch and on every PR (twice per PR push), runs the whole suite under `bazel test` and again under `bazel coverage`, with no cache. Coverage has no Python coverage tool configured and its failures are hidden by `|| true`. `--define=ipu_proto=1` never runs, so 19 prototyping-equivalence tests never run in CI.
- **Test cost.** A cold `bazel test //...` is 221 targets, about 5 750 test-seconds and 1 800 s wall on 4 cores. The 90 alias-profile targets are 60 % of that; they rerun every kernel case twice.
- **Docs build.** `GITHUB_TOKEN` is forwarded to every action (`.bazelrc:25`), changing cache keys repository-wide; the networked release-log genrule is not tagged `docs`. (Giscus and Pages concurrency are fixed in E00-06.)
- **Conventions.** No ruff or type-checker configuration, no conventions page, no PR template. 17 Python files exceed 600 lines; ruff finds 572 issues with a modest rule set and 409 of 597 files are not formatted. `GLOSSARY.md` and `docs/adr/` exist at the repository root; nothing links them.

## Decisions applied

| Decision | Applied as |
|---|---|
| Q61 | Priorities: dev loop and CI time first, then emulator speed, memory, assembler. Drives E01-04, E01-05 and the benchmark set in E01-14. |
| Q62 | One-way dependencies declared in Bazel: runtime-only library deps (E01-03); conventions page (E01-12). |
| Q63 | Pure Python and numpy; no compiled tools in the dev loop (E01-12). |
| Q64, Q71 | Benchmarks tracked, no gate (E01-14). CI 5-10 minutes, nightly heavy tier, no double run (E01-04 to E01-06). Review guideline: the author states all tests ran locally (E01-13). |
| Q65 | Ruff lint and format everywhere as a Bazel test (E01-09, E01-10); strict typing on `ipu_isa` and the reference API first (E01-11 scaffolds; E03 and E04-30 add the strict targets). |
| Q66 | Code-conventions page linked from `CLAUDE.md` with the listed content (E01-12); function-size limits through ruff complexity rules (E01-09). |
| Q68 | `GLOSSARY.md` terms; docs adoption check (E01-15). Code renames happen per area in later epics. |
| Q73 | `enormous` out of the default run; nightly and `--config=full` (E01-04, E01-06). |
| D3 | Refactor vs behaviour change in the PR template (E01-13); the ruff fix-up is proven output-identical (E01-10). |
| D7 | Prototyping mode is tested in CI (E01-08). |
| D9 | JS stays out of Bazel; `vscode-extension.yml` keeps and extends its PR path triggers for assembler and ISA changes (E01-05). |
| L26-L31, L43 (Q29) | Review defaults for build, CI and tooling, accepted with Q29 and assigned to this epic. The concrete choices this epic makes under them: committed `uv pip compile` lockfile with a consistency test (E01-02, E01-03), PR + master + dispatch triggers and one test pass (E01-05), coverage nightly only (E01-06), `--define=ipu_proto=1` kept with a `--config=proto` alias (E01-08). Where a choice is not covered by a decision, the issue says so as an open question. |

## Scope

- `tools/bootstrap.sh`, Dockerfile, `.dockerignore`, SessionStart hook.
- `requirements_lock.txt`, its refresh target, runtime-only library deps, a consistency test.
- `.bazelrc` configs `ci`, `full`, `proto`, `docs`; the `enormous` tier.
- `ci.yml`, new `nightly.yml`, `docs.yml` triggers and token scope, `vscode-extension.yml` path list.
- `ruff.toml`, ruff and mypy Bazel tests, the one-time fix-up.
- `docs/content/code-conventions.md`, `docs/content/code-review.md`, `.github/pull_request_template.md`, `CLAUDE.md` links.
- Benchmark runner, nightly benchmark job and history store.
- Glossary adoption test.

## Out of scope

- The fixes E00 makes first (giscus, Pages concurrency, release-log script, dead config, test-target sizes).
- The docs asm-block and command tests, and the docs restructure (E09).
- The import-graph test and package split (Q62, Q69); they belong to the epics that create the packages (E03 onward).
- Emulator-specific benchmarks (E04-02 plugs into E01-14's runner).
- Strict type checking of `ipu_isa` and the reference API (E03, E04-30); E01-11 provides the mechanism.
- `.github/CODEOWNERS` (E09-20, last).
- Moving JS into Bazel (D9).

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|---|---|---|---|---|---|
| E01-01 | [01-pinned-bootstrap-script.md](01-pinned-bootstrap-script.md) | One pinned toolchain bootstrap script for CI, the Dockerfile and the SessionStart hook | S | — | none |
| E01-02 | [02-committed-lockfile.md](02-committed-lockfile.md) | Committed Python lockfile: `requirements_lock.txt` replaces fetch-time resolution | M | E01-01 | none |
| E01-03 | [03-runtime-only-library-deps.md](03-runtime-only-library-deps.md) | Libraries depend on runtime requirements only, checked against the lock | M | E01-02 | none |
| E01-04 | [04-test-tiers-and-configs.md](04-test-tiers-and-configs.md) | Test tiers: `enormous` leaves the default run; `--config=ci` and `--config=full` | S | E00-09 | yes (test selection) |
| E01-05 | [05-ci-workflow.md](05-ci-workflow.md) | CI workflow: PR + master + dispatch, one test pass, Bazel cache | M | E01-01, E01-02, E01-04 | none (CI only) |
| E01-06 | [06-nightly-workflow-and-coverage.md](06-nightly-workflow-and-coverage.md) | Nightly workflow: the heavy tier, with coverage, in one pass | M | E01-04, E01-05 | none (CI only) |
| E01-07 | [07-docs-build-hermeticity.md](07-docs-build-hermeticity.md) | Docs build: network only under `--config=docs`, path-filtered docs workflow | S | E00-06, E00-07, E01-01, E01-04 | none |
| E01-08 | [08-ci-prototyping-job.md](08-ci-prototyping-job.md) | CI: run the default tier in prototyping mode | S | E01-05 | none |
| E01-09 | [09-ruff-bazel-test.md](09-ruff-bazel-test.md) | Ruff lint and format as a Bazel test over the whole repository | M | E01-02 | none |
| E01-10 | [10-ruff-fix-up.md](10-ruff-fix-up.md) | Ruff fix-up: format and lint the repository, record the complexity baseline, enable the test | M | E01-09 | none (proven output-identical) |
| E01-11 | [11-type-checking-scaffold.md](11-type-checking-scaffold.md) | Type checking as a Bazel test: scaffold now, strict for `ipu_isa` and the reference API first | S | E01-02 | none |
| E01-12 | [12-code-conventions-page.md](12-code-conventions-page.md) | "Code conventions" docs page, linked from CLAUDE.md | S | E01-04, E01-09, E01-11 | none (docs) |
| E01-13 | [13-pr-template-and-review-guideline.md](13-pr-template-and-review-guideline.md) | Pull request template and code-review guideline | S | E01-04, E01-12 | none |
| E01-14 | [14-benchmark-tracking.md](14-benchmark-tracking.md) | Benchmarks tracked nightly with stored history (no gate) | M | E01-06 | none |
| E01-15 | [15-glossary-adoption-check.md](15-glossary-adoption-check.md) | Glossary adoption check: no new uses of the terms `GLOSSARY.md` says to avoid | S | GLOSSARY.md | none |

## Suggested order

```
E01-01 ──▶ E01-02 ──┬──▶ E01-03
                    ├──▶ E01-09 ──▶ E01-10
                    └──▶ E01-11
E00-09 ──▶ E01-04 ──┐
E01-01, E01-02 ─────┴──▶ E01-05 ──┬──▶ E01-06 ──▶ E01-14
                                  └──▶ E01-08
E00-06, E00-07 ──▶ E01-07
E01-04 + E01-09 + E01-11 ──▶ E01-12 ──▶ E01-13
E01-15 (any time after GLOSSARY.md lands)
```

E01-01 to E01-05 are the critical path: after them, every PR of the program gets fast, reproducible CI. E01-10 (the repository-wide reformat) should land at a quiet moment, before the large refactors of E03 and E04 open many long-lived branches.

## Open questions collected from the issues

- E01-04: confirm the alias-profile targets (60 % of test time) move to the heavy tier; name any other targets that should.
- E01-06: keep Codecov uploads, or rely on the nightly artifact and job summary? Should the `manual` mobilevit stacks run weekly?
- E01-07: generate the release log during the docs build (as now), or commit it and refresh it from a master-only workflow (plan 08 question 4)?
- E01-09: ruff line length (proposal 100).
- E01-11: mypy (proposal) or pyright.
- E01-14: where the benchmark history lives (proposal: a `benchmark-data` branch).
