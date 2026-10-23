# Epic E00: Verified fixes

ID E00 · Folder `00-verified-fixes` · Phase 0 (first in the program) · Feeds E01, E02, E04

## Goal

Fix the defects the architecture review found and re-checked against the code, where the fix is small, local and needs no design decision. Each issue is one small PR. Landing them first means the golden harnesses of E02 record correct behaviour, and the later refactors do not have to carry known bugs along.

## Background

The architecture review (two passes, 11 reviewers, master at 694a8a8) produced an issue register of 53 concrete defects. Every item in this epic was re-checked against the code at 694a8a8 while writing the issue, and the evidence (file:line, measured numbers) is in each issue. Two findings changed on re-check:

- `test_debug_launch` is not "a 3-instruction test marked enormous". It holds 15 fast tests (9.7 s) and a sweep that runs every kernel under the debugger (over 400 s). E00-09 splits it.
- `random_values` (`case_support.py:7-8`) is shared by 8 call sites, but only the convolution cases draw several tensors from it. E00-03 fixes those without changing any other kernel's inputs.

## Decisions applied

| Decision | Applied as |
|---|---|
| D3 | Every behaviour change is its own issue and labelled. No issue here mixes a fix with a refactor. |
| Q24, Q34 | E00-01 fixes the FP8 encoder for all formats E1-E7 regardless of the naming and special-value questions, which stay open. |
| Q35 | AaQ is out of scope. AaQ defects (including crashes) go to the open-questions page, not here. |
| L1, L3, L4 (Q29) | E00-02 (asserts become exceptions), E00-05 (`max_cycles` default 1 000 000, stats reset per run). E04-03 and E04-21 build on them. |
| L41 (Q29) | E00-10 deletes `tools/fp32_to_fp4.py`. |
| Q38, Q59 | E00-02 uses temporary exception classes that subclass both `EmulatorError` and the matching built-in. E04-03 re-parents them under `IpuError`. |

## Scope

- FP8 encoder rounding (host-side data loading).
- Register-file input validation.
- Test-data determinism and independence.
- `IPU_EMU_PROTOTYPING` parsing; run statistics and the cycle-limit default.
- Docs deployment (giscus, Pages concurrency) and the release-log script.
- Test integrity: silent skips, target sizes, library sources, an unasserted warning.
- Dead configuration and one dead script.
- One doc command that does not run as written.

## Out of scope

- All AaQ items (Q25, Q35): activation-model divergence, `DEFAULT_ELU_ALPHA` being unpatchable, EXP2 overflow in `ACTIVATE.QUANTIZE`, reciprocal and rsqrt of zero. They are recorded as open questions (E09-05: [OQ-20](../../../docs/content/requirements/open-questions.md#emulator-and-isa) to OQ-25 and OQ-30).
- XMEM profiler reporting a 512 MiB write on `clear()` / `__setitem__`: fixed by the paged XMEM with ranged events (E04-05).
- `acc_store` in the SV package and the commit-hash stamp: E03 (HARDWARE / EMULATOR tags, SV generation).
- Comma-separated syntax in spec text and generated docs: E03 (derived syntax) and E09.
- FP8 naming (`fp8_e4m3` to `fp8_e4`) and special values: E07 and the open-questions page (Q34).
- Docs drift beyond the assemble command: E09-01, E09-02.
- Every test reruns 3x through the alias-profile targets: E01-04 moves those targets to the heavy tier; E06 owns profiling inside the generated kernel tests.

## Issue index

| ID | File | Title | Size | Depends on | Behaviour change? |
|---|---|---|---|---|---|
| E00-01 | [01-fp8-subnormal-carry.md](01-fp8-subnormal-carry.md) | FP8 encoder: a subnormal that rounds up must carry into the smallest normal | S | — | yes (host FP8 encoding near the smallest normal) |
| E00-02 | [02-regfile-asserts-to-exceptions.md](02-regfile-asserts-to-exceptions.md) | RegFile: replace the validation `assert`s with real exceptions | S | — | yes (exception type on invalid input) |
| E00-03 | [03-independent-test-data-seeds.md](03-independent-test-data-seeds.md) | Test data: independent draws for convolution cases, stable seeds for mobilevit fillers | S | — | yes (test inputs only) |
| E00-04 | [04-prototyping-env-case-insensitive.md](04-prototyping-env-case-insensitive.md) | `IPU_EMU_PROTOTYPING`: parse false values case-insensitively | S | — | yes (`FALSE`/`NO`/`Off` mean off) |
| E00-05 | [05-runstats-per-run-and-max-cycles.md](05-runstats-per-run-and-max-cycles.md) | Run statistics: reset per run, keep `total_cycles` on error, one `max_cycles` default | S | — | yes (stats scope, default limit) |
| E00-06 | [06-giscus-env-and-docs-concurrency.md](06-giscus-env-and-docs-concurrency.md) | Docs: forward giscus settings into the MkDocs build, and never cancel a master deploy | S | — | yes (deployed site) |
| E00-07 | [07-release-log-network-errors-and-latest.md](07-release-log-network-errors-and-latest.md) | Release log script: survive network errors, and make `--latest` merge instead of overwrite | S | — | yes (script output) |
| E00-08 | [08-missing-fixtures-fail-and-dead-skip.md](08-missing-fixtures-fail-and-dead-skip.md) | Kernel tests: a missing fixture fails, and the dead `needs_softmax` skip goes | S | — | none (tests fail instead of skip) |
| E00-09 | [09-test-target-hygiene.md](09-test-target-hygiene.md) | Test targets: split `test_debug_launch`, keep kernel test files out of `ipu_apps_lib` | S | — | none |
| E00-10 | [10-dead-config-and-fp32-to-fp4.md](10-dead-config-and-fp32-to-fp4.md) | Remove dead build config and the unused `tools/fp32_to_fp4.py` | S | — | none |
| E00-11 | [11-assert-str-acc-reg-warning.md](11-assert-str-acc-reg-warning.md) | Tests assert the STR_ACC_REG debug-only warning instead of ignoring it | S | — | none |
| E00-12 | [12-assemble-command-in-claude-and-skill.md](12-assemble-command-in-claude-and-skill.md) | CLAUDE.md and SKILL.md: an `ipu-as assemble` command that runs as written | S | — | none (docs) |
| E00-13 | [13-giscus-category-id.md](13-giscus-category-id.md) | giscus: commit the Announcements `category_id` in `docs/mkdocs.yml` (`ready-for-human`) | XS | E00-06 | yes (deployed site) |

## Suggested order

All issues except E00-13 are independent and can be worked in parallel. E00-13 is a manual step a human does after E00-06 (agent sessions cannot read the discussion category id). Two orderings matter for later epics:

```
E00-01 ──▶ E04-13 (element codecs use the fixed encoder as reference)
E00-02 ──▶ E04-03 (error hierarchy re-parents the temporary classes)
E00-05 ──▶ E04-21 (cycle limit becomes fault F-9)
E00-06 ─┬─▶ E01-07 (docs build hermeticity)
E00-07 ─┘
E00-09 ──▶ E01-04 (test tiers: enormous = heavy tier)
E00-10 ──▶ E01-02 (lockfile refresh command replaces the stale references)
```

Land E00-01, E00-03 and E00-05 before E02 records its goldens, so the goldens capture fixed behaviour.
