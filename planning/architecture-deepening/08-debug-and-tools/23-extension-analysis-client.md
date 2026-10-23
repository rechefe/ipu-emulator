# Extension consumes `check --json` v2; delete JS slot fit and Jinja scoping

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-23 · Size M · Depends on: E08-22, E08-21 · Behaviour change: yes (slot fit, semantic tokens and Jinja scopes come from the checkout's assembler when it supports v2)

## Goal

The extension becomes an adapter for analysis. One `AnalysisClient` spawn per document version feeds diagnostics, semantic tokens, slot-fit hints and Jinja-aware rename and completion from the v2 answer. The hand-written JS duplicates are deleted. The JS `scan()` stays only as an instant first paint and an offline fallback.

## Background

- `checker.js:120-130` spawns `check --json --input -` and returns v1 diagnostics. Everything else is computed in `language.js` from the bundled `isa.json`.
- Extension users either work from a checkout or install from the Marketplace (L38). An older checkout's `ipu-as` does not know `--json-version` and exits 2 (E05-08).
- Kept as they are:
  - spawning with an argv array;
  - debounce and cancellation (`extension.js:335-346`);
  - timeouts;
  - the byte-identical rename guard (`check_rewrites.py`).

## Implementation

1. `analysis.js` `AnalysisClient`:
   - Probe once per checker binary with `check --json-versions`. On failure, use v1 only.
   - Per document version, run `check --json-version 2 --tokens --words --jinja --input -`. Cache the result by `(uri, version)`. Every provider reads the cache.
2. Providers:
   - diagnostics: unchanged mapping;
   - semantic tokens: from `tokens`;
   - slot-fit code lens or diagnostic hints: from `words`;
   - rename and completion of Jinja names: from `jinja`.
3. With v1 only (old checkout) or no checker, fall back to `language.js` behaviour, using the E08-21 data-driven rules.
4. Delete `fits()` and the Jinja scoping functions (`jinjaDefs`, `jinjaNames`, `jinjaHints`) from `language.js`, or keep them only under a `fallback/` module used in step 3. Keep `scan()`.
5. Update `vscode-ipu-asm/README.md`: the contract version, checkout vs bundled data, and what works without a checkout.

## Tests

- [ ] JS unit tests with a fake checker returning recorded v2 JSON: each provider produces the expected VS Code objects.
- [ ] A fake checker that exits 2 on `--json-version` makes the client fall back to v1 plus the JS rules, with no error shown.
- [ ] Agreement in both directions on the corpus: JS fallback vs v2 answers, with differences listed as known gaps in a fixture file.
- [ ] Extension CI workflow green.

## Acceptance Criteria

- [ ] With a v2-capable checkout, no analysis result comes from `language.js` except the first-paint scan.
- [ ] One spawn per document version.
