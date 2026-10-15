# Language data from the checkout, bundled data as fallback

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-25 · Size M · Depends on: E08-15, E08-23 · Behaviour change: yes (hover, completion and signature help follow the ISA of the open checkout instead of the ISA the extension was built with)

## Goal

On a branch that changes the ISA, hover and completion agree with the diagnostics, because all of them come from the same checkout. Marketplace users without a checkout keep working on the bundled data, and the status bar says which source is in use (L38).

## Background

- Hover and completion load the bundled `data/isa.json` once (`extension.js:75-90`, `loadHoverData`). It is generated at extension build time from master.
- Diagnostics come from the checkout's `ipu-as` (`checker.js:130`).
- On a branch that adds or renames an instruction, the two disagree: the editor shows "unknown instruction" hover text for code that assembles cleanly, or the reverse.
- E08-21 adds `schemaVersion` to `isa.json`. E08-15 provides `ipu-gen vscode-data`.

## Implementation

1. A `LanguageSource` seam with two adapters:
   - `CheckoutLanguageSource(root)` runs `bazel run //src/tools/ipu-gen:ipu-gen -- vscode-data --out <cache dir>` once per checkout. It caches the result keyed by the checker binary's mtime, the same freshness rule as `checker.js` `isFresh`, and loads `isa.json` from there.
   - `BundledLanguageSource` loads `data/isa.json` shipped with the extension.
2. Selection:
   - a trusted workspace (E08-19) with a checkout and a compatible `schemaVersion` uses the checkout;
   - otherwise use the bundled data, with a status-bar suffix "(bundled ISA)";
   - a checkout whose `schemaVersion` is newer than the extension supports falls back to bundled data and warns once: "update the extension".
3. Generating language data must not block first paint. Use the bundled data until the checkout data is ready, then swap and refresh the open editors.
4. README: explain both sources and how to force one (`ipuAsm.languageSource: auto|checkout|bundled`).

## Tests

- [ ] JS unit tests: source selection for trusted/untrusted, compatible/newer schema, and generation failure.
- [ ] Integration test with the fixture checkout: a mnemonic present only in the checkout's ISA gets hover text after the swap.
- [ ] Extension CI workflow green.

## Acceptance Criteria

- [ ] With a trusted checkout present, hover, completion, slot fit, formatting and diagnostics all come from the checkout.
- [ ] The status bar shows when bundled data is in use.
