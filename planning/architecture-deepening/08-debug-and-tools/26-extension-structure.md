# Extension structural cleanup

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-26 · Size M · Depends on: E08-20 · Behaviour change: none

## Goal

Make the extension's own code follow the same rules as the Python packages:

- one responsibility per module;
- no monkey-patched prototypes;
- no hard-coded registry file names;
- no synchronous filesystem walks on the UI thread;
- disposables disposed.

## Background

- `editing.js` `register` (from `:24`) is about 330 lines and wires every editing provider. `kernels.js` `register` (from `:332`) is about 180 lines.
- `language.js:830-835` memoises by replacing methods on `Language.prototype` in a loop.
- `kernels.js` hard-codes registry file names (around `:411-414`). The manifest already lists them.
- Synchronous directory walks:
  - `kernels.js` (around `:380-396`);
  - `project.js:22` (`statSync`);
  - `checker.js:60-85` (`readdirSync` / `statSync` in the freshness scan used by `isFresh`, `:98`).
- `KernelIndex` is never disposed.

## Implementation

1. Split `editing.js` into `providers/{hover,completion,signature,rename,format,symbols}.js`, each exporting `register(context, env)`. `editing.js` becomes the list.
2. Replace the prototype patch with an explicit `memoised(fn)` wrapper used at the call sites, or a `TextAnalysis` object computed once per text that holds the results.
3. Read registry file names from the manifest (the `kernels` manifest JSON already carries paths), and delete the constants.
4. Make the walks asynchronous (`fs.promises`), debounce them, and cache `isFresh` per binary until a file-system watcher reports a change.
5. Push `KernelIndex` instances into `context.subscriptions`, and dispose them on folder removal.

## Tests

- [ ] Existing JS unit and integration tests pass unchanged.
- [ ] A test that the extension activates with no synchronous `fs` call on the activation path (stub `fs.*Sync` to throw during activation).
- [ ] Extension CI workflow green.

## Acceptance Criteria

- [ ] No JS function is longer than about 80 lines in `vscode-ipu-asm/*.js`.
- [ ] No `Language.prototype[...] =` assignment.
