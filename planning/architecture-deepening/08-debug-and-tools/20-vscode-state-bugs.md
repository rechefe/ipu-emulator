# VS Code: per-checkout state and retry after a failed check

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-20 · Size S · Depends on: — · Behaviour change: yes (bug fixes: a failed check is retried, status no longer leaks across checkouts, and untitled buffers are not checked against an unrelated checkout)

## Goal

Fix three state bugs in the extension's checker integration. These are small, independent fixes that land early.

## Background

- Module-level state is shared by every checkout in a multi-root window: `pending`, `checked`, `warnedOnce`, `lastError`, `running` and `status` (`vscode-ipu-asm/extension.js:30-46`). One checkout's broken toolchain shows as "unavailable" for all of them.
- `refresh` records `checked.set(key, version)` (`extension.js:308`) before the error branch (`:318-329`). After a toolchain failure, the same document version is never re-checked unless the user edits it.
- `checkoutFor(uri)` (`extension.js:209-213`) sends every non-`file:` URI (untitled buffers, git diff views, remote schemes) to the first checkout in `kernelApi.indexes`. A diff view of another project's `.asm` gets checked with this project's assembler.

## Implementation

1. Introduce a `CheckState` class holding `pending`, `checked`, `running`, `lastError` and `warnedOnce`, plus a `Map<checkoutRoot, CheckState>`. `showStatus` reflects the active editor's checkout.
2. Move `checked.set(key, version)` after the success path, so it is not set when `result.error` is present.
3. `checkoutFor(uri)`:
   - for `untitled:`, use the checkout of the active workspace folder only if exactly one folder is a checkout, else none;
   - for `git:` and other schemes, resolve `uri.fsPath` when the scheme maps to a file path, else none.

## Tests

- [ ] `test/checker.js` or a new unit test: after an error result, the next `refresh` of the same version runs the checker again.
- [ ] Two fake checkouts: an error in one does not change the status shown for a document in the other.
- [ ] An untitled buffer in a two-checkout window is not checked.
- [ ] Extension CI workflow green.

## Acceptance Criteria

- [ ] No mutable module-level state in `extension.js` other than the `CheckState` map and the VS Code disposables.
