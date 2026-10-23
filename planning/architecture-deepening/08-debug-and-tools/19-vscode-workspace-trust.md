# VS Code: Restricted Mode support and quoted manifest fields

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-19 · Size S · Depends on: — · Behaviour change: yes (in an untrusted workspace no configured command runs; manifest values are quoted and validated before they reach a terminal)

## Goal

Opening an untrusted folder in VS Code must not run commands that the folder's own settings or files chose. Values read from a kernel manifest must not be able to inject shell text into the integrated terminal. This is a small, early security fix (L39: Restricted Mode `limited`).

## Background

- `vscode-ipu-asm/package.json` has no `capabilities.untrustedWorkspaces` declaration.
- The settings `ipuAsm.checkCommand` (`package.json:106`), `ipuAsm.bazelFlags` (`:114`), `ipuAsm.manifestCommand` (`:151`) and `ipuAsm.queryCommand` (`:159`) can be set in a workspace's `.vscode/settings.json`.
  - The checker runs automatically on open (`extension.js:217-229` `checkCommand`, `refresh` on open).
  - The manifest command runs from `kernels.js` (around `:317`).
- `runs.js:64-76` (`kernelCommand`) quotes `flags`, the case name and options. It inserts `targets.run`, `targets.test` and `targets.benchmark` from the manifest unquoted (`:67-68`). `familyCommand` inserts `family.suite` unquoted (`:81`).
- The resulting string goes to `terminal.sendText(command)` (`kernels.js:150`). A crafted manifest can run arbitrary shell text when the user clicks "run".

## Implementation

1. `package.json`:

   ```json
   "capabilities": { "untrustedWorkspaces": { "supported": "limited",
     "description": "In Restricted Mode, IPU assembly highlighting and hover work; checking, kernel runs and manifest queries are disabled.",
     "restrictedConfigurations": ["ipuAsm.checkCommand", "ipuAsm.manifestCommand", "ipuAsm.queryCommand", "ipuAsm.bazelFlags"] } }
   ```

2. Gate every spawn and every `sendText` on `vscode.workspace.isTrusted`. When untrusted, show a status item "IPU: Restricted Mode" and run nothing. Listen to `onDidGrantWorkspaceTrust` to start the checker and kernel index when trust is granted.
3. In `runs.js`:
   - validate `targets.*` and `family.suite` against a Bazel label regex (`^@?[A-Za-z0-9_./-]*//[A-Za-z0-9_./-]*(:[A-Za-z0-9_./+=-]+)?$`);
   - reject anything else with a visible error;
   - `shellQuote` them like the other fields.
4. Document the Restricted Mode behaviour in `vscode-ipu-asm/README.md`.

## Tests

- [ ] `test/runs.js`: a manifest target `//a:b; rm -rf ~` is rejected. Valid labels produce the same command string as before.
- [ ] Integration test (extension host): with the workspace untrusted, opening an `.asm` spawns no process (spy on `child_process.spawn`), and granting trust starts the checker.
- [ ] Extension CI workflow green. Per D9 the JS tests run only in `vscode-extension.yml`.

## Acceptance Criteria

- [ ] Restricted Mode declared, with no configured command run while untrusted.
- [ ] Every manifest value that reaches a terminal is validated and quoted.
