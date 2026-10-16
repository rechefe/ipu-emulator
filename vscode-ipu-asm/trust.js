'use strict';

// Restricted Mode. In a workspace VS Code does not trust, the folder's own settings and
// files would choose what runs: its .vscode/settings.json names the checker and manifest
// commands, and its BUILD files decide what `bazel` does. So nothing is run there at all;
// highlighting, hover and the other editing help need no process and keep working.

const vscode = require('vscode');

const RESTRICTED = 'this workspace is in Restricted Mode, so no assembler, Bazel or kernel command runs. Trust it to turn them on.';

/** Whether this workspace may run commands. When it may not and `what` is named, the user
 *  is told that `what` did not run. */
function mayRun(what) {
  if (vscode.workspace.isTrusted) return true;
  if (what) vscode.window.showWarningMessage(`IPU: ${what} did not run: ${RESTRICTED}`);
  return false;
}

module.exports = { RESTRICTED, mayRun };
