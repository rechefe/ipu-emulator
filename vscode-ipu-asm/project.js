'use strict';

// Where the IPU checkout is, for a file or an open folder. Everything that runs a
// process (checker, manifest, kernel buttons) runs in a checkout's Bazel workspace
// and nowhere else. The open folder may be inside one, hold several, or be another
// project whose .asm is some other assembly, so the checkout is found from the path:
// the nearest workspace above holding the assembler, or for a folder, those below.
// Plain node, no `vscode` (see test/project.js).

const fs = require('fs');
const path = require('path');

const WORKSPACE_FILES = ['MODULE.bazel', 'WORKSPACE.bazel', 'WORKSPACE'];
/** What makes a workspace an IPU checkout: the package the checker is built from. */
const IPU_PACKAGE = path.join('src', 'tools', 'ipu-as-py');
/** How deep below an open folder to look for checkouts. */
const SEARCH_DEPTH = 3;
const SKIP = /^(\.|node_modules$|bazel-)/;

function isDirectory(p) {
  try {
    return fs.statSync(p).isDirectory();
  } catch {
    return false;
  }
}

function isCheckout(dir) {
  return WORKSPACE_FILES.some((f) => fs.existsSync(path.join(dir, f))) && isDirectory(path.join(dir, IPU_PACKAGE));
}

/** The IPU checkout holding `fsPath` (a file or a directory), or null. */
function checkoutOf(fsPath) {
  for (let dir = isDirectory(fsPath) ? fsPath : path.dirname(fsPath); ; dir = path.dirname(dir)) {
    if (isCheckout(dir)) return dir;
    if (path.dirname(dir) === dir) return null;
  }
}

/** The IPU checkouts an open folder stands for: the one it is in, or else
 *  those below it (a few levels down, not inside another checkout). */
function checkoutsFor(folder) {
  const above = checkoutOf(folder);
  if (above) return [above];
  const found = [];
  const walk = (dir, depth) => {
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (SKIP.test(entry.name)) continue;
      const child = path.join(dir, entry.name);
      // A linked checkout counts, but links are not followed down: they could loop.
      const link = entry.isSymbolicLink();
      if (!(link ? isDirectory(child) : entry.isDirectory())) continue;
      if (isCheckout(child)) found.push(child);
      else if (depth > 1 && !link) walk(child, depth - 1);
    }
  };
  walk(folder, SEARCH_DEPTH);
  return found.sort();
}

/** `target`'s path relative to `dir` when it lies inside it, else null. */
function within(dir, target) {
  const rel = path.relative(dir, target);
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel) ? rel : null;
}

module.exports = { checkoutOf, checkoutsFor, isCheckout, within };
