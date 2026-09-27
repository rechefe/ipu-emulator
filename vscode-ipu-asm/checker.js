'use strict';

// Which checker to run, and whether the fast one is still telling the truth.
//
// `bazel run //src/tools/ipu-as-py:ipu-as` costs about 250ms of client startup
// before the checker even begins, and it takes the workspace lock, so a check
// queues behind whatever build is running in a terminal. That is the stall
// people notice: not the checking, which is under a millisecond on a small file
// and 67ms on the largest kernel here.
//
// The binary Bazel builds is a self-contained runfiles launcher. Run it
// directly and both costs disappear -- no client, no lock -- but so does the
// one thing `bazel run` was doing for us: rebuilding when a source changed.
// That has to come back, because diagnostics from an assembler that no longer
// exists are worse than slow ones.
//
// So freshness is decided per check, by comparing the binary against the
// sources it was built from: 31 files, 0.6ms, against a 340ms check. Stale
// means this check goes through `bazel run`, which rebuilds as a side effect
// and leaves the next one fast again. Nothing to invalidate, nothing to
// remember, and it notices a `git pull` or a branch switch exactly as well as
// it notices an edit -- which a watcher, blind while the editor was closed,
// would not.
//
// Split out from extension.js so it can be tested without a running editor: it
// imports nothing from `vscode`.

const fs = require('fs');
const path = require('path');

/** The assembler target the extension checks with. */
const CHECKER_TARGET = '//src/tools/ipu-as-py:ipu-as';

/** Path of that target's binary under bazel-bin. */
const CHECKER_BINARY_PATH = path.join('src', 'tools', 'ipu-as-py', 'ipu-as');

/** Sources the checker is built from, relative to the workspace root.
 *
 *  Directories rather than files: a new module inside one of them is covered
 *  the day it is added, where a list of filenames would silently stop being
 *  the whole story. */
const SOURCE_DIRS = [
  path.join('src', 'tools', 'ipu-as-py', 'src'),
  path.join('src', 'tools', 'ipu-common', 'src'),
];

const SOURCE_EXTENSIONS = ['.py', '.lark', '.j2'];

/** Where `bazel build` leaves the checker, given `bazel info bazel-bin`. */
function checkerBinary(bazelBin, platform) {
  const binary = path.join(bazelBin, CHECKER_BINARY_PATH);
  return platform === 'win32' ? `${binary}.exe` : binary;
}

/** Newest mtime among the checker's sources, in ms; 0 if there are none.
 *
 *  Zero means this workspace does not hold the assembler, so there is nothing
 *  to be stale against and the question does not arise. */
function newestSourceMtime(root, dirs = SOURCE_DIRS) {
  let newest = 0;
  const walk = (dir) => {
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return; // absent or unreadable: not a source that can invalidate anything
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (SOURCE_EXTENSIONS.some((ext) => entry.name.endsWith(ext))) {
        try {
          const { mtimeMs } = fs.statSync(full);
          if (mtimeMs > newest) newest = mtimeMs;
        } catch {
          // Raced with a delete; the next check sees the settled tree.
        }
      }
    }
  };
  for (const dir of dirs) walk(path.join(root, dir));
  return newest;
}

/** True when `binary` is at least as new as every source it was built from.
 *
 *  `builtAgainst` is the source stamp of the last build that is known to have
 *  succeeded, and it is not an optimization -- without it the rule is wrong.
 *  Bazel rebuilds on content, not timestamps: a `git checkout`, a branch
 *  switch or a `touch` re-dates every source while leaving the bytes alone, so
 *  Bazel correctly does nothing and the binary's mtime never moves. Comparing
 *  mtimes alone would then call it stale forever and send every check back
 *  through Bazel, which is the cost this was meant to remove. Recording what a
 *  build was built against fixes it, and losing that record -- a reload, a new
 *  window -- costs exactly one `bazel run`. */
function isFresh(binary, root, builtAgainst = 0) {
  let built;
  try {
    built = fs.statSync(binary).mtimeMs;
  } catch {
    return false; // `bazel clean`, or never built
  }
  const newest = newestSourceMtime(root);
  // Equal counts as fresh: a build writes its output after reading its inputs,
  // and filesystems with coarse timestamps report both at the same tick.
  return newest === 0 || Math.max(built, builtAgainst) >= newest;
}

/** The slow path: correct whatever state the tree is in, and rebuilds. */
function bazelCommand() {
  return [
    'bazel',
    'run',
    '--ui_event_filters=-info,-stdout,-stderr',
    '--noshow_progress',
    CHECKER_TARGET,
    '--',
    'check',
    '--json',
    '--input',
  ];
}

/** The fast path: the built binary, no Bazel client and no workspace lock. */
function directCommand(binary) {
  return [binary, 'check', '--json', '--input'];
}

/** True when `argv` is the Bazel fallback, which rebuilds as it checks. A
 *  command the user configured is not it, whatever it runs. */
function isBazelCommand(argv) {
  const fallback = bazelCommand();
  return argv.length === fallback.length && argv.every((a, i) => a === fallback[i]);
}

/** Pick between them. `binary` is null until resolution finishes, and stays
 *  null in a workspace with no Bazel; both cases mean the slow path. */
function checkCommandFor(binary, root, builtAgainst = 0) {
  return binary && isFresh(binary, root, builtAgainst)
    ? directCommand(binary)
    : bazelCommand();
}

module.exports = {
  CHECKER_TARGET,
  CHECKER_BINARY_PATH,
  SOURCE_DIRS,
  SOURCE_EXTENSIONS,
  bazelCommand,
  checkCommandFor,
  checkerBinary,
  directCommand,
  isBazelCommand,
  isFresh,
  newestSourceMtime,
};
