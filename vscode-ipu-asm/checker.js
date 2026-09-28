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

const cp = require('child_process');
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

/** Keep Bazel's own chatter out of stdout, which carries the answer. */
const QUIET = ['--ui_event_filters=-info,-stdout,-stderr', '--noshow_progress'];

/** The slow path: correct whatever state the tree is in, and rebuilds. */
function bazelCommand() {
  return [
    'bazel',
    'run',
    ...QUIET,
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

/** The longest timer Node keeps: a longer one fires at once. */
const MAX_TIMER_SECONDS = 2147483;
/** How long a stopped process gets before SIGKILL, and an exited one for its pipes to drain. */
const GRACE_MS = 2000;

/** Spawn in a process group of its own (on POSIX), so `child.stop(signal)` also stops what
 *  it started: `bazel run`'s client and the checker it launched, or a wrapper's command. */
function spawnGroup(command, args, options = {}) {
  const posix = process.platform !== 'win32';
  const child = cp.spawn(command, args, { ...options, detached: posix });
  child.stop = (signal = 'SIGTERM') => {
    try {
      if (posix && child.pid) process.kill(-child.pid, signal);
      else child.kill(signal);
    } catch {
      /* already gone */
    }
  };
  return child;
}

/** Stop `child` (from spawnGroup) past `seconds` (<= 0 or beyond a timer: no limit) and mark
 *  it `timedOut`: a `bazel run` waiting on the workspace lock would otherwise hold a check,
 *  a rename or the manifest forever. SIGTERM is followed by SIGKILL after GRACE_MS, and once
 *  it exits, a descendant holding its pipes open (a daemon) is not waited for past that. */
function limit(child, seconds) {
  const timers = [];
  const later = (ms, fn) => timers.push(setTimeout(fn, ms));
  const closePipes = () => [child.stdin, child.stdout, child.stderr].forEach((stream) => stream && stream.destroy());
  if (seconds > 0 && seconds <= MAX_TIMER_SECONDS) {
    later(seconds * 1000, () => {
      child.timedOut = true;
      child.stop('SIGTERM');
      later(GRACE_MS, () => {
        child.stop('SIGKILL');
        closePipes();
      });
    });
  }
  for (const event of ['exit', 'close', 'error']) child.once(event, () => timers.forEach(clearTimeout));
  child.once('exit', () => later(GRACE_MS, closePipes));
}

/** What to tell the user when the checker fails: its own words, cut to what
 *  matters. A Python traceback is reported by its last line, the error. */
function explainFailure(stderr, command) {
  // The most likely cause by far: `check` is added by the extension's own
  // change, so a workspace on a branch predating it has an ipu-as without it.
  if (/no such command/i.test(stderr)) {
    return 'the assembler in this workspace has no "check" subcommand, so diagnostics cannot run. '
      + 'It is added alongside this extension — a branch predating that will not have it.';
  }
  const lines = String(stderr).trim().split('\n').map((l) => l.trim()).filter(Boolean);
  if (lines.some((l) => l.startsWith('Traceback (most recent call last)'))) {
    const error = lines[lines.length - 1];
    const module = /^ModuleNotFoundError: No module named '([^']+)'/.exec(error);
    if (module) {
      return `"${command}" runs a Python that cannot import ${module[1]}. Point ipuAsm.checkCommand at an `
        + 'environment with the assembler installed, or leave it empty to use the Bazel-built checker.';
    }
    return `"${command}" failed: ${error.slice(0, 300)}`;
  }
  if (/not found|ENOENT/i.test(stderr)) {
    return `"${command}" could not be run. Set ipuAsm.checkCommand to point at a working assembler.`;
  }
  const tail = lines.slice(-3).join(' ').slice(0, 300);
  return tail || `"${command}" produced no output. See the Developer Tools console.`;
}

/** Run to completion under `limit`: { code, stdout, stderr } whatever the code (-1 with
 *  `spawnError` or `timedOut`). `input` goes to stdin; `onSpawn(child)` allows stopping it. */
function run(command, args, cwd, seconds, { input, onSpawn } = {}) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawnGroup(command, args, { cwd });
    } catch (err) {
      resolve({ code: -1, stdout: '', stderr: err.message, spawnError: true });
      return;
    }
    limit(child, seconds);
    if (onSpawn) onSpawn(child);
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => (stdout += d));
    child.stderr.on('data', (d) => (stderr += d));
    child.stdin.on('error', () => {}); // a stopped child closes stdin first
    child.stdin.end(input);
    child.on('error', (err) => resolve({ code: -1, stdout, stderr: stderr + err.message, spawnError: true }));
    child.on('close', (code) => resolve(child.timedOut
      ? { code: -1, stdout: '', stderr: `${command} timed out after ${seconds}s (is another Bazel command holding the workspace lock?)`, timedOut: true }
      : { code, stdout, stderr }));
  });
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
  explainFailure,
  isBazelCommand,
  isFresh,
  limit,
  MAX_TIMER_SECONDS,
  QUIET,
  newestSourceMtime,
  run,
  spawnGroup,
};
