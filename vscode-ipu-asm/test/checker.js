'use strict';

// Which checker runs, and the time limit it runs under, tested without an editor. The interesting case:
// it stops picking the fast one the moment the assembler changes, for stale diagnostics look fresh.
//   node vscode-ipu-asm/test/checker.js

const assert = require('assert');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const checker = require('../checker');
const { test, main } = require('./harness');

/** A workspace holding one source file and one built binary, both datable. */
function workspace() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-checker-'));
  const source = path.join(root, checker.SOURCE_DIRS[0], 'ipu_as', 'diagnostics.py');
  const binary = path.join(root, 'ipu-as');
  fs.mkdirSync(path.dirname(source), { recursive: true });
  fs.writeFileSync(source, '# assembler\n');
  fs.writeFileSync(binary, '#!/usr/bin/env python3\n');
  /** Date `file` `secondsFromNow` (negative: in the past). */
  const at = (file, secondsFromNow) => fs.utimesSync(file, Date.now() / 1000 + secondsFromNow, Date.now() / 1000 + secondsFromNow);
  return { root, source, binary, at };
}
/** Assert what `checkCommandFor` picks in `w`: 'direct' (the binary) or 'bazel'. */
function picks(w, expected, builtAgainst, message) {
  const want = { direct: checker.directCommand(w.binary), bazel: checker.bazelCommand() }[expected];
  assert.deepStrictEqual(checker.checkCommandFor(w.binary, w.root, builtAgainst), want, message);
}

for (const [what, setup, expected] of [
  ['a binary newer than its sources is used directly', (w) => { w.at(w.source, -60); w.at(w.binary, 0); }, 'direct'],
  ['an edited source sends the next check back through Bazel', (w) => { w.at(w.binary, -60); w.at(w.source, 0); }, 'bazel'],
  // A build writes its output after reading its inputs, and a filesystem with coarse timestamps
  // reports both at the same tick: equal as stale would send every check through Bazel there.
  ['a source of the same age still counts as built', (w) => { w.at(w.source, 0); w.at(w.binary, 0); }, 'direct'],
  ['a deleted binary (`bazel clean`) falls back rather than failing', (w) => fs.rmSync(w.binary), 'bazel'],
  ['an unresolved checker falls back', (w) => { w.binary = null; }, 'bazel'],
  // A folder of kernels rather than this repo: no sources to compare, so a resolved binary is as good as it gets.
  ['a workspace without the assembler has nothing to be stale against', (w) => {
    fs.rmSync(path.join(w.root, 'src'), { recursive: true });
    w.at(w.binary, -3600);
    assert.strictEqual(checker.newestSourceMtime(w.root), 0);
  }, 'direct'],
]) test(what, () => { const w = workspace(); setup(w); picks(w, expected); });

test('every source extension and directory is watched for staleness', () => {
  const w = workspace();
  w.at(w.binary, 0);
  for (const dir of checker.SOURCE_DIRS) {
    fs.mkdirSync(path.join(w.root, dir, 'nested'), { recursive: true });
    for (const ext of checker.SOURCE_EXTENSIONS) {
      const file = path.join(w.root, dir, 'nested', `probe${ext}`);
      fs.writeFileSync(file, 'x');
      w.at(file, 60);
      picks(w, 'bazel', undefined, `${dir} + ${ext} did not invalidate the binary`);
      fs.rmSync(file);
    }
  }
});

test('a re-dated but unchanged source does not pin every check to Bazel; a real edit does', () => {
  // A `git checkout` or `touch` re-dates the sources and Bazel, rebuilding on content, does nothing:
  // the binary stays behind. The first check rebuilds and records what it built against; the next is fast.
  const w = workspace();
  w.at(w.binary, -60);
  w.at(w.source, 0); // re-dated, same bytes
  picks(w, 'bazel', undefined, 'the first check after a re-date should rebuild');
  picks(w, 'direct', checker.newestSourceMtime(w.root));
  const edited = workspace();
  edited.at(edited.binary, -60);
  edited.at(edited.source, -30);
  const builtAgainst = checker.newestSourceMtime(edited.root);
  edited.at(edited.source, 0); // the edit, after that build
  picks(edited, 'bazel', builtAgainst);
});

test('the commands: the fallback alone is recognised, both end at --input, the binary has a Windows suffix', () => {
  // Only the fallback rebuilds, so only the fallback may update the record.
  const configured = checker.bazelCommand().slice(0, -1).concat('--file');
  const recognised = [checker.bazelCommand(), checker.directCommand('/x'), ['bazel', 'run', '//other:target'], configured];
  assert.deepStrictEqual(recognised.map(checker.isBazelCommand), [true, false, false, false]);
  // The Bazel fallback silences the client so stdout stays pure JSON.
  const command = checker.bazelCommand();
  for (const arg of ['--ui_event_filters=-info,-stdout,-stderr', '--noshow_progress', checker.CHECKER_TARGET]) assert.ok(command.includes(arg), arg);
  assert.strictEqual(command.at(-1), '--input');
  assert.strictEqual(checker.directCommand('/x').at(-1), '--input');
  assert.strictEqual(checker.checkerBinary('/out/bin', 'linux'), path.join('/out/bin', checker.CHECKER_BINARY_PATH));
  assert.ok(checker.checkerBinary('/out/bin', 'win32').endsWith('.exe'));
});

test('a Python traceback is reported by its error, a missing module with the fix', () => {
  const traceback = 'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\nModuleNotFoundError: No module named \'ipu_as\'\n';
  const message = checker.explainFailure(traceback, '/venv/bin/python');
  assert(!message.includes('Traceback'), message);
  assert(message.includes('cannot import ipu_as') && message.includes('ipuAsm.checkCommand'), message);
  assert.strictEqual(checker.explainFailure('Traceback (most recent call last):\n  File "x", line 2\nValueError: bad input\n', 'ipu-as'),
    '"ipu-as" failed: ValueError: bad input');
  assert(checker.explainFailure('Error: no such command "check"', 'ipu-as').includes('no "check" subcommand'));
  assert(checker.explainFailure('', 'ipu-as').includes('produced no output'));
});

/** Spawn `argv` in its own group under `checker.limit(seconds)`; resolves on close. */
async function limited([cmd, ...args], seconds, onSpawn = () => {}) {
  const child = checker.spawnGroup(cmd, args);
  onSpawn(child);
  const started = Date.now();
  checker.limit(child, seconds);
  await new Promise((resolve) => child.on('close', resolve));
  return { child, took: Date.now() - started };
}
/** Whether any process of the group led by `pid` runs; a zombie (some containers' init never reaps) does not. */
const groupAlive = (pid) => {
  try {
    return execFileSync('ps', ['-o', 'stat=', '-g', String(pid)], { encoding: 'utf8' }).split('\n').some((stat) => stat.trim() && !stat.trim().startsWith('Z'));
  } catch {
    return false; // ps exits 1 when the group is empty
  }
};

// [what, argv, limit in seconds, child.timedOut, closed within ms]
for (const [what, argv, seconds, timedOut, within] of [
  ['a process past its time limit is stopped and marked', ['sleep', '5'], 0.3, true, 2000],
  ['stopping a wrapper stops the command it started too', ['sh', '-c', 'sleep 5; :'], 0.3, true, 2000],
  ['a process that ignores SIGTERM gets SIGKILL', ['sh', '-c', 'trap "" TERM; sleep 8; :'], 0.3, true, 4000],
  ['a process that ends in time is left alone', ['true'], 5, undefined, Infinity],
  ...[0, checker.MAX_TIMER_SECONDS + 1, 1e9].map((s) => [`no limit when the setting is ${s}`, ['sleep', '0.3'], s, undefined, Infinity]),
]) {
  test(what, async () => {
    const { child, took } = await limited(argv, seconds);
    assert.strictEqual(child.timedOut, timedOut);
    assert(took < within, `closed after ${took}ms`);
    if (!timedOut) return;
    await new Promise((r) => setTimeout(r, 100));
    // `ps -g` selects a process group only on Linux (procps), where CI runs.
    if (process.platform === 'linux') assert(!groupAlive(child.pid), 'what it started is gone as well');
  });
}

test('a process that exits leaving a daemon on its pipes still closes, with its output', async () => {
  let stdout = '';
  const { child, took } = await limited(['sh', '-c', 'sleep 6 & echo ok'], 30, (c) => c.stdout.on('data', (d) => (stdout += d)));
  child.stop('SIGKILL'); // the daemon, for this test's own tidiness
  assert.strictEqual(child.timedOut, undefined);
  assert.strictEqual(stdout.trim(), 'ok');
  assert(took < 4000, `closed after ${took}ms`);
});

main((n) => `Checker selection agrees with the tree: ${n} cases.`);
