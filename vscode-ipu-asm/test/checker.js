'use strict';

// The rule that decides which checker runs, tested without an editor.
//
// The interesting case is not "does it pick the fast one", it is "does it stop
// picking the fast one the moment the assembler changes underneath it". Stale
// diagnostics look exactly like fresh ones on screen, so nothing but this test
// stands between a rebuilt assembler and squiggles describing the old one.
//
//   node vscode-ipu-asm/test/checker.js

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const checker = require('../checker');

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

/** A workspace holding one source file and one built binary, both datable. */
function workspace() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-checker-'));
  const sourceDir = path.join(root, checker.SOURCE_DIRS[0], 'ipu_as');
  fs.mkdirSync(sourceDir, { recursive: true });
  const source = path.join(sourceDir, 'diagnostics.py');
  const binary = path.join(root, 'ipu-as');
  fs.writeFileSync(source, '# assembler\n');
  fs.writeFileSync(binary, '#!/usr/bin/env python3\n');

  const at = (file, secondsFromNow) => {
    const when = new Date(Date.now() + secondsFromNow * 1000);
    fs.utimesSync(file, when, when);
  };
  return { root, source, binary, at };
}

test('a binary newer than its sources is used directly', () => {
  const w = workspace();
  w.at(w.source, -60);
  w.at(w.binary, 0);
  assert.deepStrictEqual(
    checker.checkCommandFor(w.binary, w.root),
    checker.directCommand(w.binary)
  );
});

test('an edited source sends the next check back through Bazel', () => {
  const w = workspace();
  w.at(w.binary, -60);
  w.at(w.source, 0); // the edit, after the build
  assert.deepStrictEqual(checker.checkCommandFor(w.binary, w.root), checker.bazelCommand());
});

test('a source of the same age still counts as built', () => {
  // A build writes its output after reading its inputs, and a filesystem with
  // coarse timestamps reports both at the same tick. Treating equal as stale
  // would send every check through Bazel on such a filesystem.
  const w = workspace();
  w.at(w.source, 0);
  w.at(w.binary, 0);
  assert.deepStrictEqual(
    checker.checkCommandFor(w.binary, w.root),
    checker.directCommand(w.binary)
  );
});

test('a deleted binary falls back rather than failing', () => {
  const w = workspace();
  fs.rmSync(w.binary); // `bazel clean`
  assert.deepStrictEqual(checker.checkCommandFor(w.binary, w.root), checker.bazelCommand());
});

test('an unresolved checker falls back', () => {
  const w = workspace();
  assert.deepStrictEqual(checker.checkCommandFor(null, w.root), checker.bazelCommand());
});

test('a workspace without the assembler has nothing to be stale against', () => {
  // Someone opens a folder of kernels rather than this repo: there are no
  // sources to compare, so a resolved binary is as good as it gets.
  const w = workspace();
  fs.rmSync(path.join(w.root, 'src'), { recursive: true });
  w.at(w.binary, -3600);
  assert.strictEqual(checker.newestSourceMtime(w.root), 0);
  assert.deepStrictEqual(
    checker.checkCommandFor(w.binary, w.root),
    checker.directCommand(w.binary)
  );
});

test('every source extension and directory is watched for staleness', () => {
  const w = workspace();
  w.at(w.binary, 0);
  for (const dir of checker.SOURCE_DIRS) {
    for (const ext of checker.SOURCE_EXTENSIONS) {
      const nested = path.join(w.root, dir, 'nested');
      fs.mkdirSync(nested, { recursive: true });
      const file = path.join(nested, `probe${ext}`);
      fs.writeFileSync(file, 'x');
      w.at(file, 60);
      assert.deepStrictEqual(
        checker.checkCommandFor(w.binary, w.root),
        checker.bazelCommand(),
        `${dir} + ${ext} did not invalidate the binary`
      );
      fs.rmSync(file);
    }
  }
});

test('a re-dated but unchanged source does not pin every check to Bazel', () => {
  // The case that made the first version of this wrong. Bazel rebuilds on
  // content, so a `git checkout` or a `touch` re-dates the sources and Bazel
  // then legitimately does nothing: the binary's mtime stays behind forever.
  // Comparing mtimes alone called it stale on every check from then on.
  const w = workspace();
  w.at(w.binary, -60);
  w.at(w.source, 0); // re-dated, same bytes
  assert.deepStrictEqual(
    checker.checkCommandFor(w.binary, w.root),
    checker.bazelCommand(),
    'the first check after a re-date should rebuild'
  );

  // That run records what it built against, and the next check is fast again
  // even though Bazel never rewrote the binary.
  const builtAgainst = checker.newestSourceMtime(w.root);
  assert.deepStrictEqual(
    checker.checkCommandFor(w.binary, w.root, builtAgainst),
    checker.directCommand(w.binary)
  );
});

test('a real edit after a recorded build still goes back through Bazel', () => {
  const w = workspace();
  w.at(w.binary, -60);
  w.at(w.source, -30);
  const builtAgainst = checker.newestSourceMtime(w.root);
  w.at(w.source, 0); // the edit, after that build
  assert.deepStrictEqual(
    checker.checkCommandFor(w.binary, w.root, builtAgainst),
    checker.bazelCommand()
  );
});

test('the fallback is recognised, a configured command is not', () => {
  // Only the fallback rebuilds, so only the fallback may update the record.
  assert.ok(checker.isBazelCommand(checker.bazelCommand()));
  assert.ok(!checker.isBazelCommand(checker.directCommand('/x')));
  assert.ok(!checker.isBazelCommand(['bazel', 'run', '//other:target']));
  const configured = checker.bazelCommand();
  configured[configured.length - 1] = '--file';
  assert.ok(!checker.isBazelCommand(configured));
});

test('the binary is where bazel-bin says, with a Windows suffix there', () => {
  assert.strictEqual(
    checker.checkerBinary('/out/bin', 'linux'),
    path.join('/out/bin', checker.CHECKER_BINARY_PATH)
  );
  assert.ok(checker.checkerBinary('/out/bin', 'win32').endsWith('.exe'));
});

test('the Bazel fallback silences the client so stdout stays pure JSON', () => {
  const command = checker.bazelCommand();
  assert.ok(command.includes('--ui_event_filters=-info,-stdout,-stderr'));
  assert.ok(command.includes('--noshow_progress'));
  assert.ok(command.includes(checker.CHECKER_TARGET));
});

test('both commands end at --input, so the caller appends the source', () => {
  assert.strictEqual(checker.bazelCommand().at(-1), '--input');
  assert.strictEqual(checker.directCommand('/x').at(-1), '--input');
});

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
  } catch (error) {
    failed += 1;
    console.error(`FAIL ${name}\n  ${error.message}`);
  }
}
if (failed) {
  console.error(`${failed} of ${tests.length} checker tests failed.`);
  process.exit(1);
}
console.log(`Checker selection agrees with the tree: ${tests.length} cases.`);
