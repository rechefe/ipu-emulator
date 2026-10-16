'use strict';

// The extension activated in plain node, on a fake `vscode` (fake-vscode.js): what it runs,
// when, and what it shows. No API can grant workspace trust from inside VS Code, so
// Restricted Mode and leaving it are tested here rather than in integration/.
//
//   node test/activation.js

const assert = require('assert');
const cp = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const fake = require('./fake-vscode');
const { test, main, sleep, until } = require('./harness');

// A stand-in `bazel` that fails at once, so the checker's background build starts no real
// Bazel server in a temporary folder.
const bin = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-activation-bin-'));
fs.writeFileSync(path.join(bin, 'bazel'), '#!/bin/sh\nexit 1\n', { mode: 0o755 });
process.env.PATH = `${bin}${path.delimiter}${process.env.PATH}`;
process.on('exit', () => fs.rmSync(bin, { recursive: true, force: true }));

/** Every process started, as [command, args]; the real spawn still runs. */
const spawned = [];
for (const name of ['spawn', 'execFile']) {
  const real = cp[name];
  cp[name] = function spy(command, args, ...rest) {
    spawned.push([command, args]);
    return real.call(this, command, args, ...rest);
  };
}

/** An IPU checkout (a Bazel workspace holding the assembler's package) with one kernel. */
function checkout(parent, name) {
  const dir = path.join(parent, name);
  for (const [file, text] of [['MODULE.bazel', ''], ['src/tools/ipu-as-py/BUILD.bazel', ''], ['k/demo/demo.asm', 'BKPT;;\n']]) {
    fs.mkdirSync(path.dirname(path.join(dir, file)), { recursive: true });
    fs.writeFileSync(path.join(dir, file), text);
  }
  const manifest = {
    kernels: { demo: { op: 'demo', variant: '', tags: [], family: 'f', asm: 'k/demo/demo.asm', targets: { run: '//k:demo' }, cases: { default: { max_cycles: 1, options: {} } } } },
    families: { f: { suite: '//k:f', kernels: ['demo'] } }, operations: {}, query: '//k:query', skipped: [],
  };
  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest));
  return dir;
}

const node = (script) => [process.execPath, '-e', script];
const settings = {
  checkCommand: node("process.stdout.write('[]')"),
  manifestCommand: node("process.stdout.write(require('fs').readFileSync('manifest.json', 'utf8'))"),
};
const ran = (argv) => spawned.some(([command, args]) => command === argv[0] && args[1] === argv[2]);

test('in Restricted Mode nothing runs, and the status says why; trusting the workspace starts the checker and the kernels', async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-activation-'));
  const root = checkout(temp, 'repo');
  const ext = fake.install({ trusted: false, folders: [root], settings });
  spawned.length = 0;
  const api = ext.activate();
  ext.open(fake.Uri.file(path.join(root, 'k/demo/demo.asm')), 'BKPT;;\n');
  await ext.vscode.commands.executeCommand('ipuAsm.check');
  await ext.vscode.commands.executeCommand('ipuAsm.runKernel', 'demo');
  await sleep(500);
  assert.deepStrictEqual(spawned, [], 'no process started');
  assert.strictEqual(ext.log.terminals.length, 0, 'no terminal sent anything');
  assert.match(ext.statusItem.text, /IPU: Restricted Mode/);
  assert(ext.statusItem.visible);
  const index = api.kernels.indexes.get(root);
  assert.match(index.error, /Restricted Mode/);
  assert.strictEqual(api.checkerError(), null, 'Restricted Mode is not a broken checker');

  ext.trust();
  await until(() => ran(settings.checkCommand) && ran(settings.manifestCommand), 'the checker and the manifest to run', 10000);
  await until(() => ext.statusItem.text === '$(check) IPU', 'the checker to report');
  await until(() => index.manifest, 'the kernels to load');
  assert.strictEqual(index.error, null);
  ext.deactivate();
  fs.rmSync(temp, { recursive: true, force: true });
});

/** A checker that notes where it ran, and fails in a checkout holding a `broken` file. */
function recordingChecker(temp) {
  const log = path.join(temp, 'checks.log');
  const script = path.join(temp, 'checker.js');
  fs.writeFileSync(script, `const fs = require('fs');
fs.appendFileSync(${JSON.stringify(log)}, process.cwd() + '\\n');
if (fs.existsSync('broken')) { process.stderr.write('toolchain broken'); process.exit(2); }
process.stdout.write('[]');
`);
  const runs = () => (fs.existsSync(log) ? fs.readFileSync(log, 'utf8').split('\n').filter(Boolean) : []);
  return { argv: [process.execPath, script], runs };
}

/** A trusted window on checkouts `names` (in a fresh temporary folder) with the recording checker. */
function window(...names) {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-activation-'));
  const roots = names.map((name) => checkout(temp, name));
  const check = recordingChecker(temp);
  const ext = fake.install({ folders: roots, settings: { ...settings, checkCommand: check.argv } });
  const api = ext.activate();
  const kernel = (root) => fake.Uri.file(path.join(root, 'k/demo/demo.asm'));
  const done = () => {
    ext.deactivate();
    fs.rmSync(temp, { recursive: true, force: true });
  };
  return { ext, api, roots, check, kernel, done };
}

test('a check that failed is run again for the same text, once the toolchain is back', async () => {
  const w = window('repo');
  fs.writeFileSync(path.join(w.roots[0], 'broken'), '');
  const doc = w.ext.open(w.kernel(w.roots[0]), 'BKPT;;\n');
  await until(() => w.api.checkerError(), 'the check to fail');
  assert.match(w.ext.statusItem.text, /checks off/);
  fs.rmSync(path.join(w.roots[0], 'broken'));
  w.ext.events.onDidSaveTextDocument.fire(doc); // the same version, saved
  await until(() => w.check.runs().length === 2 && w.ext.statusItem.text === '$(check) IPU', 'the same version to be checked again');
  assert.strictEqual(w.api.checkerError(), null);
  // Once it has succeeded, that version is not checked again.
  w.ext.events.onDidSaveTextDocument.fire(doc);
  await sleep(300);
  assert.strictEqual(w.check.runs().length, 2);
  w.done();
});

test('each checkout has its own status: a broken toolchain in one does not show for another', async () => {
  const w = window('a', 'b');
  const [a, b] = w.roots;
  fs.writeFileSync(path.join(a, 'broken'), '');
  const docA = w.ext.open(w.kernel(a), 'BKPT;;\n');
  await until(() => w.api.checkerError(docA.uri), 'the check in a to fail');
  const docB = w.ext.open(w.kernel(b), 'BKPT;;\n');
  await until(() => w.ext.statusItem.text === '$(check) IPU', 'b to check');
  assert.deepStrictEqual(w.check.runs(), [a, b], 'each checked in its own checkout');
  assert.strictEqual(w.api.checkerError(docB.uri), null);
  assert.match(w.api.checkerError(docA.uri), /toolchain broken/);
  w.ext.show(docA);
  assert.match(w.ext.statusItem.text, /checks off/);
  w.ext.show(docB);
  assert.strictEqual(w.ext.statusItem.text, '$(check) IPU');
  assert.strictEqual(w.ext.log.warnings.length, 1, 'one pop-up, for the broken checkout');
  w.done();
});

test('an unsaved buffer is checked in the window\'s only checkout, and in none when there are two', async () => {
  const one = window('repo');
  one.ext.open(new fake.Uri('untitled', 'Untitled-1'), 'BKPT;;\n');
  await until(() => one.check.runs().length, 'the unsaved buffer to be checked');
  assert.deepStrictEqual(one.check.runs(), one.roots);
  one.done();

  const two = window('a', 'b');
  await until(() => two.api.kernels.indexes.size === 2, 'both checkouts');
  two.ext.open(new fake.Uri('untitled', 'Untitled-1'), 'BKPT;;\n');
  two.ext.open(new fake.Uri('vscode-vfs', '/a/k/demo/demo.asm'), 'BKPT;;\n'); // no file on disk
  await sleep(500);
  assert.deepStrictEqual(two.check.runs(), [], 'nothing checked');
  // A git diff view's side is the working-tree file's: checked in its own checkout.
  two.ext.open(new fake.Uri('git', path.join(two.roots[1], 'k/demo/demo.asm')), 'BKPT;;\n');
  await until(() => two.check.runs().length, 'the git side to be checked');
  assert.deepStrictEqual(two.check.runs(), [two.roots[1]]);
  two.done();
});

main((n) => `activation.js: ${n} cases pass.`);
