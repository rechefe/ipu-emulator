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
  for (const [file, text] of [['MODULE.bazel', ''], ['src/tools/ipu-as/BUILD.bazel', ''], ['k/demo/demo.asm', 'HALT;;\n']]) {
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
  const api = await ext.activate();
  ext.open(fake.Uri.file(path.join(root, 'k/demo/demo.asm')), 'HALT;;\n');
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
  await until(() => ext.statusItem.text === '$(check) IPU (bundled ISA)', 'the checker to report');
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
async function window(...names) {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-activation-'));
  const roots = names.map((name) => checkout(temp, name));
  const check = recordingChecker(temp);
  const ext = fake.install({ folders: roots, settings: { ...settings, checkCommand: check.argv } });
  const api = await ext.activate();
  const kernel = (root) => fake.Uri.file(path.join(root, 'k/demo/demo.asm'));
  const done = () => {
    ext.deactivate();
    fs.rmSync(temp, { recursive: true, force: true });
  };
  return { ext, api, roots, check, kernel, done };
}

test('a check that failed is run again for the same text, once the toolchain is back', async () => {
  const w = await window('repo');
  fs.writeFileSync(path.join(w.roots[0], 'broken'), '');
  const doc = w.ext.open(w.kernel(w.roots[0]), 'HALT;;\n');
  await until(() => w.api.checkerError(), 'the check to fail');
  assert.match(w.ext.statusItem.text, /checks off/);
  fs.rmSync(path.join(w.roots[0], 'broken'));
  w.ext.events.onDidSaveTextDocument.fire(doc); // the same version, saved
  await until(() => w.check.runs().length === 2 && w.ext.statusItem.text === '$(check) IPU (bundled ISA)', 'the same version to be checked again');
  assert.strictEqual(w.api.checkerError(), null);
  // Once it has succeeded, that version is not checked again.
  w.ext.events.onDidSaveTextDocument.fire(doc);
  await sleep(300);
  assert.strictEqual(w.check.runs().length, 2);
  w.done();
});

test('each checkout has its own status: a broken toolchain in one does not show for another', async () => {
  const w = await window('a', 'b');
  const [a, b] = w.roots;
  fs.writeFileSync(path.join(a, 'broken'), '');
  const docA = w.ext.open(w.kernel(a), 'HALT;;\n');
  await until(() => w.api.checkerError(docA.uri), 'the check in a to fail');
  const docB = w.ext.open(w.kernel(b), 'HALT;;\n');
  await until(() => w.ext.statusItem.text === '$(check) IPU (bundled ISA)', 'b to check');
  assert.deepStrictEqual(w.check.runs(), [a, b], 'each checked in its own checkout');
  assert.strictEqual(w.api.checkerError(docB.uri), null);
  assert.match(w.api.checkerError(docA.uri), /toolchain broken/);
  w.ext.show(docA);
  assert.match(w.ext.statusItem.text, /checks off/);
  w.ext.show(docB);
  assert.strictEqual(w.ext.statusItem.text, '$(check) IPU (bundled ISA)');
  assert.strictEqual(w.ext.log.warnings.length, 1, 'one pop-up, for the broken checkout');
  w.done();
});

test('an unsaved buffer is checked in the window\'s only checkout, and in none when there are two', async () => {
  const one = await window('repo');
  one.ext.open(new fake.Uri('untitled', 'Untitled-1'), 'HALT;;\n');
  await until(() => one.check.runs().length, 'the unsaved buffer to be checked');
  assert.deepStrictEqual(one.check.runs(), one.roots);
  one.done();

  const two = await window('a', 'b');
  await until(() => two.api.kernels.indexes.size === 2, 'both checkouts');
  two.ext.open(new fake.Uri('untitled', 'Untitled-1'), 'HALT;;\n');
  two.ext.open(new fake.Uri('vscode-vfs', '/a/k/demo/demo.asm'), 'HALT;;\n'); // no file on disk
  await sleep(500);
  assert.deepStrictEqual(two.check.runs(), [], 'nothing checked');
  // A git diff view's side is the working-tree file's: checked in its own checkout.
  two.ext.open(new fake.Uri('git', path.join(two.roots[1], 'k/demo/demo.asm')), 'HALT;;\n');
  await until(() => two.check.runs().length, 'the git side to be checked');
  assert.deepStrictEqual(two.check.runs(), [two.roots[1]]);
  two.done();
});

test('activating, finding the checkout, loading its kernels and the first check touch the disk only asynchronously', async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-activation-'));
  const root = checkout(temp, 'repo');
  const ext = fake.install({ folders: [root], settings });
  const doc = { uri: fake.Uri.file(path.join(root, 'k/demo/demo.asm')), languageId: 'ipu-asm', version: 1, isClosed: false, getText: () => 'HALT;;\n' };
  ext.vscode.workspace.textDocuments.push(doc);
  ext.show(doc);
  // Every synchronous fs call throws, and is noted in case the code swallows it.
  const calls = [];
  const real = {};
  for (const name of Object.keys(fs).filter((n) => n.endsWith('Sync'))) {
    real[name] = fs[name];
    fs[name] = (...args) => {
      calls.push(`${name}(${String(args[0])})`);
      throw new Error(`fs.${name} during activation`);
    };
  }
  let api;
  try {
    api = await ext.activate();
    await until(() => api.kernels.indexes.get(root) && api.kernels.indexes.get(root).manifest, 'the kernels to load');
    await until(() => ext.statusItem.text === '$(check) IPU (bundled ISA)', 'the first check');
  } finally {
    Object.assign(fs, real);
  }
  assert.deepStrictEqual(calls, []);
  assert(api.editing === null || api.editing.gapLines, 'the editing help registered, when there is language data');
  ext.deactivate();
  fs.rmSync(temp, { recursive: true, force: true });
});

test('the checker binary is used while its sources are unchanged; a change to them, in a checkout opened by a subfolder, goes back through Bazel', async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-activation-'));
  const root = checkout(temp, 'repo');
  const source = path.join(root, 'src/tools/ipu-as/src/ipu_as/parser.py');
  fs.mkdirSync(path.dirname(source), { recursive: true });
  fs.writeFileSync(source, '');
  fs.utimesSync(source, new Date(Date.now() - 60000), new Date(Date.now() - 60000));
  // A Bazel that builds a checker binary, and runs (logging which ran, to `log`). Both are
  // assemblers from before contract versions: asked for theirs (logged to `probes`), they exit 2.
  const log = path.join(temp, 'ran.log');
  const probes = path.join(temp, 'probes.log');
  const old = (name) => `case "$*" in *--json-version*) echo ${name} >> ${probes}; exit 2 ;; esac\n`;
  const bazelBin = path.join(temp, 'bazel-bin');
  const binary = path.join(bazelBin, 'src/tools/ipu-as/ipu-as');
  fs.mkdirSync(path.dirname(binary), { recursive: true });
  fs.writeFileSync(binary, `#!/bin/sh\n${old('direct')}echo direct >> ${log}\nprintf '[]'\n`, { mode: 0o755 });
  const standIn = path.join(bin, 'bazel');
  const failing = fs.readFileSync(standIn, 'utf8');
  // The language data generator (languages.js) also goes through `bazel run`: not a check.
  fs.writeFileSync(standIn, `#!/bin/sh\ncase "$*" in *vscode-data*) exit 1 ;; esac\ncase "$1" in\n  build) exit 0 ;;\n  info) echo ${bazelBin} ;;\n  run) ${old('bazel-run')}echo bazel-run >> ${log}; printf '[]' ;;\nesac\n`);
  const ran = () => (fs.existsSync(log) ? fs.readFileSync(log, 'utf8').split('\n').filter(Boolean) : []);
  try {
    // The open folder is a subfolder of the checkout: the assembler's sources are outside it.
    const ext = fake.install({ folders: [path.join(root, 'k')], settings: { manifestCommand: settings.manifestCommand } });
    const api = await ext.activate();
    const doc = ext.open(fake.Uri.file(path.join(root, 'k/demo/demo.asm')), 'HALT;;\n');
    await until(() => ran().length, 'the first check');
    const check = async (n) => {
      await ext.vscode.commands.executeCommand('ipuAsm.check');
      await until(() => ran().length === n, `check ${n}`);
      return ran()[n - 1];
    };
    await until(async () => (await check(ran().length + 1)) === 'direct', 'the built binary to be used');
    const before = ran().length;
    fs.utimesSync(source, new Date(), new Date()); // an edit, after that build
    assert.strictEqual(await check(before + 1), 'direct', 'until a watcher reports it, the answer is remembered');
    const watcher = ext.log.watchers.find((w) => w.pattern.baseUri && w.pattern.baseUri.fsPath === path.join(root, 'src/tools/ipu-as/src'));
    assert(watcher, 'the checkout\'s own sources are watched, though they are outside the open folder');
    watcher.fire('Change', fake.Uri.file(source));
    assert.strictEqual(await check(before + 2), 'bazel-run', 'the changed source sends the check through Bazel');
    assert.strictEqual(api.checkerError(doc.uri), null);
    // Each command was asked its contract versions once, however often it checked, and an
    // assembler from before versions is no failure.
    const asked = fs.readFileSync(probes, 'utf8').split('\n').filter(Boolean);
    assert.strictEqual(asked.filter((p) => p === 'direct').length, 1, asked.join());
    assert.deepStrictEqual(ext.log.warnings, []);
    ext.deactivate();
  } finally {
    fs.writeFileSync(standIn, failing);
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

main((n) => `activation.js: ${n} cases pass.`);
