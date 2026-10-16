'use strict';

// The extension activated in plain node, on a fake `vscode` (fake-vscode.js): what it runs,
// when, and what it shows. VS Code trusts every workspace while it runs extension tests,
// so Restricted Mode is tested here rather than in integration/.
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

main((n) => `activation.js: ${n} cases pass.`);
