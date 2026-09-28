'use strict';

// Runs inside VS Code (see run.js) once per IPU_TEST_LAYOUT, a way an open folder can relate to a
// checkout: `parent` (holds it in work/ipu-emulator), `other` (another project with its own .asm and
// .s), `old` (a checkout from before the kernel manifest), `switch` (switched between versions while open).

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vscode = require('vscode');

const { test, sleep, until, runAll, count } = require('../harness');

const layout = process.env.IPU_TEST_LAYOUT;
const workspace = process.env.IPU_TEST_WORKSPACE;
const temp = process.env.IPU_TEST_TEMP;
const kernelPath = 'kernels/demo/demo_kernel/demo_kernel.asm';

const extension = () => vscode.extensions.getExtension('yardencarmi.ipu-asm');
const api = () => extension().exports;
const check = () => vscode.commands.executeCommand('ipuAsm.check');
/** Open `file`, activate the extension, and return its API. */
async function open(file) {
  const doc = await vscode.workspace.openTextDocument(file);
  await vscode.window.showTextDocument(doc);
  return { doc, api: await extension().activate() };
}
const lines = (file) => (fs.existsSync(file) ? fs.readFileSync(file, 'utf8').split('\n').filter(Boolean) : []);
const sidebarSays = async (re) => assert.match(String((await api().kernels.tree.getChildren())[0].item.label), re);

if (layout === 'parent') {
  const checkout = path.join(workspace, 'work', 'ipu-emulator');
  const kernelFile = path.join(checkout, kernelPath);

  test('a folder holding the checkout lists its kernels, and checks them in the checkout', async () => {
    const { doc, api } = await open(kernelFile);
    await until(() => api.kernels.kernelOfFile(doc.uri), 'the manifest to load');
    assert.deepStrictEqual([...api.kernels.indexes.keys()], [checkout]);
    assert.strictEqual(api.kernels.kernelOfFile(doc.uri).kernel, 'demo_kernel');
    assert.deepStrictEqual(api.kernels.editorContext.kernelFiles, [doc.uri.fsPath]);
    await check();
    const marker = path.join(temp, 'parent-checks');
    await until(() => lines(marker).length, 'a check to run');
    assert(lines(marker).every((cwd) => cwd === checkout), lines(marker).join(', '));
    assert.strictEqual(api.checkerError(), null);
    assert.deepStrictEqual(vscode.languages.getDiagnostics(doc.uri), []);
  });
}

if (layout === 'other') {
  test('another project\'s .asm is coloured, but nothing is run for it; its .s is no IPU assembly', async () => {
    const { doc, api } = await open(path.join(workspace, 'boot', 'start.asm'));
    assert.strictEqual(doc.languageId, 'ipu-asm');
    await check();
    await sleep(3000);
    assert.strictEqual(api.kernels.indexes.size, 0, 'no kernel index');
    assert.deepStrictEqual(lines(path.join(temp, 'other-manifest')), [], 'the manifest command never ran');
    assert.strictEqual(api.checkerError(), null, 'no checker ran, so none failed');
    assert.deepStrictEqual(vscode.languages.getDiagnostics(doc.uri), []);
    assert.deepStrictEqual(api.kernels.editorContext.kernelFiles, []);
    assert.strictEqual((await api.kernels.tree.getChildren()).length, 1);
    await sidebarSays(/No IPU checkout is open/);
    const s = await vscode.workspace.openTextDocument(path.join(workspace, 'boot', 'crt0.s'));
    assert.notStrictEqual(s.languageId, 'ipu-asm', 'a .s file is not taken for IPU assembly');
  });
}

if (layout === 'old') {
  test('a checkout from before the kernel manifest says so; its assembler still checks the file', async () => {
    const { api } = await open(path.join(workspace, kernelPath));
    const index = await until(() => {
      const [first] = api.kernels.indexes.values();
      return first && !first.loading && first.error ? first : null;
    }, 'the manifest to fail');
    assert.match(index.error, /predates it/);
    assert((await api.kernels.tree.getChildren()).some((r) => /predates it/.test(String(r.item.label) + String(r.item.tooltip || ''))), 'shown in the sidebar');
    await check();
    await sleep(2000);
    assert.strictEqual(api.checkerError(), null);
  });
}

if (layout === 'switch') {
  const kernelFile = path.join(workspace, kernelPath);
  const assembler = path.join(workspace, 'src', 'tools', 'ipu-as-py');
  const manifest = path.join(workspace, 'manifest.json');
  const saved = fs.readFileSync(manifest, 'utf8');
  const index = () => api().kernels.indexes.get(workspace);
  // One write can land before VS Code's file watcher is up: rewrite every few seconds until seen.
  const switched = async (cond, what) => {
    for (let i = 0; i < 6; i++) {
      fs.writeFileSync(path.join(workspace, 'BUILD.bazel'), `# switch ${i}\n`);
      const done = await until(cond, what, 4000).catch(() => null);
      if (done) return done;
    }
    throw new Error(`timed out waiting for ${what}`);
  };
  const kernelShown = async () => api().kernels.kernelOfFile((await vscode.workspace.openTextDocument(kernelFile)).uri);

  test('switched to a version from before the manifest or the Python assembler, and back', async () => {
    await open(kernelFile);
    await until(kernelShown, 'the kernels to load');
    fs.rmSync(manifest);
    await switched(() => index() && !index().loading && index().error, 'the manifest to fail');
    assert.match(index().error, /predates it/);
    assert.deepStrictEqual(api().kernels.editorContext.kernelFiles, []);
    fs.writeFileSync(manifest, saved);
    await switched(kernelShown, 'the kernels to return');
    // Switched to a version from before the Python assembler, it is no checkout; switched back, it is again.
    fs.rmSync(assembler, { recursive: true });
    await until(() => api().kernels.indexes.size === 0, 'the index to go', 20000);
    assert.deepStrictEqual(api().kernels.editorContext.kernelFiles, []);
    await sidebarSays(/No IPU checkout is open/);
    fs.mkdirSync(assembler, { recursive: true });
    fs.writeFileSync(path.join(assembler, 'BUILD.bazel'), '');
    await until(kernelShown, 'the kernels to load', 20000);
  });
}

async function run() {
  assert(count(), `no checks for layout ${layout}`);
  await runAll(`${layout}-layout`, `[${layout}] `);
}

module.exports = { run };
