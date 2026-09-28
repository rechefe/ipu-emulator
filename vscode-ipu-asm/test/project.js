'use strict';

// project.js: which IPU checkout a file or an open folder belongs to.
//   node test/project.js

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const project = require('../project');
const { test, main } = require('./harness');

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-project-'));
/** Make `files` (paths relative to `base`; a trailing / makes a directory). */
function tree(base, files) {
  for (const f of files) {
    fs.mkdirSync(path.join(base, f.endsWith('/') ? f : path.dirname(f)), { recursive: true });
    if (!f.endsWith('/')) fs.writeFileSync(path.join(base, f), '');
  }
  return base;
}
const checkout = (base) => tree(base, ['MODULE.bazel', 'src/tools/ipu-as-py/BUILD.bazel', 'kernels/demo/k/k.asm']);

test('a file inside a checkout belongs to it, however deep; a subfolder stands for it', () => {
  const repo = checkout(path.join(tmp, 'a', 'repo'));
  assert.strictEqual(project.checkoutOf(path.join(repo, 'kernels/demo/k/k.asm')), repo);
  assert.strictEqual(project.checkoutOf(repo), repo);
  assert.deepStrictEqual(project.checkoutsFor(path.join(repo, 'kernels')), [repo]);
});

test('a folder holding checkouts stands for each of them', () => {
  const parent = path.join(tmp, 'c');
  const one = checkout(path.join(parent, 'one'));
  const two = checkout(path.join(parent, 'work', 'two'));
  tree(parent, ['node_modules/x/MODULE.bazel', 'node_modules/x/src/tools/ipu-as-py/']);
  assert.deepStrictEqual(project.checkoutsFor(parent), [one, two]);
});

test('a checkout linked into a folder is found through the link', () => {
  const repo = checkout(path.join(tmp, 'g', 'real', 'repo'));
  const parent = path.join(tmp, 'g', 'parent');
  fs.mkdirSync(parent, { recursive: true });
  fs.symlinkSync(repo, path.join(parent, 'linked'));
  fs.symlinkSync(parent, path.join(parent, 'loop'));
  assert.deepStrictEqual(project.checkoutsFor(parent), [path.join(parent, 'linked')]);
});

test('an unrelated project, even a Bazel one with .asm files, or a version before the Python assembler is no checkout', () => {
  for (const [dir, files, asm] of [
    ['x86', ['MODULE.bazel', 'boot/start.asm', 'lib/crt0.s'], 'boot/start.asm'],
    ['plain', ['main.asm'], 'main.asm'],
    ['old', ['WORKSPACE', 'src/tools/ipu-as/', 'app.asm'], 'app.asm'],
  ]) {
    const root = tree(path.join(tmp, 'd', dir), files);
    assert.strictEqual(project.checkoutOf(path.join(root, asm)), null, dir);
    assert.deepStrictEqual(project.checkoutsFor(root), [], dir);
  }
});

test('a checkout too deep below the open folder is not searched for; this repository is a checkout', () => {
  checkout(path.join(tmp, 'f', '1', '2', '3', 'repo'));
  assert.deepStrictEqual(project.checkoutsFor(path.join(tmp, 'f')), []);
  assert.strictEqual(project.checkoutOf(__filename), path.resolve(__dirname, '..', '..'));
});

test('(clean up)', () => fs.rmSync(tmp, { recursive: true, force: true }));
main((n) => `project.js: ${n - 1} cases pass.`);
