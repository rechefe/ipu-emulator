'use strict';

// runs.js: command lines built from a kernel manifest, and whether a terminal is free. Given a real
// manifest (`bazel run :kernel_manifest`), every kernel in it also maps back from its .asm and builds its commands.
//
//   node test/runs.js [manifest.json]

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const runs = require('../runs');
const { test, main } = require('./harness');

const kernel = {
  op: 'identity', variant: '', tags: [], family: 'reshape',
  asm: 'src/tools/ipu-apps/src/ipu_apps/kernels/reshape/identity/identity.asm',
  targets: { run: '//src/tools/ipu-apps:identity', test: '//src/tools/ipu-apps:identity' },
  cases: {
    default: { max_cycles: 1000, options: {
      rows: { flag: '--rows', default: 4, type: 'int' },
      label: { flag: '--label', default: "it's", type: 'str' },
      wide: { flag: '--wide', default: false, type: 'bool', negated_flag: '--no-wide' },
    } },
  },
};
const manifest = {
  kernels: { identity: kernel },
  families: { reshape: { suite: '//src/tools/ipu-apps:reshape', kernels: ['identity'] } },
  operations: { identity: { params: ['shape'] } }, query: '//src/tools/ipu-apps:query', skipped: [],
};

test('kernel commands: default options as flags, case and options after --, benchmark only if listed', () => {
  assert.strictEqual(runs.optionArgs(kernel.cases.default), "--rows 4 --label 'it'\\''s' --no-wide");
  const bench = { ...kernel, targets: { ...kernel.targets, benchmark: '//x:benchmark_identity' } };
  for (const [k, mode, options, command] of [
    [kernel, 'run', { caseName: 'default', options: '--rows 8' }, 'bazel run //src/tools/ipu-apps:identity -- --case default --rows 8'],
    [kernel, 'debug', { caseName: 'default', flags: ['--define=ipu_proto=1'] },
      'bazel run --config=debug --define=ipu_proto=1 //src/tools/ipu-apps:identity -- --case default'],
    [kernel, 'run', { caseName: 'a b' }, "bazel run //src/tools/ipu-apps:identity -- --case 'a b'"], // shell characters quoted
    [kernel, 'test', undefined, 'bazel test //src/tools/ipu-apps:identity'],
    [kernel, 'benchmark', undefined, null],
    [bench, 'benchmark', undefined, 'bazel run //x:benchmark_identity'],
  ]) assert.strictEqual(runs.kernelCommand(k, mode, options), command, `${mode} ${JSON.stringify(options)}`);
});

test('cases: saved ones added or over the registry\'s, run as their base with their values; values checked by type', () => {
  const cases = runs.kernelCases(kernel, {
    tall: { base: 'default', options: { rows: 64, wide: true } },
    default: { options: { label: 'x' } }, // a registry case's name: that case, edited
    orphan: { base: 'gone', options: {} }, // its base left the registry
  });
  assert.deepStrictEqual(Object.keys(cases), ['default', 'tall']);
  assert.deepStrictEqual([cases.default.base, cases.default.saved, cases.default.builtIn], ['default', true, true]);
  assert.deepStrictEqual([cases.tall.base, cases.tall.saved, cases.tall.builtIn], ['default', true, false]);
  assert.strictEqual(runs.optionArgs(cases.tall.entry, cases.tall.values), "--rows 64 --label 'it'\\''s' --wide");
  assert.strictEqual(runs.optionArgs(cases.default.entry, cases.default.values), '--rows 4 --label x --no-wide');
  const entry = kernel.cases.default;
  assert.strictEqual(runs.invalidValues(entry, { rows: 3, label: 'a b', wide: true }), null);
  assert.match(runs.invalidValues(entry, { rows: 2.5 }), /rows must be a whole number/);
  assert.match(runs.invalidValues(entry, { rows: null }), /rows/); // an empty number field
  assert.match(runs.invalidValues(entry, { wide: 'yes' }), /wide must be on or off/);
  // A manifest without `type` is read from the default; with it, 1.0 stays a float.
  assert.deepStrictEqual([{ default: 3 }, { default: 0.5 }, { default: 1, type: 'float' }, { default: '' }, { default: false }].map(runs.optionType),
    ['int', 'float', 'float', 'str', 'bool']);
});

test('query passes --json, the op and the parameters', () => {
  const argv = runs.queryCommand(manifest, 'softmax', 'shape=32,300  dim=1');
  assert.deepStrictEqual(argv.slice(-5), ['--', '--json', 'softmax', 'shape=32,300', 'dim=1']);
  assert(argv.includes('//src/tools/ipu-apps:query'));
  for (const [params, expected] of [
    ['a=1', ['a=1']],
    ['a=1 dim= b=2', ['a=1', 'b=2']], // an empty parameter is dropped
    ['shape="32, 300" name=\'a b\'', ['shape=32, 300', 'name=a b']], // quotes group a value with spaces
  ]) assert.deepStrictEqual(runs.queryCommand(manifest, 'op', params, ['q.sh']), ['q.sh', '--json', 'op', ...expected], params);
});

test('a kernel is found by its .asm path, nothing else', () => {
  const root = '/work/repo';
  assert.strictEqual(runs.kernelForFile(manifest, root, path.join(root, kernel.asm)).name, 'identity');
  assert.strictEqual(runs.kernelForFile(manifest, root, path.join(root, 'elsewhere/identity/identity.asm')), null);
  assert.strictEqual(runs.kernelForFile(null, root, path.join(root, kernel.asm)), null);
});

test('the kernels root is the common ancestor of the kernel folders; the tree groups by family; verdicts', () => {
  const two = { kernels: { a: { asm: 'k/fam1/a/a.asm' }, b: { asm: 'k/fam2/b/b.asm' } } };
  assert.strictEqual(runs.kernelsRoot(two), 'k');
  assert.strictEqual(runs.kernelsRoot({ kernels: {} }), null);
  assert.deepStrictEqual(runs.familyTree(manifest).map((f) => [f.family, f.kernels]), [['reshape', ['identity']]]);
  // A verdict summary names the kernel when there is one.
  assert.strictEqual(runs.describeVerdict({ supported: true, app: 'k', reason: 'r' }), 'SUPPORTED (k): r');
  assert.strictEqual(runs.describeVerdict({ supported: false, app: null, reason: 'r' }), 'NOT SUPPORTED: r');
});

const [manifestPath] = process.argv.slice(2);
if (manifestPath) {
  const real = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  test('every kernel in the real manifest maps back from its .asm and builds its commands', () => {
    for (const [name, k] of Object.entries(real.kernels)) {
      assert.strictEqual(runs.kernelForFile(real, '/repo', path.join('/repo', k.asm)).name, name);
      for (const [caseName, c] of Object.entries(k.cases)) assert(runs.kernelCommand(k, 'run', { caseName, options: runs.optionArgs(c) }));
      assert(runs.kernelCommand(k, 'test'));
    }
    assert(runs.kernelsRoot(real));
  });
}

test('a shell is at its prompt when it holds its terminal\'s foreground', async () => {
  // [ps output, at the prompt]: its own group in front, a command's, no terminal, no such process.
  for (const [ps, at] of [['  4242  4242\n', true], ['4300 4242', false], ['-1 4242', null], ['', null]]) assert.strictEqual(runs.foregroundFrom(ps), at, ps);
  // A real terminal needs a pty (the integration suite has VS Code's); here, no process has no answer.
  assert.strictEqual(await runs.atPrompt(undefined), null);
  assert.strictEqual(await runs.atPrompt(2 ** 22 + 12345), null);
});

test('a checkout without the manifest target is told apart from a failing one', () => {
  for (const [stderr, missing] of [
    ["ERROR: Skipping '//src/tools/ipu-apps:kernel_manifest': no such target '//src/tools/ipu-apps:kernel_manifest': target 'kernel_manifest' not declared in package 'src/tools/ipu-apps'", true],
    ["ERROR: no such package 'src/tools/ipu-apps': BUILD file not found", true],
    ['ValueError: Bazel targets and the kernel registry disagree', false],
    [undefined, false],
  ]) assert.strictEqual(runs.manifestMissing(stderr), missing, stderr);
});

main((n) => `runs.js: ${n} cases pass.`);
