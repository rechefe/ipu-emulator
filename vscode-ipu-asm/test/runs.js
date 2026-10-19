'use strict';

// runs.js: command lines built from a kernel manifest, and whether a terminal is free. Given a real
// manifest (`bazel run //kernels:manifest`), every kernel in it also maps back from its .asm and builds its commands.
//
//   node test/runs.js [manifest.json]

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const runs = require('../runs');
const { test, main } = require('./harness');

const kernel = {
  op: 'identity', variant: '', family: 'reshape',
  asm: 'kernels/reshape/identity/identity.asm',
  targets: { run: '//kernels/reshape/identity:identity', test: '//kernels/reshape/identity:identity' },
  cases: {
    // A case listing options (//kernels:manifest lists none: a case's values are fixed by its name).
    default: { options: {
      rows: { flag: '--rows', default: 4, type: 'int' },
      label: { flag: '--label', default: "it's", type: 'str' },
      wide: { flag: '--wide', default: false, type: 'bool', negated_flag: '--no-wide' },
    } },
  },
};
const manifest = {
  kernels: { identity: kernel },
  families: { reshape: { suite: '//kernels/reshape:all', kernels: ['identity'] } },
  operations: { identity: { params: ['rows'] } }, query: '//kernels:query', skipped: [],
};

test('kernel commands: default options as flags, case and options after --, benchmark only if listed', () => {
  assert.strictEqual(runs.optionArgs(kernel.cases.default), "--rows 4 --label 'it'\\''s' --no-wide");
  const bench = { ...kernel, targets: { ...kernel.targets, benchmark: '//x:benchmark_identity' } };
  for (const [k, mode, options, command] of [
    [kernel, 'run', { caseName: 'default', options: '--rows 8' }, 'bazel run //kernels/reshape/identity:identity -- --case default --rows 8'],
    [kernel, 'debug', { caseName: 'default', flags: ['--define=ipu_proto=1'] },
      'bazel run --config=debug --define=ipu_proto=1 //kernels/reshape/identity:identity -- --case default'],
    [kernel, 'run', { caseName: 'a b' }, "bazel run //kernels/reshape/identity:identity -- --case 'a b'"], // shell characters quoted
    [kernel, 'test', undefined, 'bazel test //kernels/reshape/identity:identity'],
    [kernel, 'benchmark', undefined, null],
    [bench, 'benchmark', undefined, 'bazel run //x:benchmark_identity'],
  ]) assert.strictEqual(runs.kernelCommand(k, mode, options), command, `${mode} ${JSON.stringify(options)}`);
});

test('a manifest value that is no Bazel label or flag is refused, not sent to the shell', () => {
  const evil = '//a:b; rm -rf ~';
  for (const targets of [{ run: evil }, { run: '//a:b', test: evil }, { run: '//a:b', benchmark: '$(id)' }, { run: 'a:b' }]) {
    const k = { ...kernel, targets };
    for (const action of ['run', 'debug', 'test', 'benchmark']) {
      const target = targets[action === 'debug' ? 'run' : action];
      if (target === undefined || target === '//a:b') continue;
      assert.throws(() => runs.kernelCommand(k, action, { caseName: 'default' }), /not a valid Bazel label; nothing was run/, `${action} ${target}`);
    }
  }
  assert.throws(() => runs.familyCommand({ suite: evil }), /family suite/);
  assert.throws(() => runs.familyCommand({}), /family suite/);
  assert.strictEqual(runs.optionArgs({ values: {} }), '');
  assert.strictEqual(runs.invalidValues({ values: {} }, {}), null);
  const badFlag = { options: { rows: { flag: '--rows;id', default: 1, type: 'int' } } };
  assert.throws(() => runs.optionArgs(badFlag), /flag of option rows "--rows;id" is not a valid flag/);
  const badNegated = { options: { wide: { flag: '--wide', negated_flag: '`id`', default: false, type: 'bool' } } };
  assert.throws(() => runs.optionArgs(badNegated), /flag of option wide/);
  // Valid labels, external repositories' included, pass through as they always did.
  assert.strictEqual(runs.familyCommand(manifest.families.reshape, { flags: ['--config=x'] }), 'bazel test --config=x //kernels/reshape:all');
  assert.strictEqual(runs.kernelCommand({ ...kernel, targets: { test: '@other_repo//pkg/sub:t+1' } }, 'test'), 'bazel test @other_repo//pkg/sub:t+1');
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
  assert(argv.includes('//kernels:query'));
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
      // A template and its registry entries share its .asm: the file maps to one of them.
      const found = runs.kernelForFile(real, '/repo', path.join('/repo', k.asm));
      const shared = Object.values(real.kernels).filter((other) => other.asm === k.asm).length > 1;
      assert.strictEqual(shared ? found.kernel.asm : found.name, shared ? k.asm : name);
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
    ["ERROR: Skipping '//kernels:manifest': no such target '//kernels:manifest': target 'manifest' not declared in package 'kernels'", true],
    ["ERROR: no such package 'kernels': BUILD file not found", true],
    ['ValueError: Bazel targets and the kernel registry disagree', false],
    [undefined, false],
  ]) assert.strictEqual(runs.manifestMissing(stderr), missing, stderr);
});

main((n) => `runs.js: ${n} cases pass.`);
