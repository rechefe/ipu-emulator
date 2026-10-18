'use strict';

// Where the language data comes from (language-source.js): which source a document uses, and
// a checkout's data generated once per source change, on a stand-in generator.
//
//   node test/language-source.js

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { BundledLanguageSource, CheckoutLanguageSource, GENERATOR_TARGET, SUPPORTED_SCHEMA, select } = require('../language-source');
const { test, main } = require('./harness');

const eq = assert.deepStrictEqual;
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-langsrc-'));
process.on('exit', () => fs.rmSync(temp, { recursive: true, force: true }));
const BUNDLED = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'isa.json'), 'utf8'));

test('a trusted checkout with readable data uses it; anything else uses the bundled data, and says why', () => {
  const lang = {};
  const use = (o) => select({ trusted: true, root: '/c', checkout: { lang }, ...o });
  eq(use({}), { use: 'checkout', why: null, warn: null });
  eq(use({ setting: 'checkout' }).use, 'checkout');
  for (const [o, why] of [
    [{ setting: 'bundled' }, /"bundled"/],
    [{ root: null }, /no IPU checkout/],
    [{ trusted: false }, /Restricted Mode/],
    [{ checkout: undefined }, /being generated/],
    [{ checkout: { error: 'bazel: not found' } }, /bazel: not found/],
  ]) {
    const chosen = use(o);
    eq(chosen.use, 'bundled', JSON.stringify(o));
    assert.match(chosen.why, why);
    assert.strictEqual(chosen.warn, null, 'auto falls back quietly');
  }
  // A failure is a warning when the checkout's data was asked for; a newer layout always is.
  assert.match(use({ setting: 'checkout', checkout: { error: 'boom' } }).warn, /bundled ISA\. boom/);
  const newer = use({ checkout: { newer: SUPPORTED_SCHEMA + 1 } });
  eq(newer.use, 'bundled');
  assert.match(newer.warn, /Update the extension/);
});

/** A checkout with one ISA source, and a stand-in generator writing `isa` into its out dir. */
function checkout(isa) {
  const root = fs.mkdtempSync(path.join(temp, 'checkout-'));
  const source = path.join(root, 'src', 'tools', 'ipu-isa', 'src', 'spec.py');
  fs.mkdirSync(path.dirname(source), { recursive: true });
  fs.writeFileSync(source, '# the ISA\n');
  const runs = [];
  const run = async (command, args, cwd) => {
    runs.push([command, ...args]);
    if (isa.fail) return { code: 1, stdout: '', stderr: 'ERROR: build failed\n' };
    const out = args[args.length - 1];
    fs.mkdirSync(path.join(out, 'data'), { recursive: true });
    fs.writeFileSync(path.join(out, 'data', 'isa.json'), JSON.stringify(isa.data));
    assert.strictEqual(cwd, root);
    return { code: 0, stdout: '', stderr: '' };
  };
  const touch = (ms, file = source) => {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (!fs.existsSync(file)) fs.writeFileSync(file, '');
    fs.utimesSync(file, new Date(ms), new Date(ms));
  };
  return { root, runs, run, touch, source: (cacheBase) => new CheckoutLanguageSource(root, { cacheBase, run }) };
}

test("a checkout's data is generated once, and again when its sources change", async () => {
  const isa = { data: { ...BUNDLED, instructions: { ...BUNDLED.instructions, ONLY_HERE: BUNDLED.instructions[Object.keys(BUNDLED.instructions)[0]] } } };
  const c = checkout(isa);
  const cache = path.join(temp, 'cache-1');
  const { lang } = await c.source(cache).load();
  assert.ok(lang.lookup('ONLY_HERE'), "the checkout's own mnemonic");
  eq(c.runs.length, 1);
  const [command] = c.runs;
  eq([command[0], command[1], command.includes(GENERATOR_TARGET), command.slice(-2)[0]], ['bazel', 'run', true, 'vscode-data']);
  // A new window (a new source object) reads the cache.
  await c.source(cache).load();
  eq(c.runs.length, 1, 'fresh: not generated again');
  c.touch(Date.now() + 60000);
  await c.source(cache).load();
  eq(c.runs.length, 2, 'a source newer than the data: generated again');
  // The assembler's sources too: the generator reads its grammar and slot rules.
  c.touch(Date.now() + 120000, path.join(c.root, 'src', 'tools', 'ipu-as', 'src', 'ipu_as', 'ipu_token.py'));
  await c.source(cache).load();
  eq(c.runs.length, 3, 'an assembler source newer than the data: generated again');
  eq(fs.readdirSync(cache).filter((f) => f.includes('.new-')), [], 'no generation directory left behind');
  // Another checkout has its own directory.
  const other = checkout(isa);
  await other.source(cache).load();
  eq(other.runs.length, 1);
});

test('a failed generation throws why; a newer layout is not read', async () => {
  const failing = checkout({ fail: true });
  await assert.rejects(failing.source(path.join(temp, 'cache-2')).load(), /language data failed: ERROR: build failed/);
  const newer = checkout({ data: { ...BUNDLED, schemaVersion: SUPPORTED_SCHEMA + 1 } });
  eq(await newer.source(path.join(temp, 'cache-3')).load(), { newer: SUPPORTED_SCHEMA + 1 });
  const empty = fs.mkdtempSync(path.join(temp, 'empty-'));
  await assert.rejects(new CheckoutLanguageSource(empty, { cacheBase: temp, run: assert.fail }).load(), /no ISA sources/);
  assert.ok((await new BundledLanguageSource().load()).lang.lookup(Object.keys(BUNDLED.instructions)[0]));
  eq(BUNDLED.schemaVersion, SUPPORTED_SCHEMA, 'the bundled data is the newest layout this extension reads');
});

main((n) => `language-source.js: ${n} cases pass.`);
