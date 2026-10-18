'use strict';

// Each document's language data in the extension (languages.js, E08-25), on a stand-in `vscode`
// (fake-vscode.js) and a stand-in `bazel` on PATH whose `ipu-gen vscode-data` writes the bundled
// isa.json plus one instruction only the checkout has: hover finds it once the checkout's data
// is swapped in, and the status bar says which ISA is in use.
//
//   node test/languages.js

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { SUPPORTED_SCHEMA } = require('../language-source');
const fake = require('./fake-vscode');
const { test, main, until } = require('./harness');

const eq = assert.deepStrictEqual;
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-languages-'));
process.on('exit', () => fs.rmSync(temp, { recursive: true, force: true }));

const BUNDLED = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'isa.json'), 'utf8'));
const HALT = Object.keys(BUNDLED.instructions).find((m) => m.toLowerCase() === 'halt');
/** What hover shows for HALT, whose entry CHECKOUT.ONLY copies: its title, read from the data
 *  rather than written here, since data/isa.json is generated from the tree being tested. */
const HALT_TITLE = new RegExp(BUNDLED.instructions[HALT][0].doc.title.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
/** A mnemonic the checkout's ISA has and the bundled one does not. */
const ONLY_HERE = 'CHECKOUT.ONLY';

// The stand-in `bazel`: `run … ipu-gen -- vscode-data DIR` writes DIR/data/isa.json (from the
// file the GENERATED variable names, or fails when it is empty) and logs the run; anything else
// fails, as a checkout without Bazel would.
const bin = path.join(temp, 'bin');
fs.mkdirSync(bin);
const generations = path.join(temp, 'generations.log');
fs.writeFileSync(path.join(bin, 'bazel'), `#!/bin/sh
exec "${process.execPath}" -e '
const fs = require("fs"), path = require("path");
const args = process.argv.slice(1);
if (!args.includes("vscode-data")) process.exit(1);
fs.appendFileSync(${JSON.stringify(generations)}, args.join(" ") + "\\n");
if (!process.env.GENERATED) { process.stderr.write("ERROR: //src/ipu-gen:ipu-gen failed to build\\n"); process.exit(1); }
const out = args[args.length - 1];
fs.mkdirSync(path.join(out, "data"), { recursive: true });
fs.copyFileSync(process.env.GENERATED, path.join(out, "data", "isa.json"));
' -- "$@"
`, { mode: 0o755 });
process.env.PATH = `${bin}${path.delimiter}${process.env.PATH}`;
const ran = () => (fs.existsSync(generations) ? fs.readFileSync(generations, 'utf8').split('\n').filter(Boolean) : []);

/** Write the isa.json the stand-in generates: the bundled one, with ONLY_HERE and `extra`. */
function generates(extra = {}) {
  const file = path.join(temp, `isa-${Math.random().toString(36).slice(2)}.json`);
  const isa = { ...BUNDLED, instructions: { ...BUNDLED.instructions, [ONLY_HERE]: BUNDLED.instructions[HALT] }, ...extra };
  fs.writeFileSync(file, JSON.stringify(isa));
  process.env.GENERATED = file;
}

let n = 0;
/** The extension activated on a fresh checkout (with ISA sources), a document open in it. */
async function opened({ trusted = true, settings = {} } = {}) {
  fs.rmSync(generations, { force: true });
  const root = path.join(temp, `checkout-${++n}`);
  for (const marker of ['MODULE.bazel', 'src/ipu-as/src/ipu_as/__init__.py', 'src/ipu-isa/src/spec.py']) {
    fs.mkdirSync(path.dirname(path.join(root, marker)), { recursive: true });
    fs.writeFileSync(path.join(root, marker), '');
  }
  const ext = fake.install({
    trusted,
    folders: [root],
    storage: path.join(temp, `storage-${n}`),
    settings: { checkCommand: [process.execPath, '-e', "process.stdout.write('[]')"], ...settings },
  });
  const api = await ext.activate();
  const doc = ext.open(fake.Uri.file(path.join(root, 'k.asm')), `${ONLY_HERE};;\n`);
  const hover = async () => {
    const found = await ext.provider('Hover').provideHover(doc, new fake.Position(0, 1));
    return found ? found.contents.value : null;
  };
  return { ext, api, doc, hover };
}

test("a trusted checkout's own ISA answers once generated; until then the bundled one, and the status says so", async () => {
  generates();
  const { ext, api, doc, hover } = await opened();
  assert.strictEqual(await hover(), null, 'the bundled ISA has no such instruction');
  await until(() => api.languages.chosen(doc).use === 'checkout', 'the checkout data');
  assert.match(await hover(), HALT_TITLE, "the checkout's entry (a copy of HALT's)");
  await until(() => ext.statusItem.text === '$(check) IPU', 'the status without "(bundled ISA)"');
  eq(ran().length, 1, 'generated once');
  // A source change generates it again, the current data answering meanwhile.
  const [root] = api.languages.entries.keys();
  const later = new Date(Date.now() + 60000);
  fs.utimesSync(path.join(root, 'src/ipu-isa/src/spec.py'), later, later);
  api.languages.invalidate(root);
  assert.strictEqual(api.languages.chosen(doc).use, 'checkout');
  await until(() => ran().length === 2 && !api.languages.entries.values().next().value.loading, 'generated again');
  // A failed regeneration (sources mid-edit) keeps the data it had.
  delete process.env.GENERATED;
  fs.utimesSync(path.join(root, 'src/ipu-isa/src/spec.py'), new Date(Date.now() + 120000), new Date(Date.now() + 120000));
  api.languages.invalidate(root);
  api.languages.chosen(doc);
  await until(() => ran().length === 3 && !api.languages.entries.get(root).loading, 'the regeneration to fail');
  eq(api.languages.chosen(doc).use, 'checkout');
  assert.match(await hover(), HALT_TITLE);
  ext.deactivate();
});

test('ipuAsm.languageSource "bundled", and Restricted Mode, generate nothing', async () => {
  generates();
  for (const options of [{ settings: { languageSource: 'bundled' } }, { trusted: false }]) {
    const { ext, api, doc, hover } = await opened(options);
    await new Promise((r) => setTimeout(r, 300));
    eq([ran(), api.languages.chosen(doc).use, await hover()], [[], 'bundled', null], JSON.stringify(options));
    await until(() => /\(bundled ISA\)/.test(ext.statusItem.text || ''), 'the status to say so');
    ext.deactivate();
  }
});

test('a newer isa.json layout, or a failed generation, keeps the bundled ISA; a warning says why when it should', async () => {
  generates({ schemaVersion: SUPPORTED_SCHEMA + 1 });
  const newer = await opened();
  await until(() => newer.ext.log.warnings.some((w) => /Update the extension/.test(w)), 'the update warning');
  eq(newer.api.languages.chosen(newer.doc).use, 'bundled');
  newer.api.languages.chosen(newer.doc);
  eq(newer.ext.log.warnings.filter((w) => /Update the extension/.test(w)).length, 1, 'once');
  assert.match(newer.ext.statusItem.text, /\(bundled ISA\)/);
  newer.ext.deactivate();

  delete process.env.GENERATED;
  for (const [languageSource, warned] of [['auto', false], ['checkout', true]]) {
    const failing = await opened({ settings: { languageSource } });
    await until(() => ran().length === 1 && !failing.api.languages.entries.values().next().value.loading, 'the generation to fail');
    failing.api.languages.chosen(failing.doc);
    eq([failing.api.languages.chosen(failing.doc).use, failing.ext.log.warnings.some((w) => /failed to build/.test(w))], ['bundled', warned], languageSource);
    failing.ext.deactivate();
  }
});

main((count) => `languages.js: ${count} cases pass.`);
