'use strict';

// The analysis client (analysis.js) and the providers it feeds, on a stand-in `vscode`
// (fake-vscode.js) and a stand-in checker (fake-checker.js) answering what the real one answered
// (fixtures/analysis-v2.json, recorded by record-analysis.js). Expected values are written out
// by hand from the source texts (fixtures/analysis-snippets.json), in VS Code's UTF-16 positions.
//
//   node test/analysis.js

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { AnalysisClient, Positions, commandsFor, parseCheck, versionsFrom } = require('../analysis');
const checker = require('../checker');
const fake = require('./fake-vscode');
const { test, main, until } = require('./harness');
const { load } = require('./record-analysis');

const eq = assert.deepStrictEqual;
const entries = load(path.join(__dirname, 'fixtures', 'analysis-v2.json'));
eq(entries.filter((e) => e.stale).map((e) => e.name), [], 'changed since recorded: re-record with test/record-analysis.js');
const snippet = (name) => entries.find((e) => e.name === `snippet:${name}`).text;

// The stand-in checker reads the recordings, texts included.
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-analysis-'));
process.on('exit', () => fs.rmSync(temp, { recursive: true, force: true }));
const bundle = path.join(temp, 'bundle.json');
fs.writeFileSync(bundle, JSON.stringify({ entries }));
let runs = 0;
/** A checker command answering from the recordings, logging each run to its own file. */
function fakeChecker(...flags) {
  const log = path.join(temp, `runs-${++runs}.log`);
  fs.writeFileSync(log, '');
  const argv = [process.execPath, path.join(__dirname, 'fake-checker.js'), bundle, log, ...flags, 'check', '--json', '--input'];
  const ran = () => fs.readFileSync(log, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  return { argv, ran, probes: () => ran().filter((a) => a.includes('--json-versions')), checks: (v2) => ran().filter((a) => !a.includes('--json-versions') && a.includes('--json-version') === v2) };
}

// --- the contract, without an editor -----------------------------------------------------------

test('a checker command is probed and asked for version 2 by its own argv; any other command runs as written', () => {
  const bazel = checker.bazelCommand();
  const { probe, v2 } = commandsFor(bazel);
  eq(probe, [...bazel.slice(0, -2), '--json-versions']);
  eq(v2, [...bazel.slice(0, -1), '--json-version', '2', '--tokens', '--words', '--jinja', '--input']);
  eq(commandsFor(['/venv/ipu-as', 'check', '--define', 'rows=4', '--json', '--input']).v2,
    ['/venv/ipu-as', 'check', '--define', 'rows=4', '--json', '--json-version', '2', '--tokens', '--words', '--jinja', '--input']);
  for (const argv of [['sh', '-c', 'echo []'], ['ipu-as', 'check', '--json', '--input', 'x'], ['ipu-as', 'check', '--json', '--json-version', '1', '--input']]) {
    eq(commandsFor(argv), { v1: argv, probe: null, v2: null }, argv.join(' '));
  }
  eq([versionsFrom({ code: 0, stdout: '[1, 2]' }), versionsFrom({ code: 2, stdout: '' }), versionsFrom({ code: 0, stdout: 'Usage' })], [[1, 2], [1], [1]]);
});

test('code-point columns and offsets become UTF-16 ones; a text inside the BMP is left alone', () => {
  // Line 1: "a😀b😀c" -- each 😀 is one code point and two UTF-16 units.
  const p = new Positions('x\na\u{1F600}b\u{1F600}c\n');
  eq([0, 1, 2, 3, 4, 5, 9].map((cp) => p.column(1, cp)), [0, 1, 3, 4, 6, 7, 11]);
  eq([0, 2, 3, 4, 5, 6, 7, 8].map((cp) => p.offset(cp)), [0, 2, 3, 5, 6, 8, 9, 10]);
  eq(p.diagnostic({ line: 1, column: 2, end_line: 1, end_column: 4 }), { line: 1, column: 3, end_line: 1, end_column: 6 });
  const plain = { line: 0, column: 3, end_line: 0, end_column: 9 };
  assert.strictEqual(new Positions('SET lr0 cr0;;').diagnostic(plain), plain);
  eq(parseCheck('[]', '').answer, null);
  assert.throws(() => parseCheck('{"schemaVersion": 3}', ''), /unrecognised/);
  assert.throws(() => parseCheck('Traceback', ''));
});

test('each command is probed once; a probe that could not run is asked again; waiting ends with the check', async () => {
  const asked = [];
  const client = new AnalysisClient({
    waitMs: 1000,
    run: async (command, args) => {
      asked.push(args);
      return command === 'gone' ? { code: -1, stdout: '', stderr: 'ENOENT', spawnError: true } : { code: 0, stdout: '[1, 2]' };
    },
  });
  const argv = ['ipu-as', 'check', '--json', '--input'];
  for (let i = 0; i < 3; i++) eq((await client.plan(argv, '.', 1)).version, 2);
  assert.strictEqual(asked.length, 1, 'probed once');
  await client.plan(['gone', 'check', '--json', '--input'], '.', 1);
  await client.plan(['gone', 'check', '--json', '--input'], '.', 1);
  assert.strictEqual(asked.length, 3, 'a probe that did not run is not remembered');
  client.forgetVersions();
  await client.plan(argv, '.', 1);
  assert.strictEqual(asked.length, 4, 'forgotten when the checker may have changed');
  // The first paint, and a document a version-1 checker checks, do not wait.
  assert.strictEqual(await client.current('k', 1, 'HALT;;', { first: false }), null);
  client.store('v1', 1, 'HALT;;', [], null);
  assert.strictEqual(await client.current('v1', 2, 'HALT;; ', { first: false }), null);
  // A provider waits for its version's check; an older one's release does not answer it.
  const { found, answer } = parseCheck(JSON.stringify({ schemaVersion: 2, diagnostics: [] }), 'HALT;;');
  const waiting = client.current('k', 2, 'HALT;;');
  client.release('k', 1);
  client.store('k', 2, 'HALT;;', found, answer);
  client.release('k', 2);
  assert.strictEqual(await waiting, answer);
  assert.strictEqual(await client.current('k', 3, 'HALT;; '), null, 'a version that is not due waits out waitMs');
  client.release('k', 4);
  assert.strictEqual(await client.current('k', 4, 'x'), null, 'a check that ended without an answer: none');
});

// --- the extension, with a version-2 checker ---------------------------------------------------

/** The extension activated in a fresh folder with `checkCommand`. */
async function activated(checkCommand) {
  const dir = fs.mkdtempSync(path.join(temp, 'ws-'));
  const ext = fake.install({ folders: [dir], settings: checkCommand ? { checkCommand } : {} });
  const api = await ext.activate();
  let n = 0;
  const open = (text) => ext.open(fake.Uri.file(path.join(dir, `doc${++n}.asm`)), text);
  return { ext, api, open };
}
const P = (line, character) => new fake.Position(line, character);
const span = (r) => [r.start.line, r.start.character, r.end.line, r.end.character];
const labelsOf = (items) => (items || []).map((i) => i.label);
const diagnosticsOf = (ext, doc, name = 'ipu-asm') => ext.log.diagnostics[name] && ext.log.diagnostics[name].get(doc.uri.toString());
/** `doc` once its check is done: until then the providers paint from the editor's own rules. */
const checked = async (ext, doc) => {
  await until(() => diagnosticsOf(ext, doc), 'the check');
  return doc;
};

test('a v2 checker: diagnostics in UTF-16 columns; semantic tokens by role; one check per document version', async () => {
  const fc = fakeChecker();
  const { ext, open } = await activated(fc.argv);
  const doc = open(snippet('astral-crlf'));
  // Asked before its check is done: answered once it is.
  const tokens = ext.provider('DocumentSemanticTokens').provideDocumentSemanticTokens(doc);
  const diags = await until(() => diagnosticsOf(ext, doc), 'the diagnostics');
  // `B 😀x;;` on line 3: the assembler's columns 2..3 are UTF-16 columns 2..4.
  eq(diags.map((d) => [span(d.range), d.message, d.source]), [[[3, 2, 3, 4], "unexpected '\u{1F600}'; expected one of: ':', token, ';', ';;'", 'ipu-as (parse)']]);
  assert(await tokens, 'semantic tokens');

  const labelled = open(snippet('astral-label'));
  const { tokens: got } = await ext.provider('DocumentSemanticTokens').provideDocumentSemanticTokens(labelled);
  const legend = ext.log.providers.DocumentSemanticTokens[0].rest[0];
  const named = got.map(([line, ch, len, type, mods]) => [line, ch, len, legend.tokenTypes[type], legend.tokenModifiers.filter((_, i) => mods & (1 << i)).join()]);
  eq(named, [
    [0, 0, 11, 'comment', ''], // `# 😀😀 HALT`: 9 code points, 11 units
    [1, 0, 3, 'function', 'declaration'], [1, 5, 3, 'keyword', ''], [1, 9, 3, 'variable', 'readonly'], [1, 13, 3, 'variable', 'readonly'],
    [1, 19, 4, 'comment', ''],
    [2, 4, 5, 'keyword', ''], [2, 10, 3, 'variable', 'readonly'], [2, 14, 3, 'variable', 'readonly'], [2, 18, 3, 'function', ''],
  ]);
  // Every provider asked, twice: still one check per version, and one probe in all.
  const hints = snippet('hints-unused');
  const h = open(hints);
  await until(() => diagnosticsOf(ext, h), 'the check');
  for (let i = 0; i < 2; i++) {
    await ext.provider('InlayHints').provideInlayHints(h, new fake.Range(0, 0, 5, 0));
    await ext.provider('DocumentSymbol').provideDocumentSymbols(h);
    await ext.provider('Hover', 1).provideHover(h, P(2, 7));
    await ext.provider('DocumentSemanticTokens').provideDocumentSemanticTokens(h);
    await ext.provider('CompletionItem').provideCompletionItems(h, P(4, 8), undefined, {});
  }
  eq([fc.probes().length, fc.checks(true).length, fc.checks(false).length], [1, 3, 0]);
  ext.edit(h, snippet('labels'));
  await until(() => fc.checks(true).length === 4, 'the edited version checked');
  await ext.provider('DocumentSemanticTokens').provideDocumentSemanticTokens(h);
  eq([fc.probes().length, fc.checks(true).length], [1, 4]);
  assert.deepStrictEqual(ext.log.warnings, []);
  ext.deactivate();
});

test('a v2 checker: hover, inlay values, unused names and the outline come from its Jinja answer', async () => {
  const { ext, open } = await activated(fakeChecker().argv);
  const params = snippet('macro-param');
  const m = await checked(ext, open(params));
  const hover = async (offset) => (await ext.provider('Hover', 1).provideHover(m, m.positionAt(offset))).contents.value;
  const inMacro = await hover(params.indexOf('{{ r }}') + 3);
  assert(inMacro.includes('macro parameter') && !inMacro.includes('lr0'), inMacro);
  const after = await hover(params.lastIndexOf('{{ r }}') + 3);
  assert(after.startsWith('`r` = `"lr0"`'), after);

  const h = await checked(ext, open(snippet('hints-unused')));
  const hints = await ext.provider('InlayHints').provideInlayHints(h, new fake.Range(0, 0, 5, 0));
  eq(hints.map((x) => [x.label, x.position.line, x.position.character]), [['lr0', 2, 11]]);
  const faded = await until(() => diagnosticsOf(ext, h, 'ipu-asm-unused'), 'the unused names');
  eq(faded.map((d) => [d.message, span(d.range)]), [['end is never branched to', [3, 0, 3, 3]], ['n is set but never used', [1, 7, 1, 8]]]);

  // The ipu:params header's names are the harness's: not in the outline, offered in a tag.
  const header = snippet('header');
  const hd = await checked(ext, open(header));
  eq((await ext.provider('DocumentSymbol').provideDocumentSymbols(hd)).map((s) => [s.name, s.detail]), [['r', '"lr0"']]);
  const inTag = await ext.provider('CompletionItem').provideCompletionItems(hd, hd.positionAt(header.indexOf('{{ r }}') + 3), undefined, {});
  eq(labelsOf(inTag).sort(), ['r', 'rows']);
  ext.deactivate();
});

test('a v2 checker: rename takes every place of the name, keyword arguments of its parameter too; refuses a clash and a header parameter', async () => {
  const fc = fakeChecker();
  const { ext, open } = await activated(fc.argv);
  const text = snippet('kwarg');
  const doc = open(text);
  await until(() => diagnosticsOf(ext, doc), 'the check');
  const rename = ext.provider('Rename');
  const at = doc.positionAt(text.indexOf('set n') + 4);
  eq(span((await rename.prepareRename(doc, at)).range), [3, 7, 3, 8]);
  const edit = await rename.provideRenameEdits(doc, at, 'count');
  const n = text.indexOf('n=n');
  eq(edit.edits.map((e) => [doc.offsetAt(e.range.start), e.newText]),
    [text.indexOf('n=1'), text.indexOf('{{ n }}') + 3, text.indexOf('set n') + 4, n, n + 2].map((o) => [o, 'count']));
  // The original was checked already: only the renamed text is checked, as version 1.
  eq([fc.checks(true).length, fc.checks(false).length], [1, 1]);
  await assert.rejects(rename.provideRenameEdits(doc, at, 'dest'), /already defined or used/);
  const hd = await checked(ext, open(snippet('header')));
  await assert.rejects(Promise.resolve().then(() => rename.prepareRename(hd, hd.positionAt(snippet('header').indexOf('range(rows') + 7))), /Only labels and Jinja names/);
  ext.deactivate();
});

test('a v2 checker: completion fits mnemonics to the word as the assembler rendered it', async () => {
  const { ext, open } = await activated(fakeChecker().argv);
  // `{{ op }}` renders HALT, in cond: with SET replaced, cond is full and LR is free.
  const doc = open(snippet('templated-mnemonic'));
  await until(() => diagnosticsOf(ext, doc), 'the check');
  const names = labelsOf(await ext.provider('CompletionItem').provideCompletionItems(doc, P(1, 14), undefined, {}));
  for (const m of ['SET', 'ADD', 'NOP']) assert(names.includes(m), `${m} offered`);
  for (const m of ['BEQ', 'HALT', 'B']) assert(!names.includes(m), `${m} offered`);
  ext.deactivate();
});

// --- without a version-2 checker -----------------------------------------------------------------

test('a checker from before versions (exit 2 on --json-version): version 1 and the editor\'s own rules, no error shown', async () => {
  const fc = fakeChecker('--old');
  const { ext, api, open } = await activated(fc.argv);
  const text = snippet('astral-crlf');
  const doc = open(text);
  const diags = await until(() => diagnosticsOf(ext, doc), 'the diagnostics');
  eq(diags.map((d) => span(d.range)), [[3, 2, 3, 4]]);
  eq(fc.checks(false).map((a) => a.slice(-4)), [['check', '--json', '--input', '-']]);
  eq([fc.probes().length, fc.checks(true).length], [1, 0]);
  eq([ext.log.warnings, ext.log.errors, ext.statusItem.text], [[], [], '$(check) IPU']);
  assert.strictEqual(api.analysis.latest(doc.uri.toString()), null);
  assert.strictEqual(await ext.provider('DocumentSemanticTokens').provideDocumentSemanticTokens(doc), undefined, 'the grammar colours alone');
  // The JS rules: a templated mnemonic is left to the checker, so cond looks free.
  const t = open(snippet('templated-mnemonic'));
  const names = labelsOf(await ext.provider('CompletionItem').provideCompletionItems(t, P(1, 14), undefined, {}));
  assert(names.includes('BEQ') && names.includes('SET'));
  // Jinja scoping, as before: the header is not read.
  const header = snippet('header');
  const hd = open(header);
  eq(labelsOf(await ext.provider('CompletionItem').provideCompletionItems(hd, hd.positionAt(header.indexOf('{{ r }}') + 3), undefined, {})), ['r']);
  eq(fc.probes().length, 1, 'asked once');
  ext.deactivate();
});

test('no checker (a file in no checkout): every provider answers from the editor\'s own rules at once', async () => {
  const { ext, open } = await activated(null);
  const params = snippet('macro-param');
  const m = open(params);
  assert.strictEqual(await ext.provider('DocumentSemanticTokens').provideDocumentSemanticTokens(m), undefined);
  const value = (await ext.provider('Hover', 1).provideHover(m, m.positionAt(params.lastIndexOf('{{ r }}') + 3))).contents.value;
  assert(value.startsWith('`r` = `"lr0"`'), value);
  const edit = await ext.provider('Rename').provideRenameEdits(m, m.positionAt(params.indexOf('m(r)') + 2), 'q');
  eq(edit.edits.map((e) => m.offsetAt(e.range.start)), [params.indexOf('m(r)') + 2, params.indexOf('{{ r }}') + 3], 'the parameter, in its macro only');
  ext.deactivate();
});

main((n) => `analysis.js: ${n} cases pass.`);
