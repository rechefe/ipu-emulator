'use strict';

// The checkout's assembler as the formatter (formatting.js) and the format provider over it:
// the commands it runs, what it reads back from them, and the editor's rules where it cannot be
// asked. A stand-in assembler (fake-formatter.js) answers by the editor's rules. With an
// assembler command after the ISA, the assembler's `format` must also answer as the editor's
// rules do, on every corpus file and layout:
//
//   node test/formatting.js <data/isa.json> [<ipu-as> ...]

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const checker = require('../checker');
const { CheckoutFormatter, FORMAT_SECONDS, assemblerOf, formatArgv, linesOf, spliceLines } = require('../formatting');
const { Language } = require('../language');
const fake = require('./fake-vscode');
const { test, main } = require('./harness');

const [isaPath, ...assembler] = process.argv.slice(2);
if (!isaPath) {
  console.error('usage: formatting.js <isa.json> [<ipu-as> ...]');
  process.exit(2);
}
const isa = path.resolve(isaPath);
const lang = new Language(JSON.parse(fs.readFileSync(isa, 'utf8')));
const eq = assert.deepStrictEqual;

const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-format-'));
process.on('exit', () => fs.rmSync(temp, { recursive: true, force: true }));
let runs = 0;
/** A stand-in checker command, logging each run. */
function standIn(...flags) {
  const log = path.join(temp, `runs-${++runs}.log`);
  fs.writeFileSync(log, '');
  const argv = [process.execPath, path.join(__dirname, 'fake-formatter.js'), isa, log, ...flags, 'check', '--json', '--input'];
  const ran = () => fs.readFileSync(log, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  return { argv, ran, formats: () => ran().filter((a) => a[0] === 'format' && !a.includes('--help')) };
}

const MESSY = 'top: SET lr0 cr0;; HALT;;\n  # note\nSET lr1 cr0;;\n';
const ALL = (text, extra = {}) => ({ indent: '    ', blankLine: false, fromLine: 0, toLine: text.split('\n').length - 1, ...extra });

/** The corpus as recorded for the assembler's formatter: [name, text, options] per layout. */
function corpusCases() {
  const root = path.resolve(__dirname, '..', '..');
  const recorded = JSON.parse(fs.readFileSync(path.join(root, 'src/tools/ipu-as-py/test/golden/format/corpus.json'), 'utf8'));
  const middle = (text) => {
    const n = text.split('\n').length;
    return { fromLine: Math.floor(n / 3), toLine: Math.floor((2 * n) / 3) };
  };
  return Object.entries(recorded).flatMap(([name, entry]) => {
    const raw = fs.readFileSync(path.join(root, name), 'utf8');
    return [['', raw], ['mangled/', entry.mangled]].flatMap(([prefix, text]) => [
      [`${name}:${prefix}default`, text, ALL(text), entry[`${prefix}default`]],
      [`${name}:${prefix}indent-2-range`, text, { ...ALL(text, { indent: '  ' }), ...middle(text) }, entry[`${prefix}indent-2-range`]],
      [`${name}:${prefix}blank-line-range`, text, { ...ALL(text, { blankLine: true }), ...middle(text) }, entry[`${prefix}blank-line-range`]],
    ]);
  });
}

// --- the commands, and reading their output ----------------------------------------------------

test('format runs the checker command before its `check`; a command of another shape has no format', () => {
  eq(assemblerOf(['bazel', 'run', '//x:ipu-as', '--', 'check', '--json', '--input']), ['bazel', 'run', '//x:ipu-as', '--']);
  eq([assemblerOf(['sh', '-c', 'x']), assemblerOf(['check', '--json', '--input']), assemblerOf(['ipu-as', 'check', '--input', 'f'])], [null, null, null]);
  eq(formatArgv(['ipu-as'], ALL('a\nb'), 2), ['ipu-as', 'format', '--indent', '4', '--input', '-']);
  eq(formatArgv(['ipu-as'], { indent: '\t', blankLine: true, fromLine: 1, toLine: 1 }, 3),
    ['ipu-as', 'format', '--indent', 'tab', '--blank-line', '--range', '2:2', '--input', '-']);
});

test('the lines of a range come back as the editor formats them, CRLF and past-the-end ranges too', () => {
  for (const [name, text, options, out] of corpusCases()) {
    eq(linesOf(text, out, options.fromLine, options.toLine), lang.format(text, options), name);
  }
  const crlf = 'a: HALT;;\r\nHALT;; HALT;;\r\nHALT;;';
  eq(linesOf(crlf, spliceLines(crlf, 1, 1, 'X\r\nY'), 1, 1), 'X\r\nY');
  eq(linesOf('a\nb', spliceLines('a\nb', 1, 9, 'B'), 1, 9), 'B');
  assert.strictEqual(linesOf('a\nb\nc', 'z\nB\nc', 1, 1), null, 'a line outside the range changed');
});

// --- the formatter -----------------------------------------------------------------------------

test('a checker is asked once whether it formats; an old one, a failure or another shape leave it to the editor', async () => {
  const formatter = new CheckoutFormatter();
  const cur = standIn();
  for (let i = 0; i < 2; i++) eq(await formatter.format(cur.argv, temp, 10, MESSY, ALL(MESSY)), { lines: lang.format(MESSY, ALL(MESSY)) });
  eq(cur.ran().filter((a) => a.includes('--help')).length, 1, 'probed once');
  const refused = await formatter.format(cur.argv, temp, 10, 'REFUSE', ALL('REFUSE'));
  assert.match(refused.refused, /would change the program/);
  assert.strictEqual(await formatter.format(cur.argv, temp, 10, 'CRASH', ALL('CRASH')), null);
  const old = standIn('--old');
  assert.strictEqual(await formatter.format(old.argv, temp, 10, MESSY, ALL(MESSY)), null);
  eq(old.formats().length, 0, 'an assembler without format is not asked to format');
  formatter.forget(temp);
  await formatter.format(cur.argv, temp, 10, MESSY, ALL(MESSY));
  eq(cur.ran().filter((a) => a.includes('--help')).length, 2, 'asked again once its sources changed');
  assert.strictEqual(await new CheckoutFormatter({ run: assert.fail }).format(['sh', '-c', 'x'], temp, 10, MESSY, ALL(MESSY)), null);
});

test('the bazel run path, a slow run and a cancelled one leave formatting to the editor', async () => {
  const bazel = new CheckoutFormatter({ run: assert.fail });
  assert.strictEqual(await bazel.format(checker.bazelCommand(), temp, 300, MESSY, ALL(MESSY)), null, 'nothing runs');
  // A run past FORMAT_SECONDS: the editor's rules, and no wait on the next format either.
  let limits = [];
  const slow = new CheckoutFormatter({
    retryMs: 60000,
    run: async (_c, args, _cwd, seconds) => (limits.push(seconds), args.includes('--help')
      ? { code: 0, stdout: '', stderr: '' } : { code: -1, stdout: '', stderr: 'timed out', timedOut: true }),
  });
  const argv = ['/bin/ipu-as', 'check', '--json', '--input'];
  assert.strictEqual(await slow.format(argv, '/c', 300, MESSY, ALL(MESSY)), null);
  eq(limits, [FORMAT_SECONDS, FORMAT_SECONDS], 'its own short limit, not timeoutSeconds');
  assert.strictEqual(await slow.format(argv, '/c', 300, MESSY, ALL(MESSY)), null);
  eq(limits.length, 2, 'not asked again for a while');
  slow.forget('/c');
  await slow.format(argv, '/c', 300, MESSY, ALL(MESSY));
  eq(limits.length, 4, 'until its sources change (probed again, then run)');
  // Cancelling stops the run.
  const stopped = [];
  let cancel;
  const token = { isCancellationRequested: false, onCancellationRequested: (f) => ((cancel = f), { dispose() {} }) };
  const hanging = new CheckoutFormatter({
    run: async (_c, args, _cwd, _s, { onSpawn } = {}) => {
      if (args.includes('--help')) return { code: 0, stdout: '', stderr: '' };
      return new Promise((resolve) => onSpawn({ stop: () => (stopped.push(1), resolve({ code: null, stdout: '', stderr: '' })) }));
    },
  });
  const pending = hanging.format(argv, '/c', 300, MESSY, ALL(MESSY), token);
  await new Promise((r) => setImmediate(r));
  token.isCancellationRequested = true;
  cancel();
  assert.strictEqual(await pending, null);
  eq(stopped, [1]);
});

// --- the provider ------------------------------------------------------------------------------

/** The extension activated in a fresh folder with checker `argv`; the document formatter. */
async function activated(argv) {
  const dir = fs.mkdtempSync(path.join(temp, 'ws-'));
  const ext = fake.install({ folders: [dir], settings: { checkCommand: argv } });
  await ext.activate();
  return { ext, dir, open: (text) => ext.open(fake.Uri.file(path.join(dir, 'k.asm')), text) };
}
const OPTIONS = { insertSpaces: true, tabSize: 4 };
const applied = (doc, edits) => edits.reduceRight((text, e) => text.slice(0, doc.offsetAt(e.range.start)) + e.newText + text.slice(doc.offsetAt(e.range.end)), doc.getText());

test('the provider formats by the assembler where it can, else by the editor; a refusal edits nothing', async () => {
  for (const flags of [[], ['--old']]) {
    const checker = standIn(...flags);
    const { ext, open } = await activated(checker.argv);
    const doc = open(MESSY);
    const formatting = ext.provider('DocumentFormattingEdit');
    eq(applied(doc, await formatting.provideDocumentFormattingEdits(doc, OPTIONS)), lang.format(MESSY, ALL(MESSY)));
    eq(checker.formats().length, flags.length ? 0 : 1, flags.join(' ') || 'current');
    const range = await ext.provider('DocumentRangeFormattingEdit').provideDocumentRangeFormattingEdits(doc, new fake.Range(2, 0, 2, 0), OPTIONS);
    eq(applied(doc, range), spliceLines(MESSY, 2, 2, lang.format(MESSY, ALL(MESSY, { fromLine: 2, toLine: 2 }))));
    if (!flags.length) {
      eq(checker.formats().at(-1).slice(-4), ['--range', '3:3', '--input', '-']);
      const refused = open('REFUSE');
      eq(await formatting.provideDocumentFormattingEdits(refused, OPTIONS), []);
      assert.match(ext.log.warnings.at(-1), /left as it is, since formatting would change the program/);
      assert.doesNotMatch(ext.log.warnings.at(-1), /Error formatting/);
      // An edit while the assembler ran: its answer was for the old text, so the editor's rules
      // format the new one.
      const typed = open(MESSY);
      const pending = formatting.provideDocumentFormattingEdits(typed, OPTIONS);
      ext.edit(typed, `${MESSY}HALT;;\n`);
      eq(applied(typed, await pending), lang.format(typed.getText(), ALL(typed.getText())));
    }
    ext.deactivate();
  }
});

test('in Restricted Mode nothing runs: the editor formats', async () => {
  const cur = standIn();
  const dir = fs.mkdtempSync(path.join(temp, 'ws-'));
  const ext = fake.install({ trusted: false, folders: [dir], settings: { checkCommand: cur.argv } });
  await ext.activate();
  const doc = ext.open(fake.Uri.file(path.join(dir, 'k.asm')), MESSY);
  eq(applied(doc, await ext.provider('DocumentFormattingEdit').provideDocumentFormattingEdits(doc, OPTIONS)), lang.format(MESSY, ALL(MESSY)));
  eq(cur.ran(), [], 'not even the probe');
  ext.deactivate();
});

// --- the real assembler ------------------------------------------------------------------------

if (assembler.length) {
  test('the assembler formats the corpus as the editor does', async () => {
    const argv = [...assembler, 'check', '--json', '--input'];
    const formatter = new CheckoutFormatter();
    for (const [name, text, options] of corpusCases()) {
      eq(await formatter.format(argv, process.cwd(), 120, text, options), { lines: lang.format(text, options) }, name);
    }
  });
}

main((n) => `formatting.js: ${n} cases pass${assembler.length ? '' : ' (no assembler given: stand-in only)'}.`);
