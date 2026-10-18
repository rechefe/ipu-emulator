'use strict';

// Records what the editor's formatter (language.js `format`) makes of the corpus, as the expected
// values of the assembler's own formatter (src/tools/ipu-as-py/test/test_format.py). Run once when
// the formatter's rules change on purpose, or when test_format.py skips a kernel that changed since
// it was recorded; the recorded outputs are then the specification.
//
//   bazel run //vscode-ipu-asm:gen_vscode               # writes vscode-ipu-asm/data/isa.json
//   node vscode-ipu-asm/test/record-format.js            # every file below, into test/golden/format
//   node vscode-ipu-asm/test/record-format.js <isa.json> <out-dir> <file.asm>...
//
// It is not an update_* target in tools/update_baselines.sh: Bazel has no node toolchain here.
// With no arguments it records the extension's corpus and every kernel source (DEFAULT_DIRS).
//
// Each file is formatted with each of VARIANTS, as written and with its layout stripped (mangle.js).
// A range variant is recorded as the whole text with the range's lines replaced by what `format`
// returns for them, which is what `ipu-as format --range` prints. Files under
// vscode-ipu-asm/test/corpus go to corpus.json with every output in full; the kernels go to
// kernels.json as SHA-256 digests (a megabyte of kernels, already in the house layout), with the
// digest of each kernel as recorded (`input`), so the Python side can tell a kernel edit from a
// formatter change, and of each stripped copy, so it can check that it stripped the same.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { Language } = require('../language');
const { spliceLines } = require('../formatting');
const { mangle } = require('./mangle');

const root = path.resolve(__dirname, '..', '..');
const DEFAULT_DIRS = [
  'vscode-ipu-asm/test/corpus',
  'kernels',
];

/** Every .asm file under dir, recursively. */
const asmUnder = (dir) =>
  fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) return asmUnder(full);
    return e.name.endsWith('.asm') ? [full] : [];
  });

const args = process.argv.slice(2);
if (args.length === 1 || args.length === 2) {
  console.error('usage: record-format.js [<isa.json> <out-dir> <file.asm>...]');
  process.exit(2);
}
const [isaPath, outDir, ...files] = args.length
  ? args
  : [
      path.join(root, 'vscode-ipu-asm/data/isa.json'),
      path.join(root, 'src/tools/ipu-as-py/test/golden/format'),
      ...DEFAULT_DIRS.flatMap((d) => asmUnder(path.join(root, d))),
    ];
const lang = new Language(JSON.parse(fs.readFileSync(isaPath, 'utf8')));

/** The middle third of the lines, 0-based and inclusive. */
const middle = (text) => {
  const n = text.split('\n').length;
  return { fromLine: Math.floor(n / 3), toLine: Math.floor((2 * n) / 3) };
};

/** [name, options for a text]; the names are the keys test_format.py reads. */
const VARIANTS = [
  ['default', () => ({})],
  ['blank-line', () => ({ blankLine: true })],
  ['indent-tab', () => ({ indent: '\t' })],
  ['indent-2-range', (text) => ({ indent: '  ', ...middle(text) })],
  ['blank-line-range', (text) => ({ blankLine: true, ...middle(text) })],
];

/** The whole text with lines [fromLine, toLine] replaced by their formatted version. */
function formatted(text, options) {
  const out = lang.format(text, options);
  return options.fromLine === undefined ? out : spliceLines(text, options.fromLine, options.toLine, out);
}

const sha = (text) => crypto.createHash('sha256').update(text, 'utf8').digest('hex');

const corpus = {};
const kernels = {};
for (const file of [...files].sort()) {
  const name = path.relative(root, path.resolve(file)).split(path.sep).join('/');
  const raw = fs.readFileSync(file, 'utf8');
  const mangled = mangle(lang, raw);
  const entry = {};
  for (const [variant, options] of VARIANTS) {
    entry[variant] = formatted(raw, options(raw));
    entry[`mangled/${variant}`] = formatted(mangled, options(mangled));
  }
  if (name.startsWith('vscode-ipu-asm/test/corpus/')) {
    corpus[name] = { mangled, ...entry };
  } else {
    kernels[name] = Object.fromEntries([['input', sha(raw)], ['mangled', sha(mangled)], ...Object.entries(entry).map(([k, v]) => [k, sha(v)])]);
  }
}
const write = (file, value) => fs.writeFileSync(path.join(outDir, file), `${JSON.stringify(value, null, 1)}\n`);
write('corpus.json', corpus);
write('kernels.json', kernels);
console.log(`Recorded ${VARIANTS.length * 2} formats of ${Object.keys(corpus).length} corpus files and ${Object.keys(kernels).length} kernels.`);
