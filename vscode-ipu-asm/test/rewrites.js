'use strict';

// Every rewrite the editor could make to the corpus, for the assembler to judge.
//
//   node test/rewrites.js <data/isa.json> <parser-tokens.json> <rewrites.json>
//
// Per file: each label and Jinja symbol renamed via language.js (as the rename
// provider does), and the file formatted as written and mangled (see
// mangle.js). `check_rewrites` then assembles each: a correct rewrite leaves
// the binary byte for byte the same; a missed use or code moved into a comment
// does not.

const fs = require('fs');

const { Language } = require('../language');
const { mangle } = require('./mangle');

const [isaPath, corpusPath, outPath] = process.argv.slice(2);
if (!outPath) {
  console.error('usage: rewrites.js <isa.json> <parser-tokens.json> <rewrites.json>');
  process.exit(2);
}
const lang = new Language(JSON.parse(fs.readFileSync(isaPath, 'utf8')));

const files = [];
let [renames, formats] = [0, 0];
for (const { file, raw } of JSON.parse(fs.readFileSync(corpusPath, 'utf8'))) {
  const variants = [];
  const seen = new Set();
  for (const [kind, def] of [...lang.labels(raw).defs.map((d) => ['label', d]), ...lang.jinjaDefs(raw).map((d) => ['jinja', d])]) {
    const sym = lang.symbolAt(raw, def.start + 1);
    if (!sym) {
      variants.push({ kind, name: def.name, error: 'no symbol at its own definition' });
      continue;
    }
    // One rename per symbol: a label by name, a Jinja name per scope (two
    // macros' parameters of one name are two symbols).
    const key = kind === 'label' ? `label:${def.name}` : `jinja:${def.name}@${sym.scope ? sym.scope.start : -1}`;
    if (seen.has(key)) continue;
    seen.add(key);
    variants.push({ kind, name: def.name, uses: sym.refs.length, text: lang.rename(raw, sym, `${def.name}_renamed`) });
    renames++;
  }
  for (const blankLine of [true, false]) {
    variants.push({ kind: 'format', name: blankLine ? 'blank lines' : 'no blank lines', uses: 0, text: lang.format(raw, { blankLine }) });
    variants.push({ kind: 'format', name: 'mangled, then formatted', uses: 0, text: lang.format(mangle(lang, raw), { blankLine }) });
    formats += 2;
  }
  files.push({ file, original: raw, variants });
}
fs.writeFileSync(outPath, JSON.stringify(files));
console.log(`Wrote ${renames} renames and ${formats} formatted versions across ${files.length} files.`);
