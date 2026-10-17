'use strict';

// Every rewrite the editor could make to the corpus, for the assembler to judge.
//
//   node test/rewrites.js <data/isa.json> <parser-tokens.json> <rewrites.json> [analysis bundle]
//
// Per file: each label and Jinja symbol renamed as the rename provider does it
// without a version-2 checker (fallback/) and, for a file the analysis bundle
// (default fixtures/analysis-v2.json; record-analysis.js) has the assembler's
// answer for, with one (analysis-semantics.js); and the file formatted as
// written and mangled (see mangle.js). `check_rewrites` then assembles each: a correct rewrite leaves
// the binary byte for byte the same; a missed use or code moved into a comment
// does not.

const fs = require('fs');
const path = require('path');

const { Answer } = require('../analysis');
const { V2Semantics } = require('../analysis-semantics');
const { load } = require('./record-analysis');
const { Language } = require('../language');
const { fallbackOf } = require('../fallback');
const { mangle } = require('./mangle');

const [isaPath, corpusPath, outPath, bundlePath = path.join(__dirname, 'fixtures', 'analysis-v2.json')] = process.argv.slice(2);
if (!outPath) {
  console.error('usage: rewrites.js <isa.json> <parser-tokens.json> <rewrites.json> [analysis bundle]');
  process.exit(2);
}
const lang = new Language(JSON.parse(fs.readFileSync(isaPath, 'utf8')));
const fb = fallbackOf(lang);
/** The assembler's answers, by text. */
const answers = new Map(load(bundlePath).filter((e) => !e.stale).map((e) => [e.text, e.answer]));

const files = [];
let [renames, formats] = [0, 0];
for (const { file, raw } of JSON.parse(fs.readFileSync(corpusPath, 'utf8'))) {
  const variants = [];
  const seen = new Set();
  for (const [kind, def] of [...lang.labels(raw).defs.map((d) => ['label', d]), ...fb.jinjaDefs(raw).map((d) => ['jinja', d])]) {
    const sym = fb.symbolAt(raw, def.start + 1);
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
  // With the assembler's answer: one rename per Jinja name, every place of it.
  if (answers.has(raw)) {
    const known = new V2Semantics(lang, new Answer(raw, answers.get(raw)), raw);
    const named = new Set();
    for (const o of known.model.occurrences) {
      if (!o.def || named.has(o.name)) continue;
      named.add(o.name);
      const sym = known.symbolAt(o.start + 1);
      if (!sym) continue; // a name the ipu:params header declares
      variants.push({ kind: 'jinja (analysis)', name: o.name, uses: sym.refs.length, text: lang.rename(raw, sym, `${o.name}_renamed`) });
      renames++;
    }
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
