'use strict';

// The editor's verdict on every text the assembler is to judge, both ways round: the corpus (which
// assembles), its mutations (mangle.js; some assemble, some do not) and the negative corpus
// (test/corpus/negative: one file per known divergence class).
//
//   node test/verdicts.js <data/isa.json> <parser-tokens.json> <verdicts.json>
//
// check_verdicts.py then asks the assembler about each and fails where the two disagree, but for
// the reviewed gaps in test/corpus/negative/allowed-gaps.json.

const fs = require('fs');
const path = require('path');

const { Language } = require('../language');
const { mangle, mutations } = require('./mangle');
const { verdict } = require('./verdict');

const [isaPath, corpusPath, outPath] = process.argv.slice(2);
if (!outPath) {
  console.error('usage: verdicts.js <isa.json> <parser-tokens.json> <verdicts.json>');
  process.exit(2);
}
const lang = new Language(JSON.parse(fs.readFileSync(isaPath, 'utf8')));
const judged = (name, text) => ({ name, text, reasons: verdict(lang, text) });

// Names are repo-relative paths, so allowed-gaps.json keys are unique.
const repo = path.resolve(__dirname, '..', '..');
const cases = [];
const negative = path.join(__dirname, 'corpus', 'negative');
for (const file of fs.readdirSync(negative).filter((f) => f.endsWith('.asm')).sort()) {
  cases.push(judged(path.relative(repo, path.join(negative, file)), fs.readFileSync(path.join(negative, file), 'utf8')));
}
const corpus = JSON.parse(fs.readFileSync(corpusPath, 'utf8'));
let mutated = 0;
for (const { file, text } of corpus) {
  const name = path.relative(repo, path.resolve(repo, file));
  // The rendered text: what the assembler accepted, with no Jinja left to render; and its layout
  // stripped (mangle.js), which must still assemble.
  cases.push(judged(name, text), judged(`${name}: mangled`, mangle(lang, text)));
  for (const m of mutations(lang, text)) {
    cases.push(judged(`${name}: ${m.name}`, m.text));
    mutated++;
  }
}
fs.writeFileSync(outPath, JSON.stringify(cases));
console.log(`Wrote ${cases.length} verdicts: ${corpus.length} corpus files, mangled and as is, ${mutated} mutations of them `
  + `and ${cases.length - 2 * corpus.length - mutated} negative cases.`);
