'use strict';

// The editor's own rules (fallback/, used without a version-2 checker) against the assembler's
// analysis (`check --json --json-version 2`), both ways round, on recorded answers:
//
//   node test/analysis-agreement.js <data/isa.json> [bundle.json] [--write-gaps]
//
// For each source: every assembly token either side reads (by role), every word's slot fit, and
// every Jinja tag's defined and used names must be found the same on the other side. What is
// not is a known gap, listed in fixtures/analysis-known-gaps.json by file; the test fails on a
// difference not listed there, and on a listed one that no longer happens (fixed: take it off).
// The bundle defaults to the committed fixtures/analysis-v2.json; test/record-analysis.js records
// one for the whole corpus. --write-gaps rewrites the list for the bundle's files.

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const { Answer } = require('../analysis');
const { fallbackOf } = require('../fallback');
const { Language, jinjaRegions } = require('../language');
const { load } = require('./record-analysis');

const GAPS = path.join(__dirname, 'fixtures', 'analysis-known-gaps.json');

/** The assembler's roles, as coarse as the fallback's scan reads them. */
const COARSE = { mnemonic: 'mnemonic', 'label-def': 'label', register: 'operand', enum: 'operand', number: 'operand', 'label-ref': 'operand' };

/** Every difference between the fallback and the answer for `text`, as sentences. */
function differences(lang, text, json) {
  const fb = fallbackOf(lang);
  const answer = new Answer(text, json);
  const out = [];
  const where = (offset) => {
    const before = text.slice(0, offset);
    const line = before.split('\n').length - 1;
    return `${line + 1}:${offset - before.lastIndexOf('\n')}`;
  };
  const a = (role) => (/^[aeiou]/.test(role) ? `an ${role}` : `a ${role}`);
  const quote = (s) => JSON.stringify(s.length > 40 ? `${s.slice(0, 40)}…` : s);
  const { positions } = answer;

  // --- tokens ---------------------------------------------------------------------------------
  const scanned = lang.scan(text);
  const mine = new Map(scanned.tokens.filter((t) => !t.templated).map((t) => [t.start, { role: t.role, end: t.end }]));
  const theirs = new Map(answer.tokens.filter((t) => COARSE[t.role])
    .map((t) => [positions.offsetAt(t.line, 0) + t.character, { role: COARSE[t.role], end: positions.offsetAt(t.line, 0) + t.character + t.length }]));
  if (!theirs.size && mine.size) {
    out.push(`tokens: the assembler lists no assembly tokens (${mine.size} in the scan)`);
  } else {
    for (const [at, t] of mine) {
      const other = theirs.get(at);
      if (!other) out.push(`token ${where(at)} ${quote(text.slice(at, t.end))}: the scan reads ${a(t.role)}, the assembler nothing`);
      else if (other.role !== t.role || other.end !== t.end) out.push(`token ${where(at)} ${quote(text.slice(at, t.end))}: the scan reads ${a(t.role)}, the assembler ${a(other.role)} ${quote(text.slice(at, other.end))}`);
    }
    for (const [at, t] of theirs) {
      if (!mine.has(at)) out.push(`token ${where(at)} ${quote(text.slice(at, t.end))}: the assembler reads ${a(t.role)}, the scan nothing`);
    }
  }

  // --- words ----------------------------------------------------------------------------------
  const fit = (yes) => (yes ? 'fits' : 'does not fit');
  const words = new Map(answer.words.map((w) => [w.start, w]));
  const matched = new Set();
  if (!answer.words.length && scanned.words.some((w) => w.instructions.length)) {
    out.push(`words: the assembler lists none (${scanned.words.length} in the scan)`);
  } else {
    for (const w of scanned.words) {
      const names = w.instructions.map((i) => i.mnemonic.text);
      // The assembler's range starts at the word's first token in the source: past a templated
      // label (`{{ lbl }}_loop:`), or one a loop puts outside the rendering it lists. (The scan
      // reads each `{% if %}` branch, so one source word can be several of its words.)
      const tokens = w.instructions.flatMap((i) => [i.mnemonic, ...i.operands]).filter((t) => !t.templated);
      const other = [w.start, w.label && !w.label.templated ? w.label.start : -1, tokens.length ? tokens[0].start : -1]
        .map((at) => words.get(at)).find(Boolean);
      // A label alone (before a macro call's words) is no word to fit.
      if (!w.instructions.length) {
        if (other) matched.add(other);
        continue;
      }
      if (!other) {
        out.push(`word ${where(w.start)} [${names.join(' ')}]: in the scan only`);
        continue;
      }
      matched.add(other);
      if (fb.fits(names) !== other.fits) out.push(`word ${where(w.start)} [${names.join(' ')}]: the fallback says it ${fit(fb.fits(names))}, the assembler that it ${fit(other.fits)}`);
    }
    for (const w of answer.words) {
      if (!matched.has(w)) out.push(`word ${where(w.start)} [${w.mnemonics.join(' ')}]: in the assembler's answer only`);
    }
  }

  // --- Jinja ----------------------------------------------------------------------------------
  const regions = jinjaRegions(text);
  const sameRegions = regions.length === answer.jinja.length && regions.every((r, i) => r.start === answer.jinja[i].start && r.end === answer.jinja[i].end);
  if (!sameRegions) out.push(`jinja: the tags differ (${regions.length} found, the assembler ${answer.jinja.length})`);
  else {
    const defs = fb.jinjaDefs(text);
    const names = fb.jinjaNames(text);
    const list = (s) => `[${[...s].sort().join(' ')}]`;
    const differ = (a, b) => a.size !== b.size || [...a].some((x) => !b.has(x));
    for (const r of answer.jinja) {
      const inside = (x) => r.start <= x.start && x.start < r.end;
      const myDefines = new Set(defs.filter(inside).map((d) => d.name));
      const defAt = new Set(defs.filter(inside).map((d) => d.start));
      const myUses = new Set(names.filter((n) => inside(n) && !defAt.has(n.start)).map((n) => n.name));
      const theirDefines = new Set(r.defines);
      const theirUses = new Set(r.uses);
      const tag = `jinja ${where(r.start)} ${quote(text.slice(r.start, r.end).replace(/\s+/g, ' '))}`;
      if (differ(myDefines, theirDefines)) out.push(`${tag}: defines ${list(myDefines)} by the fallback, ${list(theirDefines)} by the assembler`);
      if (differ(myUses, theirUses)) out.push(`${tag}: uses ${list(myUses)} by the fallback, ${list(theirUses)} by the assembler`);
    }
  }
  return out;
}

function main() {
  const args = process.argv.slice(2);
  const write = args.includes('--write-gaps');
  const [isaPath, bundlePath = path.join(__dirname, 'fixtures', 'analysis-v2.json')] = args.filter((a) => a !== '--write-gaps');
  if (!isaPath) {
    console.error('usage: analysis-agreement.js <isa.json> [bundle.json] [--write-gaps]');
    process.exit(2);
  }
  const lang = new Language(JSON.parse(fs.readFileSync(isaPath, 'utf8')));
  const entries = load(bundlePath);
  const stale = entries.filter((e) => e.stale).map((e) => e.name);
  assert.deepStrictEqual(stale, [], 'changed since recorded: re-record with test/record-analysis.js');
  const known = fs.existsSync(GAPS) ? JSON.parse(fs.readFileSync(GAPS, 'utf8')) : {};
  const found = Object.fromEntries(entries.map((e) => [e.name, differences(lang, e.text, e.answer)]));
  if (write) {
    const merged = { ...known, ...Object.fromEntries(Object.entries(found).filter(([, gaps]) => gaps.length)) };
    for (const [name, gaps] of Object.entries(found)) if (!gaps.length) delete merged[name];
    const sorted = Object.fromEntries(Object.keys(merged).sort().map((k) => [k, merged[k]]));
    fs.writeFileSync(GAPS, `${JSON.stringify(sorted, null, 2)}\n`);
    console.log(`Wrote the known gaps of ${Object.keys(sorted).length} files.`);
    return;
  }
  const problems = [];
  let total = 0;
  for (const [name, gaps] of Object.entries(found)) {
    const listed = new Set(known[name] || []);
    total += gaps.length;
    for (const g of gaps) if (!listed.has(g)) problems.push(`${name}: ${g}`);
    for (const g of listed) if (!gaps.includes(g)) problems.push(`${name}: listed as a known gap, but it no longer happens: ${g}`);
  }
  if (problems.length) {
    console.error(`${problems.slice(0, 40).join('\n')}${problems.length > 40 ? `\n… and ${problems.length - 40} more` : ''}`);
    console.error('\nA difference between fallback/ and the assembler\'s analysis that is not a reviewed gap in test/fixtures/analysis-known-gaps.json.');
    process.exit(1);
  }
  console.log(`fallback/ agrees with the assembler's analysis on ${entries.length} sources, but for ${total} known gaps.`);
}

if (require.main === module) main();

module.exports = { differences };
