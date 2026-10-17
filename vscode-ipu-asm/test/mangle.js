'use strict';

/** A layout-only rewrite for formatter tests: indentation stripped and
 *  consecutive words joined onto one line, only where no comment or Jinja on
 *  the line could swallow what follows. */
function mangle(lang, raw) {
  const codeLines = lang.mask(raw).masked.split('\n');
  const out = [];
  raw.split('\n').forEach((line, i) => {
    const simple = codeLines[i].trim() && codeLines[i] === line; // no comment, no Jinja
    const prev = out[out.length - 1];
    if (simple && prev && prev.simple && prev.text.trimEnd().endsWith(lang.wordEnd)) prev.text = `${prev.text.trimEnd()} ${line.trim()}`;
    else out.push({ text: simple ? line.trim() : line, simple });
  });
  return out.map((o) => o.text).join('\n');
}

/** How many rewrites mutations() makes of one word, at most. */
const MUTATIONS_PER_WORD = 4;

/** Rewrites that may change whether a text assembles, for both sides to judge with one verdict
 *  rule (verdicts.js, check_verdicts.py). Seeded by position, so a run is reproducible: in up to
 *  `limit` words spread over the text, the word's first instruction copied into it (a slot kind
 *  over-full or not), its last operand dropped, and its first number operand negated as a signed
 *  hex literal and made too wide for any operand. Each is `{ name, text }`. */
function mutations(lang, text, limit = 4) {
  const words = lang.scan(text).words.filter((w) => w.instructions.length && !w.instructions[0].mnemonic.templated);
  const step = Math.max(1, Math.floor(words.length / limit));
  const out = [];
  const splice = (start, end, insert) => text.slice(0, start) + insert + text.slice(end);
  for (let k = 0; k < words.length && out.length < MUTATIONS_PER_WORD * limit; k += step) {
    const w = words[k];
    const ins = w.instructions[0];
    const end = ins.operands.length ? ins.operands[ins.operands.length - 1].end : ins.mnemonic.end;
    const at = `word ${k} (${ins.mnemonic.text})`;
    out.push({ name: `${at}: doubled`, text: splice(end, end, `${lang.slotSep} ${text.slice(ins.mnemonic.start, end)}`) });
    if (ins.operands.length) {
      const last = ins.operands[ins.operands.length - 1];
      out.push({ name: `${at}: last operand dropped`, text: splice(ins.mnemonic.end, end, text.slice(ins.mnemonic.end, last.start).trimEnd()) });
    }
    const number = ins.operands.find((op) => !op.templated && lang.numberRe.test(op.text));
    if (number) {
      const value = lang.numberValue(number.text);
      const hex = (n) => `${n < 0n ? '+' : '-'}0x${(n < 0n ? -n : n).toString(16)}`;
      out.push({ name: `${at}: ${number.text} negated`, text: splice(number.start, number.end, hex(value)) });
      out.push({ name: `${at}: ${number.text} too wide`, text: splice(number.start, number.end, `0x1${'0'.repeat(20)}`) });
    }
  }
  return out;
}

module.exports = { mangle, mutations };
