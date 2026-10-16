'use strict';

// The editor's verdict on a whole text: the reasons language.js would give for it not to assemble,
// none when it accepts it. check_verdicts.py holds it against the assembler's (`ipu-as check`):
// accepts on one side exactly when it accepts on the other, but for the reviewed allowed gaps.
//
// Only what language.js knows from isa.json: unknown mnemonics, operand counts and values, label
// references, words that do not fit their slots, and Jinja blocks left open. Text inside a
// template (`{{ }}`) is not judged; the template decides it.

/** Why `lang` would reject `text`, as a list of reasons; empty when it accepts it. */
function verdict(lang, text) {
  const reasons = [];
  for (const block of lang.pairedBlocks(text)) if (block.end === null) reasons.push(`{% ${block.name} %} is never closed`);
  const { defs, locals } = lang.labels(text);
  const labelAccepts = (name) => /^[+-]\d+$/.test(name)
    || (/^[0-9]+[fb]$/.test(name) && locals.has(String(Number(name.slice(0, -1)))))
    || defs.some((d) => lang.sameLabel(d.name, name));
  for (const word of lang.scan(text).words) {
    const known = [];
    for (const ins of word.instructions) {
      if (ins.mnemonic.templated) continue;
      const name = ins.mnemonic.text;
      const entry = lang.lookup(name);
      if (!entry) {
        reasons.push(`${name}: no such instruction`);
        continue;
      }
      known.push(name);
      if (!entry.forms.some((f) => f.operands.length === ins.operands.length)) {
        reasons.push(`${name}: ${ins.operands.length} operands, no form takes that many`);
        continue;
      }
      ins.operands.forEach((op, i) => {
        if (op.templated) return;
        const typed = lang.operandType(name, i);
        const ok = typed && (typed.domain.kind === 'label' ? labelAccepts(op.text) : lang.accepts(typed.domain, op.text));
        if (!ok) reasons.push(`${name}: operand ${i} "${op.text}" is not accepted`);
      });
    }
    if (!lang.fits(known)) reasons.push(`the word at ${word.start} does not fit its slots: ${known.join('; ')}`);
  }
  return reasons;
}

module.exports = { verdict };
