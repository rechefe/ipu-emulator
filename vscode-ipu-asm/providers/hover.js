'use strict';

// Hover: an instruction's reference and a register's role (from isa.json, built with the
// extension, so always matching), and a Jinja name's value where the cursor is.

const vscode = require('vscode');

const { provide, rangeOf, switched } = require('./shared');

function describeSlot(lang, slot) {
  const meta = (lang.isa.slots || {})[slot];
  return meta && meta.hardware === false ? `${slot} (simulation-only)` : slot;
}

function renderInstruction(lang, mnemonic, forms) {
  const doc = (forms[0] && forms[0].doc) || {};
  const out = [`**${doc.title || mnemonic}**`, ''];

  // A mnemonic can occupy several slots -- NOP exists in all nine, each with its
  // own summary -- so list every form rather than silently showing the first.
  for (const form of forms) {
    const label = form.pseudo ? `pseudo → ${form.expandsTo}` : describeSlot(lang, form.slot);
    const summary = (form.doc && form.doc.summary) || '';
    out.push(`- \`${label}\`${summary ? ` — ${summary}` : ''}`);
  }

  if (doc.syntax) out.push('', '```ipu-asm', doc.syntax, '```');

  if (Array.isArray(doc.operands) && doc.operands.length) {
    out.push('', '**Operands**', '');
    // Already markdown in instruction_spec.py -- pass straight through.
    for (const operand of doc.operands) out.push(`- ${operand}`);
  }
  if (doc.operation) out.push('', `**Operation** — ${doc.operation}`);
  if (doc.example) out.push('', '**Example**', '', '```ipu-asm', doc.example, '```');
  if (doc.notes) out.push('', `**Notes** — ${doc.notes}`);

  return out.join('\n');
}

/** A register's hover, from the simulator's definitions (isa.json `registerDocs`). */
function renderRegister(lang, name) {
  const doc = lang.isa.registerDocs[name] || {};
  const lines = [`**\`${name.toUpperCase()}\`** — ${doc.about || 'register'}`];
  if (doc.pair) lines.push(`\`${doc.pair.map((r) => r.toUpperCase()).join('`:`')}\` as one value.`);
  if (doc.fixed !== undefined) lines.push(`Fixed at ${doc.fixed} (read-only).`);
  return lines.join('\n\n');
}

const jinjaKinds = { macro: 'macro', param: 'macro parameter', for: 'loop variable', import: 'imported name', header: 'template parameter' };

/** A Jinja name's hover: its value where the cursor is, then the reference for that value if
 *  it is a register or an instruction. */
function renderJinja(lang, sym) {
  const def = sym.binding;
  const lines = [def.kind !== 'set' ? `\`${sym.name}\` — Jinja ${jinjaKinds[def.kind]}`
    : def.expression === null ? `\`${sym.name}\` — block \`set\`` : `\`${sym.name}\` = \`${def.expression}\``];
  const value = def.value && def.value.toLowerCase();
  if (value && lang.registers.has(value)) lines.push('', renderRegister(lang, value));
  const instruction = value && lang.lookup(value);
  if (instruction) lines.push('', renderInstruction(lang, instruction.mnemonic, instruction.forms));
  return lines.join('\n');
}

function register(context, { langOf, semantics }) {
  // Hover's word: the assembler's own token shape (mnemonics have dots), per ISA.
  const patterns = new WeakMap();
  const wordPattern = (lang) => {
    if (!patterns.has(lang)) patterns.set(lang, new RegExp(lang.isa.lexical.token));
    return patterns.get(lang);
  };
  provide(context, 'Hover', switched('hover', {
    provideHover(document, position) {
      const lang = langOf(document);
      const range = document.getWordRangeAtPosition(position, wordPattern(lang));
      if (!range) return undefined;
      const key = document.getText(range).toLowerCase();
      const instruction = lang.lookup(key);
      const text = instruction ? renderInstruction(lang, instruction.mnemonic, instruction.forms)
        : lang.registers.has(key) ? renderRegister(lang, key) : null;
      return text ? new vscode.Hover(new vscode.MarkdownString(text), range) : undefined;
    },
  }));
  provide(context, 'Hover', switched('hover', {
    async provideHover(document, position) {
      const sym = (await semantics(document)).symbolAt(document.offsetAt(position));
      if (!sym || sym.kind !== 'jinja') return undefined;
      return new vscode.Hover(new vscode.MarkdownString(renderJinja(langOf(document), sym)), rangeOf(document, sym.range));
    },
  }));
}

module.exports = { register, renderInstruction, renderRegister };
