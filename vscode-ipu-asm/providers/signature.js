'use strict';

// Operand hints (signature help): the instruction's forms, the operand being typed highlighted.

const vscode = require('vscode');

const { provide } = require('./shared');

/** One line summarising what an operand type accepts. */
function describeDomain(domain) {
  if (!domain) return '';
  const range = 'min' in domain ? `${domain.min} … ${domain.max}` : '';
  const values = (domain.values || []).join(', ');
  const kinds = {
    enum: `One of: ${values}`,
    number: `Integer, ${range}`,
    mixed: `Integer ${range}, or one of: ${values}`,
    label: `A label defined in this file, or +N (a relative offset up to +${domain.relative_max})`,
  };
  return Object.hasOwn(kinds, domain.kind) ? kinds[domain.kind] : '';
}

function register(context, { langOf }) {
  provide(context, 'SignatureHelp', {
    provideSignatureHelp(document, position) {
      const lang = langOf(document);
      const ctx = lang.context(document.getText(), document.offsetAt(position));
      if (ctx.kind !== 'operand') return undefined;
      const entry = lang.lookup(ctx.mnemonic);
      if (!entry) return undefined;
      const help = new vscode.SignatureHelp();
      for (const form of entry.forms) {
        if (!form.operands.length) continue;
        let label = entry.mnemonic;
        const parameters = form.operands.map((op, i) => {
          const start = label.length + 1;
          label += ` ${op.name}`;
          const domain = lang.isa.operandTypes[op.type];
          const docLine = ((form.doc && form.doc.operands) || [])[i] || '';
          const md = [docLine, `\`${op.type}\` — ${describeDomain(domain)}`, domain && domain.description].filter(Boolean);
          return new vscode.ParameterInformation([start, start + op.name.length], new vscode.MarkdownString(md.join('\n\n')));
        });
        help.signatures.push(Object.assign(new vscode.SignatureInformation(label, (form.doc && form.doc.summary) || ''), { parameters }));
      }
      if (!help.signatures.length) return undefined;
      help.activeSignature = 0;
      help.activeParameter = Math.min(ctx.index, help.signatures[0].parameters.length - 1);
      return help;
    },
  }, { triggerCharacters: [' '], retriggerCharacters: [' '] });
}

module.exports = { register };
