'use strict';

// Completion: at an instruction start, the mnemonics that still fit the word, each bringing its
// operands as tab stops; at an operand, what its type accepts, labels and fitting Jinja names.
// Slot fit and Jinja names come from the latest analysis (`semantics`, stale by a keystroke or two
// while typing), else from the editor's own rules.

const vscode = require('vscode');

const { provide } = require('./shared');
const { renderInstruction, renderRegister } = require('./hover');

const item = (label, kind, props) => Object.assign(new vscode.CompletionItem(label, vscode.CompletionItemKind[kind]), props);

function register(context, { lang, semantics }) {
  // Instruction docs never change: rendered once per mnemonic, not per request.
  const docs = new Map();
  const docFor = (mnemonic, forms) => {
    if (!docs.has(mnemonic)) docs.set(mnemonic, new vscode.MarkdownString(renderInstruction(lang, mnemonic, forms)));
    return docs.get(mnemonic);
  };
  // --- Completion ---
  provide(context, 'CompletionItem', {
    async provideCompletionItems(document, position, _token, completionContext) {
      const text = document.getText();
      const offset = document.offsetAt(position);
      const ctx = lang.context(text, offset);
      // Space triggers so operand values appear as the operand starts; at an instruction's
      // start it (or the newline after `;;`) must not open the mnemonic list, or Enter would
      // accept an item instead of ending the line. Typing a letter opens it anyway.
      const typedTrigger = completionContext
        && completionContext.triggerKind === vscode.CompletionTriggerKind.TriggerCharacter;
      if (typedTrigger && ctx.kind !== 'operand') return undefined;
      if (ctx.kind === 'comment' || ctx.kind === 'label' || (ctx.kind === 'jinja' && ctx.region.kind === 'comment')) return undefined;
      const known = await semantics(document, { stale: true });
      if (ctx.kind === 'jinja') {
        return known.jinjaVisible(offset)
          .filter((d) => d.start < offset)
          .map((d) => item(d.name, d.kind === 'macro' ? 'Function' : 'Variable', {
            detail: d.kind === 'set' && d.expression ? `= ${d.expression}` : d.kind === 'header' ? 'template parameter' : d.kind,
          }));
      }
      if (ctx.kind === 'mnemonic') {
        // Operands already follow the cursor (a mnemonic being replaced): the name alone. A `;`
        // or a comment after it is no operand.
        const bare = /^\s*(?:[^\s;{]|\{\{)/.test(document.lineAt(position).text.slice(position.character).replace(/^[^\s;]*/, ''));
        return known.mnemonicsFor(ctx, offset).map(({ mnemonic, forms }) => {
          const form = forms[0];
          const operands = bare ? [] : (forms.find((f) => f.operands.length) || form).operands;
          return item(mnemonic, 'Keyword', {
            detail: forms.length > 1 ? 'any slot' : form.pseudo ? `${form.slot} · pseudo → ${form.expandsTo}` : form.slot,
            documentation: docFor(mnemonic, forms),
            // Its operands as tab stops, named as the instruction spec names them.
            ...(operands.length && {
              insertText: new vscode.SnippetString(`${mnemonic} ${operands.map((op, i) => `\${${i + 1}:${op.name}}`).join(' ')}`),
              command: { command: 'editor.action.triggerParameterHints', title: '' },
            }),
          });
        });
      }
      const typed = lang.operandType(ctx.mnemonic, ctx.index);
      if (!typed) return undefined;
      const { domain } = typed;
      // In the assembler's order (lr0 … lr15, not lr0, lr1, lr10, …), after the file's Jinja names.
      const items = (domain.values || []).map((value, i) => {
        const register = lang.registers.has(value.toLowerCase());
        return item(value, register ? 'Variable' : 'EnumMember', {
          sortText: `1${String(i).padStart(4, '0')}`,
          ...(register && { documentation: new vscode.MarkdownString(renderRegister(lang, value.toLowerCase())) }),
        });
      });
      if (domain.kind === 'label') {
        const labels = lang.labels(text);
        const names = new Set(labels.defs.map((d) => d.name));
        for (const name of names) items.push(item(name, 'Reference', { sortText: `2${name}` }));
        // Local labels: the next (`1f`) or nearest previous (`1b`) `1:`.
        for (const n of labels.locals) {
          for (const ref of [`${n}b`, `${n}f`]) items.push(item(ref, 'Reference', { sortText: `3${ref}` }));
        }
      }
      // Template names whose value in force at the cursor this operand accepts: `{{ lr_row }}`.
      // Braces already typed are replaced, not doubled.
      const braces = ctx.prefix.startsWith('{');
      for (const d of known.jinjaVisible(offset)) {
        if (d.kind !== 'set' || !lang.accepts(domain, d.value)) continue;
        items.push(item(`{{ ${d.name} }}`, 'Variable', {
          filterText: braces ? ctx.prefix + d.name : d.name,
          ...(braces && { range: new vscode.Range(document.positionAt(offset - ctx.prefix.length), position) }),
          detail: `= ${d.expression}`,
          sortText: `0${d.name}`, // the file's own names first
        }));
      }
      return items;
    },
  }, ' ');
}

module.exports = { register };
