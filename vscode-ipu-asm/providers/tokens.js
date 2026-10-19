'use strict';

// Semantic tokens, from the assembler's own reading of the source (`--tokens`): which token is a
// mnemonic, a register, an enum value, a number, a label defined or branched to. The grammar
// colours by shape alone (a label spelled like a mnemonic, a register name used as a label);
// these colour by role, over it. Jinja tags are left to the grammar, which colours inside them.
// Without a version-2 checker there are none, and the grammar's colours stand.

const vscode = require('vscode');

const { provide } = require('./shared');

/** The legend: standard token types and modifiers, so every theme colours them. */
const TYPES = ['keyword', 'variable', 'enumMember', 'number', 'function', 'comment'];
const MODIFIERS = ['declaration', 'readonly'];
/** Contract role -> [type, modifiers]. A role not listed (`jinja`, or one a later contract adds)
 *  is not sent. */
const ROLES = {
  mnemonic: ['keyword', []],
  register: ['variable', ['readonly']],
  enum: ['enumMember', []],
  number: ['number', []],
  'label-def': ['function', ['declaration']],
  'label-ref': ['function', []],
  comment: ['comment', []],
};

/** An answer's tokens as [line, character, length, type index, modifier bits], in order. */
function encode(answer) {
  const out = [];
  for (const t of answer.tokens) {
    const role = ROLES[t.role];
    if (!role || t.length <= 0) continue;
    const [type, modifiers] = role;
    out.push([t.line, t.character, t.length, TYPES.indexOf(type), modifiers.reduce((bits, m) => bits | (1 << MODIFIERS.indexOf(m)), 0)]);
  }
  return out;
}

function register(context, { answer, onAnswer }) {
  const legend = new vscode.SemanticTokensLegend(TYPES, MODIFIERS);
  // A document asked about before its answer came: asked again once it has one.
  const missed = new Set();
  const changed = new vscode.EventEmitter();
  const provider = {
    onDidChangeSemanticTokens: changed.event,
    async provideDocumentSemanticTokens(document) {
      const found = await answer(document);
      if (!found) {
        missed.add(document.uri.toString());
        return undefined;
      }
      const builder = new vscode.SemanticTokensBuilder(legend);
      for (const t of encode(found)) builder.push(...t);
      return builder.build();
    },
  };
  provide(context, 'DocumentSemanticTokens', provider, legend);
  context.subscriptions.push(changed, onAnswer((key, found) => {
    if (found && missed.delete(key)) changed.fire();
  }));
  return { semanticTokens: provider, legend };
}

module.exports = { register, encode, TYPES, MODIFIERS, ROLES };
