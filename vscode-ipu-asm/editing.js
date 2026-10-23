'use strict';

// Editing help: one provider module each, registered in this order. Each reads language.js (the
// document's ISA: its checkout's, else data/isa.json; language-source.js) and
// `semantics(document)`: the checkout's assembler's analysis of the document (analysis.js) when
// it has one, else the editor's own rules (fallback/), so all of it works without a toolchain.
// The assembler is also consulted before a rename.

const providers = [
  require('./providers/hover'),
  require('./providers/completion'),
  require('./providers/signature'),
  require('./providers/symbols'),
  require('./providers/rename'),
  require('./providers/format'),
  require('./providers/tokens'),
];

/** Register every provider with `env` ({ langOf, onLanguage, checkText, semantics, answer, onAnswer,
 *  format }); what they return for tests, merged. `langOf(document)`: the Language (language.js)
 *  of the document's ISA; `onLanguage(listener)`: called when one changes. */
const register = (context, env) => Object.assign({}, ...providers.map((p) => p.register(context, env)));

module.exports = { register };
