'use strict';

// Editing help: one provider module each, registered in this order. Each reads language.js (the
// ISA from data/isa.json) and `semantics(document)`: the checkout's assembler's analysis of the
// document (analysis.js) when it has one, else the editor's own rules (fallback/), so all of it
// works without a toolchain. The assembler is also consulted before a rename.

const providers = [
  require('./providers/hover'),
  require('./providers/completion'),
  require('./providers/signature'),
  require('./providers/symbols'),
  require('./providers/rename'),
  require('./providers/format'),
  require('./providers/tokens'),
];

/** Register every provider with `env` ({ lang, checkText, semantics, answer, onAnswer, format }); what they
 *  return for tests, merged. */
const register = (context, env) => Object.assign({}, ...providers.map((p) => p.register(context, env)));

module.exports = { register };
