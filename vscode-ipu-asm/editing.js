'use strict';

// Editing help that needs no toolchain, all on language.js (the ISA from data/isa.json): one
// provider module each, registered in this order. The assembler is consulted only where a
// guess would be unsafe: before a rename.

const providers = [
  require('./providers/hover'),
  require('./providers/completion'),
  require('./providers/signature'),
  require('./providers/symbols'),
  require('./providers/rename'),
  require('./providers/format'),
];

/** Register every provider with `env` ({ lang, checkText }); what they return for tests, merged. */
const register = (context, env) => Object.assign({}, ...providers.map((p) => p.register(context, env)));

module.exports = { register };
