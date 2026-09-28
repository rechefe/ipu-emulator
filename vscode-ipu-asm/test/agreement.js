'use strict';

// Prove the generated TextMate grammar agrees with the real Lark parser.
//
// The previous grammar was hand-written from reading asm_grammar.lark, and it
// drifted: it coloured `BEQ lr0, cr0, +1;;` as valid (it is a lex error) and
// failed to see `end:` in `BKPT;; end: BKPT;;` as a label. Nothing detected
// either, because nothing compared the two.
//
// This tokenizes the same text twice — once with the parser (via
// dump_parser_tokens.py) and once with the exact engine VS Code ships — and
// fails on any disagreement about a token's boundaries or its role.
//
//   node agreement.js <parser-tokens.json> <grammar.tmLanguage.json>

const fs = require('fs');
const path = require('path');
const vsctm = require('vscode-textmate');
const oniguruma = require('vscode-oniguruma');

// The grammar has a single catch-all TOKEN terminal, so the parser cannot tell
// a mnemonic from a register from a label lexically — but its parse TREE can,
// by position. Those structural roles are the contract worth checking, and they
// are exactly what a regex-only grammar has to reconstruct.
const ROLE_SCOPE = {
  LABEL: 'entity.name.label',
  MNEMONIC: 'keyword.other.mnemonic',
};

// Terminals that do carry a scope of their own regardless of position.
const TERMINAL_SCOPE = {
  COMMENT_LINE: 'comment.line.double-slash',
  COMMENT_HASH: 'comment.line.number-sign',
  _SEMI2: 'punctuation.terminator.bundle',
  _SEMI: 'punctuation.separator.slot',
};

// Whitespace carries no scope of its own.
const UNSCOPED_TERMINALS = new Set(['WS_INLINE', '_NL']);

// Jinja regions of the RAW file -> the scope each must carry. The parser only sees the rendered
// text, so these are checked against Jinja's own reading (jinja_regions in dump_parser_tokens.py).
const JINJA_SCOPE = { comment: 'comment.block.jinja', block: 'meta.embedded.block.jinja', variable: 'meta.embedded.line.jinja' };

// An operand must NOT be coloured as an instruction. Asserting this negative is
// what catches a keyword pattern missing its word boundary: `bne_target` is an
// operand, and a rule matching `BNE` inside it would otherwise go unnoticed,
// since no positive assertion covers a plain identifier operand.
const OPERAND_FORBIDDEN_SCOPES = ['keyword.other.mnemonic'];

async function loadGrammar(grammarPath) {
  const wasm = fs.readFileSync(require.resolve('vscode-oniguruma/release/onig.wasm'));
  const onigLib = oniguruma.loadWASM(wasm.buffer).then(() => ({
    createOnigScanner: (patterns) => new oniguruma.OnigScanner(patterns),
    createOnigString: (s) => new oniguruma.OnigString(s),
  }));

  const raw = JSON.parse(fs.readFileSync(grammarPath, 'utf8'));
  const registry = new vsctm.Registry({ onigLib, loadGrammar: async () => raw });
  const grammar = await registry.loadGrammar(raw.scopeName);
  if (!grammar) throw new Error(`could not load grammar ${raw.scopeName}`);
  return grammar;
}

/** Tokenize whole text, returning absolute-offset tokens. */
function tokenize(grammar, text) {
  const out = [];
  let stack = vsctm.INITIAL;
  let offset = 0;

  for (const line of text.split('\n')) {
    const result = grammar.tokenizeLine(line, stack);
    // Without this a timeLimit hit truncates silently and the test passes vacuously.
    if (result.stoppedEarly) throw new Error('tokenizer hit its time limit');
    for (const t of result.tokens) {
      out.push({
        start: offset + t.startIndex,
        end: offset + t.endIndex,
        scopes: t.scopes,
      });
    }
    stack = result.ruleStack;
    offset += line.length + 1; // +1 for the '\n' removed by split
  }
  return out;
}

/** Disagreements between the raw file's Jinja regions and the grammar, once per line at its first
 *  wrong character. A region must carry its Jinja scope (a template in an assembly comment is still
 *  live code), and text outside every region none (an unclosed begin rule swallows the file). */
function jinjaFailures(grammar, entry) {
  const raw = entry.raw;
  const want = new Array(raw.length).fill(null); // the region kind at each character
  for (const r of entry.jinja) want.fill(r.kind, r.start, r.end);
  const got = new Array(raw.length).fill('');
  for (const t of tokenize(grammar, raw)) got.fill(t.scopes.join(' '), t.start, t.end);
  const failures = [];
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] === '\n' || (want[i] ? got[i].includes(JINJA_SCOPE[want[i]]) : !got[i].includes('jinja'))) continue;
    const line = raw.slice(0, i).split('\n').length;
    failures.push(`${path.basename(entry.file)}:${line}  ${JSON.stringify(raw.slice(i).split('\n')[0].slice(0, 40))}  ` +
      `Jinja says ${want[i] ? `Jinja ${want[i]}` : 'assembly'}, grammar gave [${got[i] || 'none'}]`);
    const next = raw.indexOf('\n', i);
    i = next < 0 ? raw.length : next;
  }
  return failures;
}

/** Scopes covering [start,end), flattened. */
function scopesAt(tmTokens, start, end) {
  const scopes = new Set();
  for (const t of tmTokens) {
    if (t.end > start && t.start < end) t.scopes.forEach((s) => scopes.add(s));
  }
  return scopes;
}

function hasScope(scopes, prefix) {
  for (const s of scopes) if (s.startsWith(prefix)) return true;
  return false;
}

async function main() {
  const [tokensPath, grammarPath] = process.argv.slice(2);
  if (!tokensPath || !grammarPath) {
    console.error('usage: agreement.js <parser-tokens.json> <grammar.tmLanguage.json>');
    process.exit(2);
  }

  const grammar = await loadGrammar(grammarPath);
  const corpus = JSON.parse(fs.readFileSync(tokensPath, 'utf8'));

  const failures = [];
  let checked = 0;
  let jinjaRegions = 0;

  for (const entry of corpus) {
    const name = path.basename(entry.file);
    const tmTokens = tokenize(grammar, entry.text);

    failures.push(...jinjaFailures(grammar, entry));
    jinjaRegions += entry.jinja.length;

    for (const tok of entry.tokens) {
      if (UNSCOPED_TERMINALS.has(tok.terminal)) continue;

      const expected = ROLE_SCOPE[tok.role] ?? TERMINAL_SCOPE[tok.terminal];
      const isOperand = tok.role === 'OPERAND';
      if (!expected && !isOperand) continue;

      checked++;
      const scopes = scopesAt(tmTokens, tok.start, tok.end);

      // Jinja regions are rendered away before the parser sees them, so a
      // token inside one has no counterpart in the raw editor text.
      if (hasScope(scopes, 'comment.block.jinja')) continue;

      if (isOperand) {
        const wrong = OPERAND_FORBIDDEN_SCOPES.filter((s) => hasScope(scopes, s));
        if (wrong.length) {
          const line = entry.text.slice(0, tok.start).split('\n').length;
          failures.push(
            `${name}:${line}  ${JSON.stringify(tok.text)}  ` +
              `parser says operand, but grammar scoped it as [${wrong.join(', ')}]`
          );
        }
        continue;
      }

      if (!hasScope(scopes, expected)) {
        const line = entry.text.slice(0, tok.start).split('\n').length;
        failures.push(
          `${name}:${line}  ${JSON.stringify(tok.text)}  ` +
            `parser says ${tok.role || tok.terminal}, expected scope ${expected}, ` +
            `grammar gave [${[...scopes].join(', ') || 'none'}]`
        );
      }
    }
  }

  if (failures.length) {
    console.error(`Grammar disagrees with the parser or Jinja on ${failures.length} token(s):\n`);
    for (const f of failures.slice(0, 25)) console.error('  ' + f);
    if (failures.length > 25) console.error(`  … and ${failures.length - 25} more`);
    console.error(
      '\nThe TextMate grammar is generated from the parser, so a mismatch means ' +
        'gen_vscode.py mapped a terminal (or a Jinja rule) wrongly.'
    );
    process.exit(1);
  }

  console.log(
    `Grammar agrees with the parser: ${checked} tokens across ${corpus.length} files.`
  );
  console.log(`Grammar agrees with Jinja: ${jinjaRegions} template regions.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
