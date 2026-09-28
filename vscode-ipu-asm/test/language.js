'use strict';

// language.js against small cases with known answers and, given parser-tokens.json (the dump the
// agreement test uses), against the parser on every corpus file.
//
//   node test/language.js <data/isa.json> [parser-tokens.json]

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const { Language, jinjaRegions } = require('../language');
const { mangle } = require('./mangle');
const { test, main } = require('./harness');

const [isaPath, corpusPath] = process.argv.slice(2);
if (!isaPath) {
  console.error('usage: language.js <isa.json> [parser-tokens.json]');
  process.exit(2);
}
const isa = JSON.parse(fs.readFileSync(isaPath, 'utf8'));
const lang = new Language(isa);
const eq = assert.deepStrictEqual;

/** [text, offset] for `marked`, whose `|` marks the offset. */
const at = (marked) => [marked.replace('|', ''), marked.indexOf('|')];
const contextAt = (marked) => lang.context(...at(marked));
/** The `set` definition of `name` in force at the `|`, or null where `name` is no `set`. */
const valueAt = (marked, name) => {
  const [text, offset] = at(marked);
  const d = lang.jinjaBinding(text, name, offset);
  return d && d.kind === 'set' ? d : null;
};

// --- cursor context, slots, operand values -----------------------------------------------

test('the cursor context: mnemonic, operand index, comment or Jinja', () => {
  for (const [marked, expected] of [
    ['SET lr0 cr0;;\n    |', { kind: 'mnemonic' }],
    ['MULT.RC.VE lr15 lr5 |', { kind: 'operand', mnemonic: 'MULT.RC.VE', index: 2 }],
    ['loop: BLT lr1 |', { mnemonic: 'BLT', index: 1 }], // a label is not the mnemonic
    ['# SET |', { kind: 'comment' }], ['// SET |', { kind: 'comment' }], ['{% set x = | %}', { kind: 'jinja' }],
    ['{% set x = |', { kind: 'jinja' }], ['SET lr0 {{ lr|', { kind: 'jinja' }], // unclosed at the end of the file
    ['SET {{ lr_row }} |', { index: 1 }], // a {{ }} operand is one operand
    ['SET {{ lr_row\n }} |', { index: 1 }], // even spanning lines
    ['SET {% if a %}lr0{% else %}lr1{% endif %} cr0|;;', { index: 1 }], // alternatives keep their position
    // asm_packed_output_linear_*.asm: a templated label closed in another iteration.
    ['{% for w in ws %}\nskip_{{ w }}:\n{% endfor %}\ndone:\n    SET |lr0 cr0;;\n', { mnemonic: 'SET' }],
  ]) {
    const ctx = contextAt(marked);
    for (const [k, v] of Object.entries(expected)) assert.strictEqual(ctx[k], v, `${JSON.stringify(marked)}: ${k}`);
  }
});

test('only mnemonics that fit the word\'s free slots are offered; operand types and their values', () => {
  // maxpool2d_nms*.asm: if / else alternatives are alternatives, not a mnemonic and its operand.
  const alternatives = 'MULT.VE lr0 cr1 0 lr0 cr15 ; {% if a %}ACC.MAX.FIRST{% else %}ACC.MAX|{% endif %} ;;';
  for (const [marked, absent, present] of [
    ['SET lr0 cr0; SET lr1 cr0; SET lr2 cr0; |', ['SET'], ['BKPT']], // three fill the LR slots
    ['| ; BLT lr0 cr4 top;;', ['BEQ', 'B'], ['SET']], // pseudo B also expands to cond
    [alternatives, [], ['ACC.MAX']],
  ]) {
    const names = lang.mnemonicsFor(contextAt(marked).others).map((m) => m.mnemonic);
    for (const m of absent) assert(!names.includes(m), `${marked}: ${m} offered`);
    for (const m of present) assert(names.includes(m), `${marked}: ${m} not offered`);
  }
  const ctx = contextAt(alternatives);
  eq([ctx.kind, ctx.others.before], ['mnemonic', ['MULT.VE']], 'the other branch is not in this word');
  // CompoundInst._fill_instructions: in source order, NOP takes the first free slot in slotOrder -- cond.
  eq([lang.fits(['NOP', 'BKPT']), lang.fits(['BKPT', 'NOP'])], [false, true]);
  // Operand types come from isa.json; accepts follows the domain, case and number bases.
  const t = lang.operandType('ACTIVATE.QUANTIZE', 0);
  assert(t.operand.type === 'ActivationFn' && t.domain.values.includes('window'));
  assert.strictEqual(lang.operandType('BKPT', 0), null);
  const lr = isa.operandTypes.LrIdx;
  assert(lang.accepts(lr, 'LR3') && !lang.accepts(lr, 'cr3'));
  const inc = isa.operandTypes.LrIncDecImmediate;
  assert(lang.accepts(inc, '0x10') && lang.accepts(inc, String(inc.max)) && !lang.accepts(inc, String(inc.max + 1)));
});

// --- labels, Jinja names, rename, symbols, folding ---------------------------------------------

test('label definitions and references, as the assembler reads them', () => {
  for (const [text, defs, refs] of [
    ['top:\n    SET lr0 cr0;;\n    BLT lr0 cr4 top;;\n    B +1;;\n', ['top'], ['top']], // `+1` is an offset
    ['{{ name }}_loop:\n    B {{ name }}_loop;;\n', [], []], // templated labels are no literal names
    // softmax_rows_partial.asm: a macro call where an instruction starts is whole words.
    ['run_p1:\n{{- block(0, true) }}\n    B drain ;;\ndrain:\n    BKPT ;;\n', ['run_p1', 'drain'], ['drain']],
    ['{% for w in ws %}\nskip_{{ w }}:\n{% endfor %}\ndone:\n    SET lr0 cr0;;\n', ['done'], []],
  ]) {
    const found = lang.labels(text);
    eq(found.defs.map((d) => d.name), defs, text);
    eq(found.refs.map((r) => r.name), refs, text);
  }
  assert.strictEqual(lang.sameLabel('Top', 'top'), isa.lexical.caseInsensitive.labels);
});

test('set, macro, parameter and for definitions, with literal values', () => {
  const text = '{%- set lr_row = "lr1" -%}\n{% set n = 4 %}\n{% set e = a + 1 %}\n{% macro step(x) %}{% endmacro %}\n{% for i, j in pairs %}{% endfor %}';
  const { lr_row: row, n, e, step, i, j } = Object.fromEntries(lang.jinjaDefs(text).map((d) => [d.name, d]));
  eq([row.value, n.value, e.value, e.expression, step.kind], ['lr1', '4', null, 'a + 1', 'macro']);
  assert(i && j);
  assert.strictEqual(text.slice(row.start, row.end), 'lr_row');
  const tuple = '{% set a, b = 1, 2 %}{% macro m(x, y=3) %}{{ x }}{% endmacro %}{{ a }}{{ b }}';
  const names = lang.jinjaDefs(tuple).map((d) => `${d.kind}:${d.name}`);
  for (const n of ['set:a', 'set:b', 'macro:m', 'param:x', 'param:y']) assert(names.includes(n), n);
  assert(lang.symbolAt(tuple, tuple.lastIndexOf('{{ a }}') + 3));
  const params = '{% macro m(a, b=f(1, c), d="x,y") %}{% endmacro %}'; // names, not their defaults
  eq(lang.jinjaDefs(params).filter((d) => d.kind === 'param').map((d) => d.name), ['a', 'b', 'd']);
});

test('the value in force is the last set before the cursor, scoped to its loop', () => {
  const loop = '{% set x = 1 %}{% for i in range(2) %}{% set x = 2 %}{{ x }}{% endfor %}{{ x }}';
  for (const [marked, name, value] of [
    ['{% set r = "lr1" %}\nSET |{{ r }} cr0;;\n{% set r = "lr2" %}\nSET {{ r }} cr0;;', 'r', 'lr1'],
    ['{% set r = "lr1" %}\nSET {{ r }} cr0;;\n{% set r = "lr2" %}\nSET |{{ r }} cr0;;', 'r', 'lr2'],
    ['{% set r = "lr1" %}{% set |r = "lr2" %}', 'r', 'lr2'], // at a redefinition, the new one
    [loop.replace('{{ x }}', '{{ |x }}'), 'x', '2'],
    [`${loop.slice(0, -4)}|x }}`, 'x', '1'],
  ]) assert.strictEqual(valueAt(marked, name).value, value, marked);
  assert.strictEqual(lang.jinjaBinding(loop, 'i', loop.lastIndexOf('{{ x }}')), null);
  eq(lang.jinjaVisible(loop, loop.indexOf('{{ x }}')).map((d) => `${d.kind}:${d.name}:${d.value}`).sort(), ['for:i:undefined', 'set:x:2']);
});

test('name uses skip strings, attributes and Jinja words; symbolAt finds every use of a name or label', () => {
  const uses = lang.jinjaNames('{% for x in items if x.lr_row %}{{ "lr_row" }}{{ lr_row }}{% endfor %}').map((u) => u.name);
  assert.strictEqual(uses.filter((u) => u === 'lr_row').length, 1);
  assert(!uses.includes('for') && !uses.includes('in') && !uses.includes('if'));
  const text = '{% set r = "lr1" %}\ntop: SET {{ r }} cr0;;\n    BLT {{ r }} cr4 top;;';
  const jinja = lang.symbolAt(text, text.indexOf('{{ r }}') + 3);
  const label = lang.symbolAt(text, text.lastIndexOf('top') + 1);
  eq([jinja.kind, jinja.refs.length, label.kind, label.refs.length], ['jinja', 3, 'label', 2]);
});

test('rename refuses names that cannot be tokens, and Jinja words and constants in either case', () => {
  for (const [kind, names, ok] of [
    ['label', ['row_loop2'], true],
    ['label', ['+3', 'has space'], false], // a leading sign makes a relative offset
    ['jinja', ['lr_row'], true],
    ['jinja', ['for', 'a.b', 'None', 'none', 'True', 'false', 'endset'], false],
  ]) for (const n of names) assert.strictEqual(Boolean(lang.validName(kind, n)), ok, `${kind} ${n}`);
});

test('a macro parameter hides the top-level name, and each macro has its own', () => {
  const text = '{% set r = "lr0" %}\n{% macro m(r) %}SET {{ r }} cr0 ;;{% endmacro %}\n'
    + '{% macro n(r) %}SET {{ r }} cr1 ;;{% endmacro %}\nSET {{ r }} cr2 ;;';
  const inM = text.indexOf('{{ r }}') + 3;
  const last = text.lastIndexOf('{{ r }}') + 3;
  const mark = (offset) => `${text.slice(0, offset)}|${text.slice(offset)}`;
  assert.strictEqual(lang.jinjaBinding(text, 'r', inM).kind, 'param');
  assert.strictEqual(valueAt(mark(inM), 'r'), null);
  assert.strictEqual(valueAt(mark(last), 'r').value, 'lr0');
  // m's parameter is used in m only; the top-level name outside the macros only.
  const sym = lang.symbolAt(text, inM);
  eq(sym.refs.map((r) => r.start), [text.indexOf('m(r') + 2, inM]);
  eq(lang.symbolAt(text, text.indexOf('r =')).refs.map((r) => r.start), [text.indexOf('r ='), last]);
  // Another macro's parameter of the same name is no clash; the top-level name is.
  assert.strictEqual(lang.jinjaClash(text, lang.symbolAt(text, text.indexOf('n(r') + 2), 'q'), false);
  assert.strictEqual(lang.symbolAt(text, text.indexOf('{{ r }}', inM) + 3).refs.length, 2);
  const mText = text.replace('m(r)', 'm(q)');
  assert.strictEqual(lang.jinjaClash(mText, lang.symbolAt(mText, mText.indexOf('n(r') + 2), 'q'), false);
  assert.strictEqual(lang.jinjaClash(text, sym, 'm'), true);
  // Nor may a rename take a name the file uses without defining it.
  const uses = '{% for i in range(2) %}{{ i }} {{ rows | default(8) }}{% endfor %}';
  const i = lang.symbolAt(uses, uses.indexOf('i in'));
  eq([lang.jinjaClash(uses, i, 'rows'), lang.jinjaClash(uses, i, 'col')], [true, false]);
});

test('a set after a parameter replaces it from there on; a loop header is read outside the loop', () => {
  const text = '{% macro inc(dest, n) %}{%- set n = n % 512 -%}{{ n }}{% endmacro %}';
  const kind = (offset) => lang.jinjaBinding(text, 'n', offset).kind;
  eq([kind(text.lastIndexOf('{{ n }}') + 3), kind(text.indexOf('n % 512'))], ['set', 'param']);
  const loop = '{% set i=5 %}{% for i in range(i) %}{{ i }}{% endfor %}';
  eq(lang.symbolAt(loop, loop.indexOf('i in')).refs.map((r) => r.start), [loop.indexOf('i in'), loop.indexOf('{{ i }}') + 3]);
  assert.strictEqual(lang.symbolAt(loop, loop.indexOf('range(i)') + 6).binding.kind, 'set');
});

test('a {{ name }} shows the literal value in force there; unused labels and sets, conservatively', () => {
  const text = '{% set r = "lr0" %}{% set n = 3 %}{% macro m(r) %}SET {{ r }} cr0;;{% endmacro %}\n'
    + 'SET {{ r }} cr0;;\n{% set r = "lr1" %}SET {{- r -}} cr0;;\nSET {{ r ~ "x" }} cr0;;\n';
  // Not inside the macro (its parameter hides the set) nor on an expression.
  eq(lang.jinjaHints(text).map((h) => [text.slice(text.lastIndexOf('{{', h.at), h.at), h.value]), [['{{ r }}', 'lr0'], ['{{- r -}}', 'lr1']]);
  const unused = (t) => lang.unused(t).map((u) => `${u.kind}:${u.name}`);
  eq(unused('{% set a = 1 %}{% set b = 2 %}{% set n = 0 %}{% set n = n + 1 %}{% set c = 1 %}{% set c = 2 %}\n'
    + 'top:\nend:\n    BLT lr0 cr0 top ;;\n    SET {{ b }} cr0 ;;\n'), ['label:end', 'set:a', 'set:c', 'set:c']);
  eq(unused('{# loop: see far_loop #}\nfar_loop:\n{% set t = "far_loop" %}{{ t }}\n'), []); // named anywhere else: used
});

test('symbols list labels, sets and macros in order; folding pairs blocks and multi-line comments', () => {
  eq(lang.symbols('{% set r = "lr1" %}\n{% macro m() %}{% endmacro %}\ntop: BKPT;;').map((s) => `${s.kind}:${s.name}`),
    ['set:r', 'macro:m', 'label:top']);
  const text = '{#\n header\n#}\n{% for i in range(2) %}\n{% if i %}\nBKPT;;\n{% endif %}\n{% endfor %}\n{# one line #}';
  const folds = lang.folding(text).map((f) => [f.kind, text.slice(f.start, f.start + 6)]).sort();
  eq(folds, [['comment', '{#\n he'], ['region', '{% for'], ['region', '{% if ']]);
});

// --- formatting and word separation -------------------------------------------------------------

const [B, NO, BL] = ['    BKPT;;', { blankLine: false }, { blankLine: true }];
const words = 'top:\n    SET lr0 cr0;\n    SET lr1 cr0;;\n    BKPT;;';
const forLoop = 'BKPT;;\n{% for i in range(2) %}\nBKPT;;\n{% endfor %}\nBKPT;;';
const spanning = 'SET lr0 {{ f(a,\n b) }};;\nBKPT;;\nBKPT;;\nBKPT;;';
const callAfter = '    BKPT;; {{ block(0) }}\n    BKPT;;';
const callAlone = `${B}\n\n    {{ block(0) }}\n\n${B}`;
const tagAfter = '    BKPT;; {% set s = "a\nb" %}\n    BKPT;;';

test('Format Document and Format Selection', () => {
  // [input, options, output (undefined: unchanged)]
  for (const [text, options, expected] of [
    ['  top:\nSET lr0 cr0;;', NO, 'top:\n    SET lr0 cr0;;'], // labels at column 0, instructions one level in
    ['\t\tBKPT;;', { indent: '\t' }, '\tBKPT;;'],
    ['BKPT;; BKPT;;', NO, `${B}\n${B}`], // a second word after ;; moves to its own line
    ['a: BKPT;; b: BKPT;;', NO, `a:\n${B}\nb:\n${B}`],
    [words, BL, 'top:\n    SET lr0 cr0;\n    SET lr1 cr0;;\n\n    BKPT;;'], // an empty line between words, not in one
    [words, undefined, undefined], // off by default
    ['BKPT;;\n\n\nBKPT;;', BL, `${B}\n\n\n${B}`], // existing blank lines stay
    ['{%- set r = "lr0" -%}\n# note\n    MULT.VE {{r}}   cr1 0 {{r}} cr15 ;     {#- aligned -#}\n    ACC.ADD.FIRST ;; // done', NO, undefined], // spacing kept
    ['    BKPT;; # BKPT;; more', undefined, undefined], // code after ;; in a comment or tag is not split off
    ['    BKPT;; {# x;; y #}', undefined, undefined],
    ['{%- macro m(a,\n      b) -%}\n{{ block(0) }}\n{% endmacro %}', undefined, undefined], // multi-line tags and calls
    ['  a:\nBKPT;;\nBKPT;;', { fromLine: 1, toLine: 1, blankLine: false }, B], // a range: its lines only, but it sees
    ['BKPT;;\nBKPT;;\nBKPT;;', { blankLine: true, fromLine: 1, toLine: 2 }, `\n${B}\n\n${B}`], // the word before it
    ['BKPT;;\nBKPT;;\nBKPT;;', { blankLine: true, fromLine: 1, toLine: 1 }, `\n${B}`],
    ['top: BKPT;; BKPT;;  \r\nBKPT;;\r\n', undefined, `top:\r\n${B}\r\n${B}\r\n${B}\r\n`], // CRLF stays CRLF
    ['  top:  # c \r\n', undefined, 'top:  # c\r\n'],
    ['a: BKPT;;\r\nBKPT;; BKPT;;\nBKPT;;', undefined, `a:\r\n${B}\r\n${B}\n${B}\n${B}`], // mixed: each line its own
    // A comment or Jinja block goes with the word after it; a closing tag with the word before.
    ['BKPT;;\n# next\nBKPT;;', BL, `${B}\n\n# next\n${B}`],
    [forLoop, BL, `${B}\n\n{% for i in range(2) %}\n${B}\n{% endfor %}\n\n${B}`],
    ['BKPT;;\n\n# c\nBKPT;;', BL, `${B}\n\n# c\n${B}`],
    ['BKPT;;\n{{ lbl(0) }}:\nBKPT;;\nBKPT;;', BL, `${B}\n\n{{ lbl(0) }}:\n${B}\n\n${B}`], // a macro call as a label
    ['    SET lr0\n        cr0;;', undefined, undefined], // a continuation line keeps its deeper indent
    ['SET lr0\ncr0;;', undefined, '    SET lr0\n    cr0;;'],
    ['SET lr0 cr0;\n        BKPT;;', undefined, `    SET lr0 cr0;\n${B}`],
    [callAfter, BL, callAlone], // a macro call split onto its own line ends its word...
    [callAlone, BL, undefined], // ...so formatting is idempotent
    [tagAfter, BL, `    BKPT;; {% set s = "a\nb" %}\n\n${B}`], // no gap inside a tag that spans lines
  ]) assert.strictEqual(lang.format(text, options), expected === undefined ? text : expected, JSON.stringify([text, options]));
});

test('separators go under each word that another follows, space above the word after, past comments and Jinja', () => {
  for (const [text, ends] of [
    ['BKPT;;\r\nBKPT;;\r\n', [0]],
    ['top:\n    SET lr0 cr0;\n    SET lr1 cr0;;\n    BKPT;; # x\n{{ block(0) }}\n{# a;; #}\n    BKPT;;\n', [2, 3, 4]],
    [spanning, [1, 2, 3]], // line numbers stay right after a {{ }} spanning lines
    ['BKPT;;\n{{ lbl(0) }}:\nBKPT;;\nBKPT;;', [0, 2]],
    [callAfter, [0]], // separators and formatting agree on a macro call after ;; or a label
    ['a: {{ block(0) }}\nBKPT;;', [0]],
  ]) eq(lang.wordEndLines(text), ends, text);
  for (const [text, gaps] of [
    ['top:\n    SET lr0 cr0;\n    SET lr1 cr0;;\n    BKPT;;\n\n    BKPT;;\n    BKPT;;', [3, 6]], // unless a line is empty
    [spanning, [2, 3, 4]],
    ['BKPT;;\n# next\nBKPT;;', [1]], [forLoop, [1, 4]], ['BKPT;;\n\n# c\nBKPT;;', []], [callAlone, []], [tagAfter, [2]],
  ]) eq(lang.wordGapLines(text), gaps, text);
  for (const text of [callAfter, 'a: {{ block(0) }}\nBKPT;;', '    BKPT;;\n{# a\nb #} BKPT;;']) {
    eq(lang.wordGapLines(lang.format(text, BL)), [], `formatted ${text}`);
  }
  for (const [text, line, ends] of [
    ['BKPT;; # x', 0, true], ['{# ;; #}', 0, false], ['SET lr0 cr0;', 0, false],
    [spanning, 0, false], [spanning, 1, true], ['BKPT;;\n{{ block(0) }}\nBKPT;;', 1, true],
  ]) assert.strictEqual(lang.lineEndsWord(text, line), ends, `lineEndsWord ${text} @${line}`);
});

// --- the parser as oracle, over the whole corpus --------------------------------------------------

let [corpusFiles, corpusTokens, corpusContexts, corpusFormats] = [0, 0, 0, 0];
if (corpusPath) test('the corpus: parser, completion, formatting and Jinja agree', () => {
  const problems = [];
  const fail = (message) => problems.push(message);
  for (const entry of JSON.parse(fs.readFileSync(corpusPath, 'utf8'))) {
    corpusFiles++;
    const name = path.basename(entry.file);
    // The rendered text is what the parser saw, so it has no Jinja left.
    const mine = new Map();
    for (const w of lang.scan(entry.text).words) {
      if (w.label) mine.set(w.label.start, 'LABEL');
      for (const ins of w.instructions) {
        mine.set(ins.mnemonic.start, 'MNEMONIC');
        for (const op of ins.operands) mine.set(op.start, 'OPERAND');
      }
      if (!lang.fits(w.instructions.map((i) => i.mnemonic.text))) fail(`${name}: word at ${w.start} assembles, but language.js says it does not fit its slots`);
    }
    const theirs = new Map(entry.tokens.filter((t) => t.role).map((t) => [t.start, t.role]));
    corpusTokens += theirs.size;
    const [start, role] = [...theirs].find(([at, r]) => mine.get(at) !== r) || [];
    if (role) fail(`${name}: offset ${start} ${JSON.stringify(entry.text.slice(start, start + 20))}: parser says ${role}, language.js ${mine.get(start) || 'nothing'}`);
    const myExtra = [...mine].find(([start]) => !theirs.has(start));
    if (myExtra) fail(`${name}: language.js found a ${myExtra[1]} at ${myExtra[0]} the parser did not`);
    // Every Label operand refers to a label the file defines (it assembled).
    const { defs, refs } = lang.labels(entry.text);
    for (const ref of refs) if (!defs.some((d) => lang.sameLabel(d.name, ref.name))) fail(`${name}: reference to ${ref.name} has no definition`);
    // Completion's view of every instruction the file uses: at each mnemonic, that mnemonic is offered;
    // at each operand, the context names the right instruction and position, and its type accepts it.
    const audit = (label, text) => {
      const defs = lang.labels(text).defs.map((d) => d.name);
      const same = (a, b) => a.toLowerCase() === b.toLowerCase();
      for (const w of lang.scan(text).words) {
        for (const ins of w.instructions) {
          if (ins.mnemonic.templated) continue;
          const at = lang.context(text, ins.mnemonic.end);
          corpusContexts++;
          if (!(at.kind === 'mnemonic' && lang.mnemonicsFor(at.others).some((m) => same(m.mnemonic, ins.mnemonic.text)))) {
            fail(`${label}: ${ins.mnemonic.text} at ${ins.mnemonic.start} is not offered where it is written`);
          }
          ins.operands.forEach((op, i) => {
            if (op.templated) return;
            const oc = lang.context(text, op.end);
            corpusContexts++;
            const typed = lang.operandType(ins.mnemonic.text, i);
            const right = oc.kind === 'operand' && oc.index === i && same(oc.mnemonic, ins.mnemonic.text) && typed;
            const accepted = right && (typed.domain.kind === 'label'
              ? defs.includes(op.text) || /^\+\d+$/.test(op.text)
              : lang.accepts(typed.domain, op.text));
            if (!accepted) fail(`${label}: operand ${i} of ${ins.mnemonic.text} ("${op.text}" at ${op.start}) is not offered where it is written`);
          });
        }
      }
    };
    audit(name, entry.text);
    const raw = entry.raw;
    if (raw === undefined) continue;
    audit(`${name} (raw)`, raw);
    // Formatting: every kernel is already in the house layout, so without the blank lines formatting
    // changes nothing but trailing whitespace; it is idempotent; and a copy with its layout stripped
    // formats back to the same.
    const kernel = entry.file.includes('/src/tools/ipu-apps/');
    const untrailed = raw.split('\n').map((l) => l.replace(/[ \t]+$/, '')).join('\n');
    for (const blankLine of [false, true]) {
      const once = lang.format(raw, { blankLine });
      corpusFormats++;
      if (lang.format(once, { blankLine }) !== once) fail(`${name}: formatting twice differs from once (blankLine ${blankLine})`);
      if (lang.format(mangle(lang, raw), { blankLine }) !== once) fail(`${name}: a copy with its layout stripped does not format back (blankLine ${blankLine})`);
      if (kernel && !blankLine && once !== untrailed) fail(`${name}: formatting changes a kernel already in the house layout`);
    }
    // The raw file's Jinja regions, as the agreement test's oracle found them.
    const regions = (list) => list.map((r) => `${r.kind}:${r.start}:${r.end}`).join();
    if (regions(jinjaRegions(raw)) !== regions(entry.jinja)) fail(`${name}: Jinja regions differ from the dump`);
  }
  if (problems.length) throw new Error(problems.join('\n     '));
});

main((cases) => `language.js: ${cases} cases pass${!corpusPath ? '.' : `; agrees with the parser on ${corpusTokens} tokens across ${corpusFiles} files`
  + `, offers what the corpus uses at ${corpusContexts} places, and formats it consistently (${corpusFormats} checks).`}`);
