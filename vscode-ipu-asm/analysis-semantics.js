'use strict';

// What the providers ask, answered from a version-2 analysis (analysis.js): the same interface
// as fallback/'s `semantics(text)`, so a provider does not know which answered it.
//
// The analysis says which names each Jinja tag defines and uses, as Jinja's own parser reads
// them; it does not say where in the tag they are, nor how Jinja scopes them. So:
//
// - A name's places are its whole-word occurrences inside the tags that define or use it
//   (strings and attributes `x.name` aside). A keyword argument `f(name=…)` names a parameter:
//   it is a place of `name` when the file defines a macro parameter of that name, else none.
// - A name is one symbol across the file, whatever scope Jinja gives it: renaming it renames
//   every use, which leaves the program the same, and a clash is any other use of the new name.
// - The definition in force (for hover, inlay values and completion) is the last one before the
//   cursor, a definition in a `macro` or `for` body (its parameters, its targets) counting only
//   inside it: the tags' nesting, by their keywords.
// - A name only the `ipu:params` header declares is the harness's, not the file's: no rename.
//
// The rest is read from the text: a tag's keyword (`set`, `for`, `macro`), a `set`'s expression
// and literal value. Labels come from Language (scan()), as on the fallback path.
//
// Imports nothing from `vscode`.

const JINJA_NAME = /[A-Za-z_][A-Za-z0-9_]*/g;
const JINJA_STRING = /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/g;
const LITERAL = /^(?:"([^"]*)"|'([^']*)'|([+-]?\d+))$/;
const KINDS = { set: 'set', for: 'for', import: 'import', from: 'import' };

/** Offsets of one text mapped to another, assuming one contiguous edit between them: what an
 *  answer for a version or two ago is read through while the next one is on its way. */
class Shift {
  constructor(from, to) {
    let p = 0;
    const max = Math.min(from.length, to.length);
    while (p < max && from.charCodeAt(p) === to.charCodeAt(p)) p++;
    let s = 0;
    while (s < max - p && from.charCodeAt(from.length - 1 - s) === to.charCodeAt(to.length - 1 - s)) s++;
    Object.assign(this, { p, fromTail: from.length - s, toTail: to.length - s, delta: to.length - from.length });
  }
  /** An offset of the new text, in the old one (the edited stretch maps to its start). */
  toOld(at) {
    return at <= this.p ? at : at >= this.toTail ? at - this.delta : this.p;
  }
  /** An offset of the old text, in the new one. */
  toNew(at) {
    return at <= this.p ? at : at >= this.fromTail ? at + this.delta : this.p;
  }
}
const SAME = { toOld: (at) => at, toNew: (at) => at };

/** The Jinja names of an answer: each occurrence, with its tag and whether it defines the name. */
function occurrencesOf(text, regions) {
  const occurrences = [];
  const external = new Set();
  const keywords = [];
  regions.forEach((r, index) => {
    if (r.kind === 'comment') {
      r.defines.forEach((n) => external.add(n)); // the ipu:params header
      return;
    }
    const names = new Set([...r.defines, ...r.uses]);
    const body = text.slice(r.start, r.end).replace(JINJA_STRING, (s) => ' '.repeat(s.length));
    const keyword = r.kind === 'block' ? (/^\{%[-+]?\s*(\w+)/.exec(body) || [])[1] : null;
    const defined = new Set();
    JINJA_NAME.lastIndex = 0;
    for (let m; (m = JINJA_NAME.exec(body)); ) {
      const name = m[0];
      if (body[m.index - 1] === '.' || (m.index > 0 && /\w/.test(body[m.index - 1]))) continue;
      const at = r.start + m.index;
      // `f(name=…)`: a keyword argument, naming the callee's parameter (in a call's brackets; a
      // macro's own brackets hold its parameters).
      const before = body.slice(0, m.index);
      const depth = (before.match(/\(/g) || []).length - (before.match(/\)/g) || []).length;
      if (depth > (keyword === 'macro' ? 1 : 0) && /[(,]\s*$/.test(before) && /^\s*=(?!=)/.test(body.slice(m.index + name.length))) {
        keywords.push({ name, start: at, end: at + name.length, region: index, def: false, kind: null, keyword: true });
        continue;
      }
      if (!names.has(name)) continue;
      const def = r.defines.includes(name) && !defined.has(name);
      if (def) defined.add(name);
      let kind = null;
      if (def) {
        if (keyword === 'macro') kind = occurrences.some((o) => o.region === index && o.def) ? 'param' : 'macro';
        else kind = KINDS[keyword] || 'set';
      }
      occurrences.push({ name, start: at, end: at + name.length, region: index, def, kind });
    }
  });
  // A keyword argument goes with the file's macro parameter of its name; one for a macro defined
  // elsewhere (imported) names that macro's parameter, which no rename here may change.
  const params = new Set(occurrences.filter((o) => o.kind === 'param').map((o) => o.name));
  occurrences.push(...keywords.filter((k) => params.has(k.name)));
  occurrences.sort((a, b) => a.start - b.start);
  return { occurrences, external };
}

/** The `macro` and `for` bodies of an answer's tags: { open (the opening tag's index), start (its
 *  end), end (the closing tag's start; the text's end while unclosed) }. */
function blocksOf(text, regions) {
  const blocks = [];
  const stack = [];
  regions.forEach((r, index) => {
    const keyword = r.kind === 'block' && (/^\{%[-+]?\s*(\w+)/.exec(text.slice(r.start, r.end)) || [])[1];
    if (keyword === 'macro' || keyword === 'for') {
      const block = { keyword, open: index, start: r.end, end: text.length };
      blocks.push(block);
      stack.push(block);
    } else if (keyword === 'endmacro' || keyword === 'endfor') {
      const k = stack.map((b) => b.keyword).lastIndexOf(keyword.slice(3));
      if (k >= 0) {
        stack[k].end = r.start;
        stack.length = k;
      }
    }
  });
  return blocks;
}

/** A `set` tag's expression and literal value for `name`, as its text has them. */
function setValue(text, region, name) {
  const inner = /^\{%[-+]?\s*set\s+([\s\S]*?)\s*[-+]?%\}$/.exec(text.slice(region.start, region.end));
  const s = inner && /^([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*(?:=\s*([\s\S]*))?$/.exec(inner[1]);
  if (!s || !s[1].split(/\s*,\s*/).includes(name)) return { expression: null, value: null };
  const expression = s[2] === undefined ? null : s[2].trim();
  const literal = !s[1].includes(',') && expression && LITERAL.exec(expression);
  return { expression, value: literal ? (literal[1] ?? literal[2] ?? literal[3]) : null };
}

class V2Semantics {
  /** `answer` (analysis.js) for `answer.text`; `text` the document's text now, which may be a
   *  little newer (offsets in are mapped; spans out are the answer's, so only completion, which
   *  returns none, should ask a stale one). */
  constructor(lang, answer, text) {
    this.source = 'analysis';
    this.lang = lang;
    this.answer = answer;
    this.text = text;
    this.shift = answer.text === text ? SAME : new Shift(answer.text, text);
    if (!answer.model) {
      const blocks = blocksOf(answer.text, answer.jinja);
      answer.model = { ...occurrencesOf(answer.text, answer.jinja), blocks };
      // Each definition's body: the innermost one around it, or the one its tag opens.
      for (const o of answer.model.occurrences) {
        if (!o.def) continue;
        const opened = blocks.find((b) => b.open === o.region);
        o.block = opened && o.kind !== 'macro' ? opened
          : blocks.filter((b) => b.start <= o.start && o.start < b.end).reduce((a, b) => (a && a.start > b.start ? a : b), null);
      }
    }
    this.model = answer.model;
  }

  /** The definition in force at `offset` (answer offsets): the last one before it, a `set` from
   *  the end of its tag, of those whose body holds `offset`; else a parameter or loop target of
   *  a body holding it. On a definition, that one. */
  bindingAt(name, offset, own = null) {
    if (own && own.def) return this.describe(own);
    const regions = this.answer.jinja;
    const defs = this.model.occurrences.filter((o) => o.def && o.name === name && (!o.block || (o.block.start <= offset && offset < o.block.end)));
    const before = defs.filter((d) => (d.kind === 'set' ? regions[d.region].end : d.start) <= offset);
    const found = before.length ? before[before.length - 1] : defs.find((d) => d.kind === 'param' || d.kind === 'for');
    return found ? this.describe(found) : null;
  }

  describe(o) {
    const region = this.answer.jinja[o.region];
    const { expression, value } = o.kind === 'set' ? setValue(this.answer.text, region, o.name) : { expression: null, value: null };
    return { kind: o.kind, name: o.name, start: o.start, end: o.end, expression, value };
  }

  symbolAt(at) {
    const offset = this.shift.toOld(at);
    const hit = this.model.occurrences.find((o) => o.start <= offset && offset <= o.end);
    if (!hit) return this.lang.labelAt(this.text, at);
    const { name } = hit;
    const all = this.model.occurrences.filter((o) => o.name === name);
    const defs = all.filter((o) => o.def);
    if (!defs.length || this.model.external.has(name)) return null;
    const span = ({ start, end }) => ({ name, start, end });
    return {
      kind: 'jinja', name, range: span(hit), scope: null,
      binding: this.bindingAt(name, hit.start, hit) || this.describe(defs[0]),
      defs: defs.map(span), refs: all.map(span),
    };
  }

  /** Each name defined at `at`, by the definition in force there; the `ipu:params` names too. */
  jinjaVisible(at) {
    const offset = this.shift.toOld(at);
    const names = new Set(this.model.occurrences.filter((o) => o.def).map((o) => o.name));
    const out = [...names].map((n) => this.bindingAt(n, offset)).filter(Boolean)
      .map((d) => ({ ...d, start: this.shift.toNew(d.start), end: this.shift.toNew(d.end) }));
    const header = this.answer.jinja.find((r) => r.kind === 'comment' && r.defines.length);
    for (const name of this.model.external) {
      if (!names.has(name)) out.push({ kind: 'header', name, start: this.shift.toNew(header.start), end: this.shift.toNew(header.end), expression: null, value: null });
    }
    return out;
  }

  /** Whether `name` is defined or used anywhere in the file's Jinja. */
  jinjaClash(_sym, name) {
    return this.answer.jinja.some((r) => r.defines.includes(name) || r.uses.includes(name));
  }

  /** `{{ name }}` tags whose name has a literal value where they stand. */
  jinjaHints() {
    const { text } = this.answer;
    const out = [];
    for (const r of this.answer.jinja) {
      const m = r.kind === 'variable' && /^\{\{[-+]?\s*([A-Za-z_][A-Za-z0-9_]*)\s*[-+]?\}\}$/.exec(text.slice(r.start, r.end));
      const def = m && this.bindingAt(m[1], r.start);
      if (def && def.kind === 'set' && def.value != null) out.push({ at: r.end, value: def.value });
    }
    return out;
  }

  /** Labels nothing branches to, and `set` names no tag uses (and the text names nowhere else). */
  unused() {
    const { text } = this.answer;
    const used = new Set(this.answer.jinja.flatMap((r) => r.uses));
    const defs = this.model.occurrences.filter((o) => o.def);
    const sets = defs.filter((d) => d.kind === 'set' && !used.has(d.name)
      && this.lang.mentions(text, d.name, '') <= defs.filter((x) => x.name === d.name).length);
    return [...this.lang.unusedLabels(text), ...sets.map(({ name, start, end }) => ({ kind: 'set', name, start, end }))];
  }

  symbols() {
    const jinja = this.model.occurrences.filter((o) => o.def && (o.kind === 'set' || o.kind === 'macro')).map((o) => {
      const d = this.describe(o);
      return { kind: d.kind, name: d.name, start: d.start, end: d.end, detail: d.kind === 'set' ? d.expression : undefined };
    });
    return [...this.lang.labelSymbols(this.answer.text), ...jinja].sort((a, b) => a.start - b.start);
  }

  /** The word holding offset `at` of the current text, as the assembler fitted it. */
  wordAt(at) {
    if (at === null || at === undefined) return null;
    const offset = this.shift.toOld(at);
    return this.answer.words.find((w) => w.start <= offset && offset <= w.end) || null;
  }

  /** Mnemonics that fit at the cursor (`ctx`: Language.context): the slots the assembler gave the
   *  word's other instructions (pseudo-instructions expanded), less what isa.json says each slot
   *  holds. A word the assembler could not fit (an unknown mnemonic being typed) has no slots;
   *  its known mnemonics are counted by their slot kind instead. */
  mnemonicsFor(ctx, offset) {
    const { lang } = this;
    const all = [...lang.instructions.values()];
    if (!lang.slotFill) return all;
    const { kinds, wildcards } = lang.slotFill;
    // By where the word starts, or by the cursor: a word that starts with a `{{ }}` starts, for
    // the assembler, at its first token in the source.
    const word = this.wordAt(ctx.wordStart) || this.wordAt(offset);
    const known = (m) => lang.lookup(m);
    const kindOf = (m) => (known(m) ? kinds[known(m).mnemonic] : undefined);
    const wild = (m) => wildcards.has(m.toLowerCase());
    const mnemonics = word ? [...word.mnemonics] : [];
    // The mnemonic under the cursor is being replaced: it gives its slot back.
    const typed = ctx.kind === 'mnemonic' && ctx.token ? ctx.token.text : null;
    const replaced = typed && mnemonics.findIndex((m) => m.toLowerCase() === typed.toLowerCase());
    const counted = word && (word.fits || Object.keys(word.slots).length);
    const used = {};
    if (counted) Object.assign(used, word.slots);
    else mnemonics.forEach((m, i) => { if (i !== replaced && kindOf(m) && !wild(m)) used[kindOf(m)] = (used[kindOf(m)] || 0) + 1; });
    const free = Object.fromEntries(Object.entries(lang.isa.slotCount).map(([s, n]) => [s, n - (used[s] || 0)]));
    let total = Object.values(free).reduce((a, n) => a + Math.max(n, 0), 0);
    if (counted && replaced !== null && replaced >= 0) {
      const m = mnemonics[replaced];
      if (wild(m)) total++;
      else if (kindOf(m)) { free[kindOf(m)]++; total++; }
    }
    // A wildcard the assembler put in a slot could move to another free one.
    const movable = Boolean(word && word.fits) && mnemonics.some((m, i) => wild(m) && i !== replaced) && total > 0;
    return all.filter(({ mnemonic }) => {
      if (wild(mnemonic)) return total > 0;
      const kind = kinds[mnemonic];
      return Boolean(kind) && (free[kind] > 0 || movable);
    });
  }
}

module.exports = { V2Semantics, Shift, occurrencesOf };
