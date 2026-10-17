'use strict';

// The editor's own answers to what the assembler's analysis (`ipu-as check --json --json-version 2`,
// analysis.js) answers otherwise: whether a word's instructions fit its slots, and how Jinja scopes
// a name. Used only when there is no such answer: an assembler from before version 2 (an older
// checkout), no checker at all (no checkout, Restricted Mode, a broken toolchain), and before the
// first answer for a document arrives. Everything here reads isa.json (E08-21's data-driven rules)
// through language.js; nothing restates the ISA.

const { jinjaRegions, JINJA_NAME, JINJA_STRING, setParts } = require('../language');

/** The comma-separated parts of a bracketed list from `open` (just after its `(`) to its `)` or
 *  `limit`: commas inside nested brackets or strings do not split. */
function topLevelParts(text, open, limit) {
  const parts = [];
  let depth = 0;
  let quote = null;
  let from = open;
  for (let i = open; i < limit; i++) {
    const ch = text[i];
    if (quote) {
      if (ch === '\\') i++;
      else if (ch === quote) quote = null;
    } else if (ch === '"' || ch === "'") quote = ch;
    else if ('([{'.includes(ch)) depth++;
    else if (')]}'.includes(ch)) {
      if (depth === 0) {
        parts.push({ start: from, end: i });
        return parts;
      }
      depth--;
    } else if (ch === ',' && depth === 0) {
      parts.push({ start: from, end: i });
      from = i + 1;
    }
  }
  parts.push({ start: from, end: limit });
  return parts;
}

class Fallback {
  /** `lang`: a Language (language.js) over isa.json. */
  constructor(lang) {
    this.lang = lang;
  }

  // --- Slot fit ---

  /** Whether one word's instructions fit its slots, filled as the assembler fills them (isa.json
   *  slotFill): in source order, a wildcard (NOP) takes the first free slot in `order`, and every
   *  other instruction needs a free slot of its own kind. Unknown names are left to the checker. */
  fits(mnemonics) {
    const free = { ...this.lang.isa.slotCount };
    for (const m of mnemonics) {
      const entry = this.lang.lookup(m);
      if (!entry) continue;
      const slot = this.slotFor(entry, free);
      if (!slot || !(free[slot] > 0)) return false;
      free[slot] -= 1;
    }
    return true;
  }

  /** The slot an instruction takes, given the free slots. */
  slotFor(entry, free) {
    const { lang } = this;
    if (!lang.slotFill) {
      // Version 0: a mnemonic with several forms took the first free slot.
      return (entry.forms.length === 1 && entry.forms[0].slot) || lang.isa.slotOrder.find((s) => free[s] > 0);
    }
    const { order, wildcards, kinds } = lang.slotFill;
    if (wildcards.has(entry.mnemonic.toLowerCase())) return order.find((s) => free[s] > 0);
    return kinds[entry.mnemonic]; // none: the word does not fit, so generator drift shows
  }

  /** Mnemonics that fit at the cursor's place in its word (`others`: from Language.context). */
  mnemonicsFor(others) {
    return [...this.lang.instructions.values()].filter(({ mnemonic }) => this.fits([...others.before, mnemonic, ...others.after]));
  }

  // --- Jinja names ---

  /** Names the template defines: `set` (with its literal value, if it has one), `macro` and `for`
   *  targets, and macro parameters. Each carries the `scope` it lives in (null at top level).
   *
   *  A scope is the body of a `macro` (its parameters and the names set inside it) or a `for` (its
   *  targets, and names set in the loop), from the end of its opening tag: the tag itself (a loop's
   *  iterable, a parameter's default) is read in the scope around it. An unclosed block runs to
   *  the end of the text. */
  jinjaDefs(text) {
    return this.lang.memo('jinjaDefs', text, () => {
      const scopes = this.lang.pairedBlocks(text)
        .filter((b) => b.name === 'macro' || b.name === 'for')
        .map((b) => ({ tag: b.name, tagStart: b.start, start: b.tagEnd, end: b.end === null ? text.length : b.end }));
      // The innermost scope around `at`, other than one that starts there.
      const around = (at, own = null) =>
        scopes.filter((x) => x !== own && x.start <= at && at < x.end).reduce((a, b) => (a && a.start > b.start ? a : b), null);
      const scopeOf = (r) => scopes.find((x) => x.tagStart === r.start) || null;
      const defs = [];
      // Each name in `list`, found at offset `at` of the text.
      const names = (list, at, kind, props) => {
        JINJA_NAME.lastIndex = 0;
        for (let n; (n = JINJA_NAME.exec(list)); ) defs.push({ kind, name: n[0], start: at + n.index, end: at + n.index + n[0].length, ...props });
      };
      for (const r of jinjaRegions(text)) {
        if (r.kind !== 'block') continue;
        const body = text.slice(r.start, r.end);
        const inner = /^\{%-?\s*/.exec(body)[0].length;
        const m = /^(set|macro|for)\s+([\s\S]*?)\s*-?%\}$/.exec(body.slice(inner));
        if (!m) continue;
        const [, tag, rest] = m;
        const restStart = r.start + inner + body.slice(inner).indexOf(rest, tag.length);
        if (tag === 'set') {
          // `{% set x = … %}`, `{% set a, b = … %}`, or the block form `{% set x %}…{% endset %}`.
          const s = setParts(rest);
          if (!s) continue;
          // `from`: in force from the end of its tag, so `{% set n = n + 1 %}` reads the n before it.
          names(s.targets, restStart, 'set', { value: s.value, expression: s.expression, scope: around(r.start), from: r.end });
        } else if (tag === 'macro') {
          const s = /^([A-Za-z_][A-Za-z0-9_]*)\s*(\()?/.exec(rest);
          if (!s) continue;
          const own = scopeOf(r);
          defs.push({ kind: 'macro', name: s[1], start: restStart, end: restStart + s[1].length, block: r, scope: around(r.start, own) });
          // Each parameter's name, not its default value.
          for (const part of s[2] ? topLevelParts(text, restStart + s[0].length, r.end) : []) {
            const name = /^\s*([A-Za-z_][A-Za-z0-9_]*)/.exec(text.slice(part.start, part.end));
            if (!name) continue;
            const at = part.start + name[0].length - name[1].length;
            defs.push({ kind: 'param', name: name[1], start: at, end: at + name[1].length, scope: own });
          }
        } else {
          const s = /^([\s\S]*?)\s+in\s/.exec(rest);
          if (s) names(s[1], restStart, 'for', { scope: scopeOf(r) });
        }
      }
      return defs;
    });
  }

  /** Every use of a name inside a Jinja tag (definitions included), skipping strings, attributes
   *  (`x.name`) and Jinja's own words. */
  jinjaNames(text) {
    return this.lang.memo('jinjaNames', text, () => {
      const out = [];
      for (const r of jinjaRegions(text)) {
        if (r.kind === 'comment') continue;
        const body = text.slice(r.start, r.end).replace(JINJA_STRING, (s) => ' '.repeat(s.length));
        JINJA_NAME.lastIndex = 0;
        for (let m; (m = JINJA_NAME.exec(body)); ) {
          if (this.lang.jinjaKeywords.has(m[0]) || body[m.index - 1] === '.') continue;
          out.push({ name: m[0], start: r.start + m.index, end: r.start + m.index + m[0].length });
        }
      }
      return out;
    });
  }

  /** The definition `name` refers to at `offset`, as Jinja resolves it: from the innermost enclosing
   *  scope out, the last definition of it in that scope before `offset` (a parameter or loop target,
   *  unless a `set` after it replaced it). Null if nothing defines it there yet. On a definition's
   *  own name, that definition. */
  jinjaBinding(text, name, offset, defs = this.jinjaDefs(text)) {
    const own = defs.find((d) => d.name === name && d.start <= offset && offset <= d.end);
    if (own) return own;
    const visible = defs.filter((d) => d.name === name && (!d.scope || (d.scope.start <= offset && offset < d.scope.end)));
    const depth = (d) => (d.scope ? d.scope.start : -1);
    for (const start of [...new Set(visible.map(depth))].sort((a, b) => b - a)) {
      const here = visible.filter((d) => depth(d) === start);
      const before = here.filter((d) => (d.from === undefined ? d.start : d.from) <= offset);
      if (before.length) return before[before.length - 1];
      const bound = here.find((d) => d.kind === 'param' || d.kind === 'for');
      if (bound) return bound;
    }
    return null;
  }

  /** Every name defined at `offset`, each by the definition in force. */
  jinjaVisible(text, offset) {
    const defs = this.jinjaDefs(text);
    return [...new Set(defs.map((d) => d.name))].map((name) => this.jinjaBinding(text, name, offset, defs)).filter(Boolean);
  }

  /** The Jinja name under `offset`, with every place it is defined and used in its scope; null
   *  when there is none, or nothing in the file defines it. */
  jinjaAt(text, offset) {
    const inside = (x) => x.start <= offset && offset <= x.end;
    const jinja = this.jinjaNames(text).find(inside);
    if (!jinja) return null;
    // One variable is a name in one scope: a macro's parameter is not the top-level `set` of the
    // same name, nor another macro's parameter.
    const all = this.jinjaDefs(text);
    // A definition is its own binding (a loop target sits in the loop's header, before the scope
    // it opens).
    const bindingAt = (at) => all.find((d) => d.name === jinja.name && d.start === at)
      || this.jinjaBinding(text, jinja.name, at, all)
      || all.find((d) => d.name === jinja.name && d.start >= at && (!d.scope || d.scope.start <= at));
    const bound = bindingAt(jinja.start);
    if (!bound) return null;
    const scope = bound.scope;
    return {
      kind: 'jinja', name: jinja.name, range: jinja,
      defs: all.filter((d) => d.name === jinja.name && d.scope === scope), binding: bound, scope,
      // A definition's scope is null or a scope object, never undefined.
      refs: this.jinjaNames(text).filter((n) => n.name === jinja.name && bindingAt(n.start)?.scope === scope),
    };
  }

  /** The name under `offset`, as a Jinja name or a label, with every place it is defined and used. */
  symbolAt(text, offset) {
    return this.jinjaAt(text, offset) || this.lang.labelAt(text, offset);
  }

  /** Whether renaming the Jinja symbol `sym` to `name` would meet another name: one defined in the
   *  same scope, or in a scope around or inside it, where one would hide the other. */
  jinjaClash(text, sym, name) {
    const span = (scope) => (scope ? { start: scope.tagStart, end: scope.end } : { start: 0, end: text.length });
    const outer = span(sym.scope);
    const meets = (s) => s.start < outer.end && outer.start < s.end;
    const defined = this.jinjaDefs(text).some((d) => d.name === name && meets(span(d.scope)));
    // A name the file uses without defining it (one the harness passes in, a builtin, `loop`) would
    // be captured just the same.
    return defined || this.jinjaNames(text).some((n) => n.name === name && outer.start <= n.start && n.start < outer.end);
  }

  /** `{{ name }}` tags whose name has a literal value where they stand: { at (the tag's end), value }. */
  jinjaHints(text) {
    return this.lang.memo('jinjaHints', text, () => {
      const defs = this.jinjaDefs(text);
      const out = [];
      for (const r of jinjaRegions(text)) {
        const m = r.kind === 'variable' && /^\{\{-?\s*([A-Za-z_][A-Za-z0-9_]*)\s*-?\}\}$/.exec(text.slice(r.start, r.end));
        const def = m && this.jinjaBinding(text, m[1], r.start, defs);
        if (def && def.kind === 'set' && def.value != null) out.push({ at: r.end, value: def.value });
      }
      return out;
    });
  }

  /** Labels nothing branches to and `set` names nothing reads: [{ kind, name, start, end }]. A name
   *  written anywhere besides its definitions (a comment, a string, a templated operand) counts as
   *  used, so a name is only ever reported when nothing could be using it. */
  unused(text) {
    return this.lang.memo('unused', text, () => {
      const jinja = this.jinjaDefs(text);
      const unusedSets = jinja.filter((d) => d.kind === 'set' && this.lang.mentions(text, d.name, '') <= jinja.filter((x) => x.name === d.name).length)
        .map(({ name, start, end }) => ({ kind: 'set', name, start, end }));
      return [...this.lang.unusedLabels(text), ...unusedSets];
    });
  }

  /** The outline: labels, `set` names and macros, in order. */
  symbols(text) {
    const jinja = this.jinjaDefs(text).filter((d) => d.kind === 'set' || d.kind === 'macro')
      .map((d) => ({ kind: d.kind, name: d.name, start: d.start, end: d.end, detail: d.kind === 'set' ? d.expression : undefined }));
    return [...this.lang.labelSymbols(text), ...jinja].sort((a, b) => a.start - b.start);
  }

  /** These answers for one text, as the providers ask for them (the interface analysis.js's
   *  answers share). */
  semantics(text) {
    return {
      source: 'fallback',
      symbolAt: (offset) => this.symbolAt(text, offset),
      jinjaVisible: (offset) => this.jinjaVisible(text, offset),
      jinjaClash: (sym, name) => this.jinjaClash(text, sym, name),
      jinjaHints: () => this.jinjaHints(text),
      unused: () => this.unused(text),
      symbols: () => this.symbols(text),
      mnemonicsFor: (ctx) => this.mnemonicsFor(ctx.others),
    };
  }
}

/** The Fallback over `lang`, made once per Language. */
const made = new WeakMap();
function fallbackOf(lang) {
  if (!made.has(lang)) made.set(lang, new Fallback(lang));
  return made.get(lang);
}

module.exports = { Fallback, fallbackOf };
