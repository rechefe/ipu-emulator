'use strict';

// What the editor knows about an IPU assembly source, without an editor.
//
// Everything language-specific comes from data/isa.json, generated from the assembler; nothing
// here restates the ISA. The one thing written by hand is how Jinja delimits its tags: Jinja is an
// external language the assembler renders before parsing. Imports nothing from `vscode`, so
// test/language.js runs it under plain node against the parser's own reading of every kernel.

const JINJA_OPEN = /\{#|\{%|\{\{/g;
const JINJA_CLOSE = { '{#': ['comment', '#}'], '{%': ['block', '%}'], '{{': ['variable', '}}'] };
const JINJA_NAME = /[A-Za-z_][A-Za-z0-9_]*/g;
const JINJA_STRING = /"[^"]*"|'[^']*'/g;
/** Jinja block tags that open a region closed by `end<tag>`. */
const JINJA_PAIRED = ['for', 'if', 'macro', 'call', 'filter', 'raw', 'with', 'block'];

/** Every `{# #}`, `{% %}` and `{{ }}` in the raw text, as offsets; an unclosed tag runs to the end.
 *  The Jinja-layer agreement test checks this against Jinja's own lexer. */
function jinjaRegions(text) {
  const regions = [];
  JINJA_OPEN.lastIndex = 0;
  for (let m; (m = JINJA_OPEN.exec(text)); ) {
    const [kind, closer] = JINJA_CLOSE[m[0]];
    const close = text.indexOf(closer, m.index + 2);
    const end = close < 0 ? text.length : close + closer.length;
    regions.push({ kind, start: m.index, end, closed: close >= 0 });
    JINJA_OPEN.lastIndex = end;
  }
  return regions;
}

/** The last of `regions` (sorted, disjoint) that starts before `at`, if it reaches past `after`:
 *  the region holding [after, at), found by bisection. */
function regionOver(regions, after, at = after + 1) {
  let lo = 0;
  let hi = regions.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (regions[mid].start < at) lo = mid + 1;
    else hi = mid;
  }
  const r = regions[lo - 1];
  return r && r.end > after ? r : null;
}

/** How many texts `memo` remembers per analysis: a few editors side by side do not evict each other. */
const MEMO_SLOTS = 3;
const NON_BLANK = /[^ \t\r\n]/g;

/** `span` with every character replaced by `fill`, keeping newlines so offsets and lines hold --
 *  unless `acrossLines`, for a `{{ … }}` that renders to one token however many lines it spans. */
function blank(span, fill, acrossLines = false) {
  return acrossLines || !span.includes('\n') ? fill.repeat(span.length) : span.replace(/[^\n]/g, fill);
}

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

/** A `{{ }}` tag that calls a macro: `{{ name(`. */
const CALL = /^\{\{-?\s*[A-Za-z_][\w.]*\s*\(/;
/** A line that closes a Jinja block, or opens its next branch: it stays with the word before it. */
const CLOSER = /^[ \t]*\{%-?\s*(end\w*|else|elif)\b/;

/** Where each word is set apart from the one before it, given each line's
 *  { ends, code, empty, closer, inside }: for a word with code after it, the first line after it
 *  that is not a closing Jinja tag (so a comment or `{% for %}` goes with the word it introduces).
 *  None where an empty line already separates the two. Never on a line that continues a tag begun
 *  above it: the empty line would land inside the tag (a string's value, say). */
function gapsIn(entries) {
  const out = [];
  let gap = null;
  entries.forEach((e, i) => {
    if (gap) {
      if (e.empty) gap = null;
      else if (e.code) {
        out.push(gap.at === undefined ? i : gap.at);
        gap = null;
      } else if (gap.at === undefined && !e.closer && !e.inside) gap.at = i;
    }
    if (e.ends) gap = {};
  });
  return out;
}

const cloneInstruction = (ins) => ins && { ...ins, operands: [...ins.operands] };
/** A copy of the scan state that later edits leave alone: each branch starts from its own. */
const forkOf = ({ word, instruction }) => ({
  word: word && { ...word, instructions: word.instructions.map(cloneInstruction) },
  instruction: cloneInstruction(instruction),
});
const isBlank = (ch) => ch === ' ' || ch === '\n' || ch === '\t' || ch === '\r' || (ch > '\x7f' && /\s/.test(ch));

/** One run of Language.scan over a text (see there): the scan state, and a step per kind of
 *  thing the scan meets. */
class Scanner {
  constructor(lang, text, stopAt) {
    const { masked, regions, comments, templated, byStart } = lang.mask(text);
    Object.assign(this, { lang, text, stopAt, masked, regions, comments, templated, byStart });
    // Jinja `if` / `elif` / `else` / `endif` tags, by start offset.
    this.tags = new Map(lang.blockTags(text).filter((b) => /^(if|elif|else|endif)$/.test(b.name)).map((b) => [b.start, b]));
    this.tagStarts = [...this.tags.keys()]; // in text order
    this.t = 0;
    this.i = 0;
    this.words = [];
    this.tokens = [];
    this.forks = [];
    this.word = null;
    this.instruction = null;
    this.cursor = null;
  }

  run() {
    while (this.i < this.masked.length) {
      if (this.stopAt !== null && !this.cursor && this.i >= this.stopAt) this.capture(null);
      // Stop once the cursor's word is complete, or its branch was left.
      const { cursor } = this;
      if (cursor && (cursor.done || (cursor.word && this.word !== cursor.word && this.word !== null))) break;
      if (!this.step()) break;
    }
    if (this.stopAt !== null && !this.cursor) this.capture(null);
    if (this.word && !(this.cursor && this.cursor.done)) this.closeWord();
    const { words, tokens, regions, comments, cursor } = this;
    return { words, tokens, regions, comments, cursor };
  }

  /** Read what starts at `i` and move past it; false when the scan is over (the cursor's
   *  word left for a sibling branch). */
  step() {
    const { i, text } = this;
    const tag = this.tags.get(i);
    if (tag) return this.branch(tag);
    if (!this.instruction && this.byStart.has(i) && this.lang.wordsCall(text, i)) {
      this.closeWord();
      this.i = this.byStart.get(i).end;
    } else if (isBlank(this.masked[i])) {
      this.skipBlanks();
    } else if (!this.separator() && !(!this.instruction && this.label())) {
      this.token();
    }
    return true;
  }

  /** An `if` saves the scan state, an `elif` or `else` restarts from it, an `endif` drops it. */
  branch(tag) {
    if (tag.name === 'if') this.forks.push(forkOf(this));
    else if (tag.name === 'endif') this.forks.pop();
    else if (this.forks.length) {
      if (this.cursor && this.cursor.word) return false; // a sibling branch of the cursor's
      Object.assign(this, forkOf(this.forks[this.forks.length - 1]));
    }
    this.i = tag.end;
    return true;
  }

  /** Masked comments and tags are long runs of blanks: jump to the next character, branch tag
   *  or cursor, whichever comes first. */
  skipBlanks() {
    const { i, stopAt, tagStarts } = this;
    NON_BLANK.lastIndex = i;
    const next = NON_BLANK.exec(this.masked);
    let to = next ? next.index : this.masked.length;
    while (tagStarts[this.t] !== undefined && tagStarts[this.t] <= i) this.t++;
    if (tagStarts[this.t] !== undefined && tagStarts[this.t] < to) to = tagStarts[this.t];
    if (stopAt !== null && !this.cursor && i < stopAt && stopAt < to) to = stopAt;
    this.i = Math.max(to, i + 1);
  }

  /** A `;;` or `;` at `i`, if there is one. */
  separator() {
    const { lang, masked, i } = this;
    const ends = masked.startsWith(lang.wordEnd, i);
    if (!ends && !masked.startsWith(lang.slotSep, i)) return false;
    this.openWord(i);
    if (ends) this.closeWord();
    else this.closeInstruction();
    this.i += (ends ? lang.wordEnd : lang.slotSep).length;
    return true;
  }

  /** A label definition at `i`, if there is one. */
  label() {
    const { lang, i } = this;
    lang.labelRe.lastIndex = i;
    const label = lang.labelRe.exec(this.masked);
    if (!label) return false;
    if (this.word && (this.word.label || this.word.instructions.length)) this.closeWord();
    this.openWord(i);
    const end = i + label[1].length;
    const entry = { role: 'label', name: this.text.slice(i, end), start: i, end, templated: this.isTemplated(i, end) };
    if (this.cursorIn(i, end)) this.capture(entry, 'label');
    this.word.label = entry;
    this.tokens.push(entry);
    this.i = lang.labelRe.lastIndex;
    return true;
  }

  /** A mnemonic, or an operand of the instruction it began. */
  token() {
    const { lang, i } = this;
    this.openWord(i);
    lang.tokenRe.lastIndex = i;
    const end = lang.tokenRe.exec(this.masked) ? lang.tokenRe.lastIndex : i + 1;
    const token = { text: this.text.slice(i, end), start: i, end, templated: this.isTemplated(i, end) };
    if (this.cursorIn(i, end)) this.capture(token);
    const { instruction } = this;
    if (!instruction) {
      this.instruction = { mnemonic: token, operands: [] };
      if (this.cursor && this.cursor.token === token) this.cursor.own = this.instruction;
      this.tokens.push({ role: 'mnemonic', ...token });
    } else {
      this.tokens.push({ role: 'operand', mnemonic: instruction.mnemonic.text, index: instruction.operands.length, ...token });
      instruction.operands.push(token);
    }
    this.i = end;
  }

  isTemplated(start, end) {
    return Boolean(regionOver(this.templated, start, end));
  }

  /** Whether the cursor, not yet found, lies in (start, end]. */
  cursorIn(start, end) {
    return this.stopAt !== null && !this.cursor && start < this.stopAt && this.stopAt <= end;
  }

  openWord(at) {
    if (this.word) return;
    this.word = { start: at, label: null, instructions: [] };
    if (this.cursor && this.cursor.word === undefined) this.cursor.word = this.word;
  }

  closeInstruction() {
    const { word, instruction, cursor } = this;
    if (instruction) {
      word.instructions.push(instruction);
      if (cursor && cursor.word === word && instruction !== cursor.own) cursor.after.push(instruction.mnemonic.text);
    }
    this.instruction = null;
  }

  closeWord() {
    if (!this.word) return;
    this.closeInstruction();
    this.words.push(this.word);
    if (this.cursor && this.cursor.word === this.word) this.cursor.done = true;
    this.word = null;
  }

  capture(token, role) {
    const { word, instruction } = this;
    this.cursor = {
      word: word || undefined,
      own: instruction,
      token,
      role: role || (instruction ? 'operand' : 'mnemonic'),
      mnemonic: instruction ? instruction.mnemonic.text : null,
      index: instruction ? instruction.operands.length : 0,
      before: word ? word.instructions.map((ins) => ins.mnemonic.text) : [],
      after: [],
      done: false,
    };
  }
}

/** One run of Language.format over a text (see there). Lines are worked on without their `\r`,
 *  and each is written back with its own line ending (the pieces of a split line share it), so
 *  a CRLF file stays CRLF and a mixed one keeps its mix. */
class Formatter {
  constructor(lang, text, indent) {
    const { masked, regions } = lang.mask(text);
    Object.assign(this, { lang, text, indent, masked, regions });
    this.lines = text.split('\n');
    this.starts = [];
    for (let n = 0, at = 0; n < this.lines.length; at += this.lines[n].length + 1, n++) this.starts.push(at);
    // Each output line, and whether it ends a word or holds code. Lines outside the range are
    // worked out too (`keep` false), since where a word ends before the range decides the empty
    // line at its start.
    this.out = [];
    this.keep = true;
    this.eol = '\n';
    // Whether the last instruction carries on to the next line (no `;` yet): that line is its
    // continuation and keeps its own deeper indent.
    this.open = false;
  }

  run(fromLine, toLine, blankLine) {
    const [first, last] = this.window(fromLine, toLine, blankLine);
    for (let n = first; n <= last; n++) {
      this.keep = n >= fromLine && n <= toLine;
      this.line(n);
    }
    const gaps = new Set(blankLine ? gapsIn(this.out) : []);
    const result = this.out.flatMap((line, k) => (!line.keep ? [] : gaps.has(k) ? [{ text: '', eol: line.eol }, line] : [line]));
    return result.map((line, i) => (i < result.length - 1 ? line.text + line.eol : line.text)).join('');
  }

  /** The lines to work on. Without empty lines to place, a range needs only the lines from the
   *  last code line before it (which says whether an instruction carries on into it) to its
   *  end: formatting as you type stays cheap in a long file. */
  window(fromLine, toLine, blankLine) {
    const { lines, starts, masked } = this;
    if (blankLine) return [0, lines.length - 1];
    let first = Math.min(fromLine, lines.length - 1);
    while (first > 0 && !(masked.slice(starts[first - 1], starts[first] - 1).trim() && !this.inRegion(starts[first - 1]))) first--;
    return [Math.max(0, first - 1), Math.min(toLine, lines.length - 1)];
  }

  inRegion(at) {
    return Boolean(regionOver(this.regions, at, at));
  }

  endsWord(maskedText) {
    return maskedText.trimEnd().endsWith(this.lang.wordEnd);
  }

  emit(text, ends = false, code = false, inside = false) {
    const { keep, eol } = this;
    this.out.push({ text, ends, code, keep, eol, empty: !text.trim(), closer: !code && CLOSER.test(text), inside });
  }

  /** Line `n`, formatted. */
  line(n) {
    const raw = this.lines[n];
    const line = raw.endsWith('\r') ? raw.slice(0, -1) : raw;
    this.eol = line.length < raw.length ? '\r\n' : '\n';
    const start = this.starts[n];
    const end = start + line.length;
    // Inside a tag begun on an earlier line: not ours to touch.
    if (this.inRegion(start)) {
      this.emit(line, this.endsWord(this.masked.slice(start, end)), false, true);
      return;
    }
    // Trailing whitespace, unless a tag carries on past this line.
    const body = this.inRegion(end) || this.inRegion(end - 1) ? line : line.replace(/[ \t]+$/, '');
    const m = this.masked.slice(start, start + body.length);
    const lead = /^[ \t]*/.exec(body)[0].length;
    if (this.lang.wordsCall(this.text, start + lead)) {
      this.emit(body, true, true); // whole words
      this.open = false;
    } else if (!m.trim()) {
      this.emit(body);
    } else {
      for (const piece of this.pieces(body, m, lead)) this.piece(piece, { body, m, lead, start });
    }
  }

  /** A line's code split at each `;;` that has more code after it on the line. */
  pieces(body, m, lead) {
    const { wordEnd } = this.lang;
    const pieces = [];
    let from = lead;
    for (let i = lead; i < m.length; i++) {
      const cut = i + wordEnd.length;
      if (m.startsWith(wordEnd, i) && m.slice(cut).trim()) {
        pieces.push({ at: from, text: body.slice(from, cut) });
        from = cut + /^[ \t]*/.exec(body.slice(cut))[0].length;
        i = from - 1;
      }
    }
    pieces.push({ at: from, text: body.slice(from) });
    return pieces;
  }

  /** One piece of a line: a label on a line of its own, then its code one indent in. */
  piece({ at, text: piece }, { body, m, lead, start }) {
    let pieceMasked = m.slice(at, at + piece.length);
    const label = this.lang.labelStart.exec(pieceMasked);
    if (label) {
      const rest = piece.slice(label[0].length);
      // The label on its own line, with a trailing comment if that is all that follows it;
      // anything else after it is an ordinary piece.
      const alone = !pieceMasked.slice(label[0].length).trim();
      this.emit(piece.slice(0, label[0].length) + (alone ? rest.replace(/[ \t]+$/, '') : ''), false, true);
      if (alone) {
        this.open = false;
        return;
      }
      const skip = label[0].length + /^[ \t]*/.exec(rest)[0].length;
      at += skip;
      piece = piece.slice(skip);
      pieceMasked = pieceMasked.slice(skip);
    }
    const { indent } = this;
    const deeper = this.open && at === lead && lead > indent.length;
    // A macro call that stands for whole words ends its word too, once a split or a label has
    // put it at the start of a line.
    const words = at !== lead && this.lang.wordsCall(this.text, start + at);
    this.emit((deeper ? body.slice(0, lead) : indent) + piece, this.endsWord(pieceMasked) || words, true);
    this.open = !pieceMasked.trimEnd().endsWith(';');
  }
}

class Language {
  constructor(isa) {
    const lexical = isa.lexical;
    this.isa = isa;
    this.tokenRe = new RegExp(lexical.token, 'y');
    this.tokenFull = new RegExp(`^(?:${lexical.token})$`);
    this.labelRe = new RegExp(lexical.label, 'y');
    this.labelStart = new RegExp(`^${lexical.label}`);
    this.commentRe = new RegExp(lexical.comments.map((c) => `(?:${c})`).join('|'), 'g');
    this.memos = new Map();
    /** Register names, lower-cased (assembly is case-insensitive). */
    this.registers = new Set(isa.registers.map((r) => r.toLowerCase()));
    this.slotSep = lexical.slotSeparator;
    this.wordEnd = lexical.wordTerminator;
    this.caseInsensitive = lexical.caseInsensitive;
    this.jinjaKeywords = new Set(lexical.jinjaKeywords);
    // mnemonic (lower case) -> { mnemonic, forms }, as in the hover index.
    this.instructions = new Map(Object.entries(isa.instructions).map(([mnemonic, forms]) => [mnemonic.toLowerCase(), { mnemonic, forms }]));
  }

  // --- Masking: the assembly layer with Jinja and comments taken out ---

  /** The text as the assembly lexer should see it, offsets preserved.
   *
   *  `{# #}` and `{% %}` become blanks. A `{{ }}` becomes one identifier-shaped run, because in
   *  assembly position it renders to a token (`{{ lr_row }}` is an operand); `templated` records
   *  those spans so they are never taken for a literal name. Comments go last, so a `#` inside a
   *  Jinja tag does not start one. */
  mask(text) {
    return this.memo('mask', text, () => {
      const regions = jinjaRegions(text);
      let jinja = '';
      let at = 0;
      for (const r of regions) {
        const variable = r.kind === 'variable';
        jinja += text.slice(at, r.start) + blank(text.slice(r.start, r.end), variable ? 'J' : ' ', variable);
        at = r.end;
      }
      jinja += text.slice(at);
      // One pass for every comment form: the leftmost comment wins, as in the lexer.
      const comments = [];
      let masked = '';
      at = 0;
      this.commentRe.lastIndex = 0;
      for (let m; (m = this.commentRe.exec(jinja)); ) {
        if (!m[0]) this.commentRe.lastIndex++;
        else {
          comments.push({ start: m.index, end: m.index + m[0].length });
          masked += jinja.slice(at, m.index) + blank(m[0], ' ');
          at = m.index + m[0].length;
        }
      }
      masked += jinja.slice(at);
      const byStart = new Map(regions.map((r) => [r.start, r]));
      return { masked, regions, comments, byStart, templated: regions.filter((r) => r.kind === 'variable') };
    });
  }

  /** `compute()`, remembered for the latest `text` per `name`: while typing, completion, spacing,
   *  outline and folding all ask about the same text. Only for results nobody changes. */
  memo(name, text, compute) {
    let slots = this.memos.get(name);
    if (!slots) this.memos.set(name, (slots = []));
    for (const hit of slots) {
      if (hit.text === text) {
        hit.text = text; // keep this very string: the next call compares by identity, not by character
        return hit.value;
      }
    }
    const value = compute();
    slots.unshift({ text, value });
    slots.length = Math.min(slots.length, MEMO_SLOTS);
    return value;
  }

  // --- Structure: words, labels, instructions, operands ---

  /** Every `{% %}` tag with its name (`for`, `endif`, …), in order. */
  blockTags(text) {
    return this.memo('blockTags', text, () => {
      return jinjaRegions(text)
        .filter((r) => r.kind === 'block')
        .map((r) => ({ ...r, name: (/^\{%-?\s*(\w+)/.exec(text.slice(r.start, r.end)) || [])[1] }))
        .filter((t) => t.name);
    });
  }

  /** Scan the assembly layer into words, instructions and tokens.
   *
   *  Mirrors asm_grammar.lark: a word is `label? instr (; instr)*` ending in `;;`, an instruction is
   *  a mnemonic followed by operand tokens. The template is read as the assembler would see one
   *  rendering of it:
   *
   *  - `{% if %}` saves the scan state and every `{% elif %}` / `{% else %}` restarts from it, so
   *    alternatives (`{% if a %}ACC.MAX.FIRST{% else %}ACC.MAX{% endif %}`) are read as
   *    alternatives, not as a mnemonic and its operand. Tokens of every branch are still recorded,
   *    so navigation and rename see all of them.
   *  - A label always starts a word: `:` follows nothing else in the grammar, so one seen mid-word
   *    means the template closed the word elsewhere.
   *  - A macro call where an instruction would start (`{{ block(0) }}`) expands to whole words.
   *
   *  With `stopAt`, also reports the cursor context there (see context()). */
  scan(text, stopAt = null) {
    // Remembered without a cursor only: each cursor position is a scan of its own.
    return stopAt === null ? this.memo('scan', text, () => new Scanner(this, text, null).run()) : new Scanner(this, text, stopAt).run();
  }

  lookup(mnemonic) {
    return this.instructions.get(mnemonic.toLowerCase()) || null;
  }

  /** Whether one word's instructions fit its slots, filled the way CompoundInst._fill_instructions
   *  does: in source order, a named instruction takes a free slot of its own kind and a NOP (whose
   *  forms name no single slot) takes the first free slot in slotOrder. Unknown names are
   *  left to the checker. */
  fits(mnemonics) {
    const free = { ...this.isa.slotCount };
    for (const m of mnemonics) {
      const entry = this.lookup(m);
      if (!entry) continue;
      const slot = (entry.forms.length === 1 && entry.forms[0].slot) || this.isa.slotOrder.find((s) => free[s] > 0);
      if (!slot || !(free[slot] > 0)) return false;
      free[slot] -= 1;
    }
    return true;
  }

  // --- Cursor context, for completion and signature help ---

  /** Where the cursor is: in Jinja, in a comment, on a label, at a mnemonic, or at the n-th operand
   *  of an instruction; plus the other instructions of its word, before and after it, on the same
   *  branch. */
  context(text, offset) {
    const { regions, comments, cursor } = this.scan(text, offset);
    // Inside a tag -- or at the end of one still being typed, which runs to the end of the file.
    const region = regions.find((r) => r.start < offset && (offset < r.end || (!r.closed && offset <= r.end)));
    if (region) return { kind: 'jinja', region };
    if (comments.some((c) => c.start < offset && offset <= c.end)) return { kind: 'comment' };
    const prefix = cursor.token ? text.slice(cursor.token.start, offset) : '';
    const others = { before: cursor.before, after: cursor.after };
    if (cursor.role !== 'operand') return { kind: cursor.role, prefix, others };
    return { kind: 'operand', mnemonic: cursor.mnemonic, index: cursor.index, prefix, others };
  }

  /** Mnemonics that fit at the cursor's place in its word. */
  mnemonicsFor(others) {
    return [...this.instructions.values()].filter(({ mnemonic }) => this.fits([...others.before, mnemonic, ...others.after]));
  }

  operandType(mnemonic, index) {
    const entry = this.lookup(mnemonic);
    // The form an operand list belongs to.
    const form = entry && (entry.forms.find((f) => f.operands.length) || entry.forms[0]);
    if (!form || index >= form.operands.length) return null;
    const operand = form.operands[index];
    return { form, operand, domain: this.isa.operandTypes[operand.type] };
  }

  /** True when a literal value is one the operand type accepts. */
  accepts(domain, value) {
    if (!domain || value == null) return false;
    const v = String(value);
    const values = domain.values || [];
    if (this.caseInsensitive.registers ? values.some((x) => x.toLowerCase() === v.toLowerCase()) : values.includes(v)) return true;
    if ('min' in domain && /^[+-]?(0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\d+)$/.test(v)) {
      const n = Number(v.replace(/_/g, ''));
      return n >= domain.min && n <= domain.max;
    }
    return false;
  }

  // --- Labels ---

  /** Label definitions and references (operands of Label type) with literal names. Templated names
   *  (`{{ name }}_loop`) are left out: their text is not in the file. */
  labels(text) {
    return this.memo('labels', text, () => {
      const defs = [];
      const refs = [];
      for (const t of this.scan(text).tokens) {
        if (t.templated) continue;
        if (t.role === 'label') defs.push({ name: t.name, start: t.start, end: t.end });
        else if (t.role === 'operand' && !/^[+-]/.test(t.text)) {
          const typed = this.operandType(t.mnemonic, t.index);
          if (typed && typed.domain && typed.domain.kind === 'label') refs.push({ name: t.text, start: t.start, end: t.end });
        }
      }
      return { defs, refs };
    });
  }

  sameLabel(a, b) {
    return this.caseInsensitive.labels ? a.toLowerCase() === b.toLowerCase() : a === b;
  }

  // --- Jinja names ---

  /** Every paired `{% tag %} … {% endtag %}` block (and the block form of `{% set x %}`), with the
   *  end of its opening tag and of its closing one (null while it is unclosed). */
  pairedBlocks(text) {
    return this.memo('pairedBlocks', text, () => {
      const out = [];
      const stack = [];
      for (const r of this.blockTags(text)) {
        if (JINJA_PAIRED.includes(r.name) || (r.name === 'set' && !text.slice(r.start, r.end).includes('='))) {
          const block = { name: r.name, start: r.start, tagEnd: r.end, end: null };
          stack.push(block);
          out.push(block);
        } else if (r.name.startsWith('end')) {
          const k = stack.map((b) => b.name).lastIndexOf(r.name.slice(3));
          if (k >= 0) {
            stack[k].end = r.end;
            stack.length = k;
          }
        }
      }
      return out;
    });
  }

  /** Names the template defines: `set` (with its literal value, if it has one), `macro` and `for`
   *  targets, and macro parameters. Each carries the `scope` it lives in (null at top level).
   *
   *  A scope is the body of a `macro` (its parameters and the names set inside it) or a `for` (its
   *  targets, and names set in the loop), from the end of its opening tag: the tag itself (a loop's
   *  iterable, a parameter's default) is read in the scope around it. An unclosed block runs to
   *  the end of the text. */
  jinjaDefs(text) {
    return this.memo('jinjaDefs', text, () => {
      const scopes = this.pairedBlocks(text)
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
          const s = /^([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*(?:=\s*([\s\S]*))?$/.exec(rest);
          if (!s) continue;
          const expression = s[2] === undefined ? null : s[2].trim();
          // A literal value only means something for a single target.
          const literal = !s[1].includes(',') && expression && /^(?:"([^"]*)"|'([^']*)'|([+-]?\d+))$/.exec(expression);
          const value = literal ? (literal[1] ?? literal[2] ?? literal[3]) : null;
          // `from`: in force from the end of its tag, so `{% set n = n + 1 %}` reads the n before it.
          names(s[1], restStart, 'set', { value, expression, scope: around(r.start), from: r.end });
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
    return this.memo('jinjaNames', text, () => {
      const out = [];
      for (const r of jinjaRegions(text)) {
        if (r.kind === 'comment') continue;
        const body = text.slice(r.start, r.end).replace(JINJA_STRING, (s) => ' '.repeat(s.length));
        JINJA_NAME.lastIndex = 0;
        for (let m; (m = JINJA_NAME.exec(body)); ) {
          if (this.jinjaKeywords.has(m[0]) || body[m.index - 1] === '.') continue;
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

  // --- Formatting ---

  /** Whether a tag that starts at offset `at` is a macro call that expands to whole words
   *  (`{{ block(0) }}`). A call followed by `:` names a label instead. */
  wordsCall(text, at) {
    const { masked, byStart } = this.mask(text);
    const region = byStart.get(at);
    return Boolean(region) && region.kind === 'variable' && CALL.test(text.slice(region.start, region.end))
      && !/^[ \t]*:/.test(masked.slice(region.end, region.end + 80));
  }

  /** Whether line `n` (0-based) ends a word: its code, comments and Jinja aside, ends in `;;`. */
  lineEndsWord(text, n) {
    return Boolean(this.lineEntries(text)[n]?.ends);
  }

  /** The lines (0-based) after which one word ends and another follows: where the editor draws a
   *  separator. A line ending in `;;` counts, and so does a macro call that expands to whole words
   *  (`{{ block(0) }}`); the last word in the file gets none. */
  wordEndLines(text) {
    const entries = this.lineEntries(text);
    const last = entries.map((e) => e.code).lastIndexOf(true);
    return entries.flatMap((e, n) => (e.ends && n < last ? [n] : []));
  }

  /** The lines (0-based) where a word is set apart from the one before it: where the editor leaves
   *  space. See `gapsIn`. */
  wordGapLines(text) {
    return gapsIn(this.lineEntries(text));
  }

  /** Each line's part in the word structure, as `gapsIn` reads it. A line that continues a tag
   *  begun above it holds no code of its own. */
  lineEntries(text) {
    return this.memo('lineEntries', text, () => {
      const { masked, regions } = this.mask(text);
      let next = 0;
      return text.split('\n').map((raw) => {
        const start = next;
        next += raw.length + 1;
        const line = raw.endsWith('\r') ? raw.slice(0, -1) : raw;
        // Sliced by offset: a `{{ }}` spanning lines is masked across newlines.
        const m = masked.slice(start, start + line.length);
        const inside = Boolean(regionOver(regions, start, start));
        // The line's last piece: after its last `;;` and a label, as format splits it. A macro call
        // there stands for whole words.
        let tail = m.lastIndexOf(this.wordEnd);
        tail = tail < 0 ? 0 : tail + this.wordEnd.length;
        tail += /^[ \t]*/.exec(line.slice(tail))[0].length;
        const label = this.labelStart.exec(m.slice(tail));
        if (label) tail += label[0].length + /^[ \t]*/.exec(line.slice(tail + label[0].length))[0].length;
        return {
          code: !inside && Boolean(m.trim()),
          ends: m.trimEnd().endsWith(this.wordEnd) || (!inside && this.wordsCall(text, start + tail)),
          empty: !line.trim(),
          closer: !inside && !m.trim() && CLOSER.test(line),
          inside,
        };
      });
    });
  }

  /** The repository's layout, applied to [fromLine, toLine] (all lines by default). Returns the
   *  formatted text of those lines, joined by `\n`.
   *
   *  - A label sits at column 0, on its own line.
   *  - An instruction line is indented one level (`indent`).
   *  - `;;` ends its line: code after it moves to the next line.
   *  - With `blankLine` (off by default: the kernels have none, and the editor draws a separator
   *    instead), an empty line separates each `;;` word from the next. A word written over several
   *    lines (`;` between its instructions) stays together, and a label stays with its word.
   *
   *  Only whitespace between those pieces changes. Lines that are only Jinja or comments keep their
   *  layout, as do lines inside a multi-line Jinja tag and macro calls that expand to whole words
   *  (`{{ block(0) }}`). Trailing whitespace goes, except inside a tag that continues on the next
   *  line. */
  format(text, { indent = '    ', blankLine = false, fromLine = 0, toLine = Infinity } = {}) {
    return new Formatter(this, text, indent).run(fromLine, toLine, blankLine);
  }

  // --- Symbols and folding ---

  /** The name under `offset`, as a label or a Jinja name, with every place it is defined and used. */
  symbolAt(text, offset) {
    const inside = (x) => x.start <= offset && offset <= x.end;
    const jinja = this.jinjaNames(text).find(inside);
    if (jinja) {
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
    const { defs, refs } = this.labels(text);
    const all = [...defs, ...refs];
    const hit = all.find(inside);
    if (!hit) return null;
    const same = (d) => this.sameLabel(d.name, hit.name);
    return { kind: 'label', name: hit.name, range: hit, defs: defs.filter(same), refs: all.filter(same) };
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

  /** `text` with every use of the symbol `sym` renamed to `name`. */
  rename(text, sym, name) {
    let out = text;
    for (const s of [...sym.refs].sort((a, b) => b.start - a.start)) out = out.slice(0, s.start) + name + out.slice(s.end);
    return out;
  }

  /** Whether `name` is lexically usable for a symbol of `kind`. The assembler has the last word
   *  (the extension re-checks a rename with it); this only rejects what cannot even be a token. */
  validName(kind, name) {
    if (kind === 'jinja') return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !this.jinjaKeywords.has(name);
    // `+N` / `-N` is a relative branch offset, never a label.
    return this.tokenFull.test(name) && !/^[+-]/.test(name);
  }

  /** `{{ name }}` tags whose name has a literal value where they stand: { at (the tag's end), value }. */
  jinjaHints(text) {
    return this.memo('jinjaHints', text, () => {
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
    return this.memo('unused', text, () => {
      const uses = (name, defs, flags) => {
        const word = new RegExp(`(?<![A-Za-z0-9_.])${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![A-Za-z0-9_])`, `g${flags}`);
        return (text.match(word) || []).length - defs;
      };
      const labels = this.labels(text).defs;
      const jinja = this.jinjaDefs(text);
      const sameLabel = (d) => labels.filter((l) => this.sameLabel(l.name, d.name)).length;
      return [
        ...labels.filter((d) => uses(d.name, sameLabel(d), this.caseInsensitive.labels ? 'i' : '') <= 0).map((d) => ({ kind: 'label', ...d })),
        ...jinja.filter((d) => d.kind === 'set' && uses(d.name, jinja.filter((x) => x.name === d.name).length, '') <= 0)
          .map(({ name, start, end }) => ({ kind: 'set', name, start, end })),
      ];
    });
  }

  symbols(text) {
    const labels = this.labels(text).defs.map((d) => ({ kind: 'label', name: d.name, start: d.start, end: d.end }));
    const jinja = this.jinjaDefs(text).filter((d) => d.kind === 'set' || d.kind === 'macro')
      .map((d) => ({ kind: d.kind, name: d.name, start: d.start, end: d.end, detail: d.kind === 'set' ? d.expression : undefined }));
    return [...labels, ...jinja].sort((a, b) => a.start - b.start);
  }

  /** Foldable spans: multi-line `{# #}` comments and paired `{% tag %} … {% endtag %}` blocks. */
  folding(text) {
    const comments = jinjaRegions(text).filter((r) => r.kind === 'comment' && text.slice(r.start, r.end).includes('\n'));
    return [
      ...comments.map((r) => ({ start: r.start, end: r.end, kind: 'comment' })),
      ...this.pairedBlocks(text).filter((b) => b.end !== null).map((b) => ({ start: b.start, end: b.end, kind: 'region' })),
    ];
  }
}

module.exports = { Language, jinjaRegions };
