'use strict';

// The checkout's assembler as the editor's analysis: one `ipu-as check` per document version
// answers every question the providers ask of it (schema/check-json.md, version 2):
//
//   ipu-as check --json --json-version 2 --tokens --words --jinja --input -
//
// diagnostics, semantic tokens, each word's slot fit and each Jinja tag's names. The extension
// is published apart from the repository, so it meets assemblers of every age: each checker
// command is asked once which versions it speaks (`check --json-versions`; one from before
// versions exits 2 on it), and one that speaks only version 1 is run as before, the providers
// falling back to the editor's own rules (fallback/).
//
// The contract counts columns (and Jinja offsets) in code points, as Python indexes a string;
// VS Code counts UTF-16 code units. Positions converts. The two differ only past the Basic
// Multilingual Plane, so a text with no surrogate pair is converted by doing nothing.
//
// Imports nothing from `vscode`: test/analysis.js runs it in plain node.

const checker = require('./checker');

/** The contract version this extension reads; it also reads version 1 (check-json.md, Rules). */
const CONTRACT_VERSION = 2;
/** What a version-2 check asks for besides diagnostics. */
const ANALYSES = ['--tokens', '--words', '--jinja'];

/** The commands a checker argv (ending `check … --json --input`; the extension appends `-`) is run
 *  as: `v1` as it is, `probe` asking its versions, `v2` asking for every analysis. A command of
 *  another shape (a user's wrapper, a stand-in) has no probe and no `v2`: it is run as written. */
function commandsFor(argv) {
  const at = argv.indexOf('check');
  const shaped = at >= 0 && argv[argv.length - 1] === '--input' && argv.includes('--json')
    && !argv.some((a) => a.startsWith('--json-version'));
  if (!shaped) return { v1: argv, probe: null, v2: null };
  return {
    v1: argv,
    probe: [...argv.slice(0, at + 1), '--json-versions'],
    v2: [...argv.slice(0, -1), '--json-version', String(CONTRACT_VERSION), ...ANALYSES, '--input'],
  };
}

/** The versions a probe's result says the checker speaks: [1] unless it printed a list. */
function versionsFrom(result) {
  if (result.code !== 0) return [1]; // exit 2: an assembler from before versions
  try {
    const versions = JSON.parse(result.stdout);
    return Array.isArray(versions) && versions.every(Number.isInteger) ? versions : [1];
  } catch {
    return [1];
  }
}

const isHigh = (c) => c >= 0xd800 && c <= 0xdbff;
const isLow = (c) => c >= 0xdc00 && c <= 0xdfff;

/** A text's lines, and the contract's code-point positions in it as UTF-16 ones. */
class Positions {
  constructor(text) {
    this.text = text;
    /** Whether any character is outside the BMP; if none, code points are code units. */
    this.wide = /[\uD800-\uDBFF][\uDC00-\uDFFF]/.test(text);
    this.starts = [0];
    for (let i = text.indexOf('\n'); i >= 0; i = text.indexOf('\n', i + 1)) this.starts.push(i + 1);
    this.units = null;
  }

  /** The UTF-16 column of code-point column `cp` on line `line`. A column past the line's end
   *  (an approximate position) stays as far past it. */
  column(line, cp) {
    if (!this.wide || line >= this.starts.length) return cp;
    const { text } = this;
    const end = line + 1 < this.starts.length ? this.starts[line + 1] - 1 : text.length;
    let u = this.starts[line];
    let c = 0;
    for (; c < cp && u < end; c++) u += isHigh(text.charCodeAt(u)) && isLow(text.charCodeAt(u + 1)) ? 2 : 1;
    return u - this.starts[line] + (cp - c);
  }

  /** The UTF-16 offset of code-point offset `cp`. */
  offset(cp) {
    if (!this.wide) return cp;
    if (!this.units) {
      const units = [];
      for (let u = 0; u < this.text.length; u++) {
        units.push(u);
        if (isHigh(this.text.charCodeAt(u)) && isLow(this.text.charCodeAt(u + 1))) u++;
      }
      units.push(this.text.length);
      this.units = units;
    }
    return cp < this.units.length ? this.units[cp] : this.text.length + (cp - this.units.length + 1);
  }

  /** The UTF-16 offset of a contract position. */
  offsetAt(line, cp) {
    if (line >= this.starts.length) return this.text.length;
    return this.starts[line] + this.column(line, cp);
  }

  /** A diagnostic of either version, its columns in UTF-16. */
  diagnostic(raw) {
    if (!this.wide) return raw;
    return { ...raw, column: this.column(raw.line, raw.column), end_column: this.column(raw.end_line, raw.end_column) };
  }
}

/** One version-2 answer for one text, in UTF-16 positions. */
class Answer {
  constructor(text, json) {
    this.text = text;
    this.json = json;
    this.positions = new Positions(text);
  }

  /** As the contract has them: what rename compares, messages and all. */
  get found() {
    return this.json.diagnostics;
  }

  /** Semantic tokens: { line, character, length, role, type }, sorted, UTF-16. */
  get tokens() {
    if (!this.tokensCache) {
      const p = this.positions;
      this.tokensCache = (this.json.tokens || []).map((t) => {
        const character = p.column(t.line, t.column);
        return { line: t.line, character, length: p.column(t.line, t.column + t.length) - character, role: t.role, type: t.type };
      });
    }
    return this.tokensCache;
  }

  /** Each word's slot fit: { start, end (offsets), mnemonics, slots, fits, error }. */
  get words() {
    if (!this.wordsCache) {
      const p = this.positions;
      this.wordsCache = (this.json.words || []).map((w) => ({
        start: p.offsetAt(w.range.line, w.range.column),
        end: p.offsetAt(w.range.end_line, w.range.end_column),
        mnemonics: w.mnemonics, slots: w.slots, fits: w.fits, error: w.error,
      }));
    }
    return this.wordsCache;
  }

  /** Each Jinja tag: { kind, start, end (offsets), defines, uses }. */
  get jinja() {
    if (!this.jinjaCache) {
      const p = this.positions;
      this.jinjaCache = (this.json.jinja || [])
        .map((r) => ({ kind: r.kind, start: p.offset(r.start), end: p.offset(r.end), defines: r.defines, uses: r.uses }));
    }
    return this.jinjaCache;
  }
}

/** What a check printed, as { found (the contract's diagnostics), answer (version 2, else null) }.
 *  Throws when it is not JSON: the caller reports the checker's failure. */
function parseCheck(stdout, text) {
  const out = JSON.parse(stdout);
  if (Array.isArray(out)) return { found: out, answer: null };
  if (out && out.schemaVersion === CONTRACT_VERSION && Array.isArray(out.diagnostics)) {
    return { found: out.diagnostics, answer: new Answer(text, out) };
  }
  throw new Error(`unrecognised check output (schemaVersion ${out && out.schemaVersion})`);
}

/** Asks each checker its versions once, and keeps each document's latest check: every provider
 *  reads it, so a document version is checked once whatever asks about it. */
class AnalysisClient {
  /** `run`: checker.run's signature. `waitMs`: the longest a provider waits for a check. */
  constructor({ run = checker.run, waitMs = 5000 } = {}) {
    this.run = run;
    this.waitMs = waitMs;
    /** Probe command (JSON) -> promise of the versions it speaks. */
    this.probes = new Map();
    /** Document key -> { version, text, found, answer } of its latest successful check. */
    this.entries = new Map();
    /** Document key -> its latest version 2 answer, of whatever version: completion reads it
     *  while the next one is on its way. */
    this.answers = new Map();
    /** Document key -> the newest version whose check is over (succeeded or not). */
    this.released = new Map();
    /** Document key -> [{ version, text, resolve }] waiting for that version's check. */
    this.waiters = new Map();
    this.listeners = new Set();
  }

  /** The versions checker `argv` speaks, asked once per command. A probe that could not run or
   *  ran out of time is asked again next time; the check itself reports why. */
  versions(argv, cwd, seconds) {
    const { probe } = commandsFor(argv);
    if (!probe) return Promise.resolve([1]);
    const key = JSON.stringify(probe);
    if (!this.probes.has(key)) {
      this.probes.set(key, this.run(probe[0], probe.slice(1), cwd, seconds).then((r) => {
        if (r.spawnError || r.timedOut) this.probes.delete(key);
        return versionsFrom(r);
      }));
    }
    return this.probes.get(key);
  }

  /** Forget every probe's answer: the checker may have been rebuilt from other sources. */
  forgetVersions() {
    this.probes.clear();
  }

  /** The argv to check a document with: version 2's when the checker speaks it, else `argv`. */
  async plan(argv, cwd, seconds) {
    const { v2 } = commandsFor(argv);
    if (v2 && (await this.versions(argv, cwd, seconds)).includes(CONTRACT_VERSION)) return { argv: v2, version: CONTRACT_VERSION };
    return { argv, version: 1 };
  }

  /** Keep a document version's check; `answer` is null from a version-1 checker. */
  store(key, version, text, found, answer) {
    this.entries.set(key, { version, text, found, answer });
    if (answer) this.answers.set(key, answer);
    else this.answers.delete(key);
    for (const listener of this.listeners) listener(key, answer);
  }

  /** Call `listener(key, answer)` on each stored check; returns a disposable. */
  onStore(listener) {
    this.listeners.add(listener);
    return { dispose: () => this.listeners.delete(listener) };
  }

  /** The kept check of `text` at `key`, if its latest check was of that very text. */
  cached(key, text) {
    const entry = this.entries.get(key);
    return entry && entry.text === text ? entry : null;
  }

  /** The latest version 2 answer at `key`, possibly for an earlier version of the text. */
  latest(key) {
    return this.answers.get(key) || null;
  }

  /** The version 2 answer for `text`, the document's version `version`: at once if it is kept,
   *  else once that version's check is over. Null when there is none: a version-1 checker, a
   *  failed or superseded check, no checker, or no answer within `waitMs`. With `first` false,
   *  null at once unless the document's last check had an answer: a document not checked yet
   *  (the first paint), or checked by a version-1 checker, has none to wait for. */
  current(key, version, text, { first = true } = {}) {
    const entry = this.entries.get(key);
    if (entry && entry.version === version) return Promise.resolve(entry.answer && entry.answer.text === text ? entry.answer : null);
    if ((this.released.get(key) ?? -Infinity) >= version) return Promise.resolve(null);
    if (!first && !(entry && entry.answer)) return Promise.resolve(null);
    return new Promise((resolve) => {
      const waiter = { version, text, resolve };
      const list = this.waiters.get(key) || [];
      list.push(waiter);
      this.waiters.set(key, list);
      waiter.timer = setTimeout(() => this.settle(key, (w) => w === waiter), this.waitMs);
    });
  }

  /** A check of version `version` at `key` is over: whoever waits for it, or an older one, has
   *  its answer. */
  release(key, version) {
    if (version === undefined) return;
    this.released.set(key, Math.max(this.released.get(key) ?? -Infinity, version));
    this.settle(key, (w) => w.version <= version);
  }

  settle(key, which) {
    const list = this.waiters.get(key) || [];
    const entry = this.entries.get(key);
    for (const w of list.filter(which)) {
      clearTimeout(w.timer);
      w.resolve(entry && entry.version === w.version && entry.answer && entry.answer.text === w.text ? entry.answer : null);
    }
    const left = list.filter((w) => !which(w));
    if (left.length) this.waiters.set(key, left);
    else this.waiters.delete(key);
  }

  /** Forget a closed document. */
  forget(key) {
    this.settle(key, () => true);
    for (const map of [this.entries, this.answers, this.released]) map.delete(key);
  }
}

module.exports = { Answer, AnalysisClient, ANALYSES, CONTRACT_VERSION, Positions, commandsFor, parseCheck, versionsFrom };
