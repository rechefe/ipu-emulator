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
      this.tokensCache = this.json.tokens.map((t) => {
        const character = p.column(t.line, t.column);
        return { line: t.line, character, length: p.column(t.line, t.column + t.length) - character, role: t.role, type: t.type };
      });
    }
    return this.tokensCache;
  }

  /** Each word's slot fit: { start, end (offsets), range (UTF-16 lines and characters),
   *  mnemonics, slots, fits, error }. */
  get words() {
    if (!this.wordsCache) {
      const p = this.positions;
      this.wordsCache = this.json.words.map((w) => ({
        start: p.offsetAt(w.range.line, w.range.column),
        end: p.offsetAt(w.range.end_line, w.range.end_column),
        range: { line: w.range.line, character: p.column(w.range.line, w.range.column), endLine: w.range.end_line, endCharacter: p.column(w.range.end_line, w.range.end_column) },
        mnemonics: w.mnemonics, slots: w.slots, fits: w.fits, error: w.error,
      }));
    }
    return this.wordsCache;
  }

  /** The words that do not fit their slots and that no diagnostic touches. */
  misfits() {
    const p = this.positions;
    const spans = this.found.map((d) => [p.offsetAt(d.line, d.column), p.offsetAt(d.end_line, d.end_column)]);
    return this.words.filter((w) => !w.fits && !spans.some(([from, to]) => from <= w.end && to >= w.start));
  }

  /** Each Jinja tag: { kind, start, end (offsets), defines, uses }. Placed by its `range`, not
   *  its `start` and `end` offsets: those index the text as the checker read it, which on a
   *  platform that translates line ends on stdin (Windows) has no `\r` -- lines and columns are
   *  the same either way. */
  get jinja() {
    if (!this.jinjaCache) {
      const p = this.positions;
      this.jinjaCache = this.json.jinja.map((r) => ({
        kind: r.kind,
        start: p.offsetAt(r.range.line, r.range.column),
        end: p.offsetAt(r.range.end_line, r.range.end_column),
        defines: r.defines, uses: r.uses,
      }));
    }
    return this.jinjaCache;
  }
}

/** The keys a version-2 check asks for besides diagnostics (ANALYSES, as JSON keys). */
const ANALYSIS_KEYS = ANALYSES.map((a) => a.slice(2));

/** What a check printed, as { found (the contract's diagnostics), answer (version 2, else null) }.
 *  An answer missing an analysis it was asked for (the analysis failed: an `internal`
 *  diagnostic says why) is no answer: the providers fall back, the diagnostics still show.
 *  Throws when it is not JSON: the caller reports the checker's failure. */
function parseCheck(stdout, text) {
  const out = JSON.parse(stdout);
  if (Array.isArray(out)) return { found: out, answer: null };
  if (out && out.schemaVersion === CONTRACT_VERSION && Array.isArray(out.diagnostics)) {
    const whole = ANALYSIS_KEYS.every((k) => Array.isArray(out[k]));
    return { found: out.diagnostics, answer: whole ? new Answer(text, out) : null };
  }
  throw new Error(`unrecognised check output (schemaVersion ${out && out.schemaVersion})`);
}

/** What a checker says about itself (its versions, whether it formats), asked once per command
 *  and checkout. A probe that could not run is asked again next time; one that ran out of time
 *  reads as `interpret` reads a failed run for `retryMs`, so callers do not each wait it out. */
class Probes {
  /** `run`: checker.run's signature. */
  constructor(run, retryMs) {
    this.run = run;
    this.retryMs = retryMs;
    /** [cwd, probe] as JSON -> promise of what `interpret` made of its result. */
    this.asked = new Map();
  }

  /** `interpret(result)` of running `probe` in `cwd`, asked once. */
  ask(probe, cwd, seconds, interpret) {
    const key = JSON.stringify([cwd, probe]);
    if (!this.asked.has(key)) {
      const asked = this.run(probe[0], probe.slice(1), cwd, seconds).then((r) => {
        const forget = () => this.asked.get(key) === asked && this.asked.delete(key);
        if (r.spawnError) forget();
        else if (r.timedOut) {
          const timer = setTimeout(forget, this.retryMs);
          if (timer.unref) timer.unref();
        }
        return interpret(r);
      });
      this.asked.set(key, asked);
    }
    return this.asked.get(key);
  }

  /** Forget every answer from `cwd`: its checker's sources changed. */
  forget(cwd) {
    for (const key of [...this.asked.keys()]) if (JSON.parse(key)[0] === cwd) this.asked.delete(key);
  }
}

/** Asks each checker its versions once, and keeps each document's latest check: every provider
 *  reads it, so a document version is checked once whatever asks about it. */
class AnalysisClient {
  /** `run`: checker.run's signature. `waitMs`: the longest a provider waits for a check.
   *  `retryMs`: how long a probe that ran out of time counts as version 1. */
  constructor({ run = checker.run, waitMs = 5000, retryMs = 60000 } = {}) {
    this.waitMs = waitMs;
    /** Each checker's versions, asked once per command and checkout. */
    this.probes = new Probes(run, retryMs);
    /** Bumped when the probes are forgotten: a kept check from before was another checker's. */
    this.epoch = 0;
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

  /** The versions checker `argv` speaks in `cwd` (a checkout), asked once per command there. A
   *  probe that could not run is asked again next time (the check itself reports why); one that
   *  ran out of time counts as version 1 for `retryMs`, so checks do not each wait it out. */
  versions(argv, cwd, seconds) {
    const { probe } = commandsFor(argv);
    if (!probe) return Promise.resolve([1]);
    return this.probes.ask(probe, cwd, seconds, versionsFrom);
  }

  /** Forget the probes' answers in `cwd`: its checker's sources changed, so it may be rebuilt
   *  into one that speaks other versions. */
  forgetVersions(cwd) {
    this.probes.forget(cwd);
    this.epoch++;
  }

  /** The argv to check a document with: version 2's when the checker speaks it, else `argv`. */
  async plan(argv, cwd, seconds) {
    const { v2 } = commandsFor(argv);
    if (v2 && (await this.versions(argv, cwd, seconds)).includes(CONTRACT_VERSION)) return { argv: v2, version: CONTRACT_VERSION };
    return { argv, version: 1 };
  }

  /** Keep a document version's check; `answer` is null from a version-1 checker. */
  store(key, version, text, found, answer) {
    this.entries.set(key, { version, text, found, answer, epoch: this.epoch });
    if (answer) this.answers.set(key, answer);
    else this.answers.delete(key);
    for (const listener of this.listeners) listener(key, answer);
  }

  /** Call `listener(key, answer)` on each stored check; returns a disposable. */
  onStore(listener) {
    this.listeners.add(listener);
    return { dispose: () => this.listeners.delete(listener) };
  }

  /** The kept check of `text` at `key`, if its latest check was of that very text, by the
   *  checker as it is now. */
  cached(key, text) {
    const entry = this.entries.get(key);
    return entry && entry.text === text && entry.epoch === this.epoch ? entry : null;
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

module.exports = { Answer, AnalysisClient, Probes, ANALYSES, CONTRACT_VERSION, Positions, commandsFor, parseCheck, versionsFrom };
