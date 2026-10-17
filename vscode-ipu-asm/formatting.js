'use strict';

// The checkout's assembler as the editor's formatter (`ipu-as format`, E08-24): the same rules
// as language.js `format`, and a check that the result is the same program. An assembler from
// before `format` (its `format --help` fails), or a checker command of another shape, formats
// with the editor's own rules instead.
//
// Imports nothing from `vscode`: test/formatting.js runs it in plain node.

const checker = require('./checker');
const { Probes } = require('./analysis');

/** `format`'s exit code when it refuses (the result would be another program) or cannot read
 *  the input (src/tools/ipu-as-py/src/ipu_as/cli.py `_CANNOT_FORMAT`). */
const CANNOT_FORMAT = 3;

/** The assembler command a checker argv (`… check … --input`) runs, before its `check`; null
 *  for a command of another shape (a user's wrapper, a stand-in), which may not have `format`. */
function assemblerOf(argv) {
  const at = argv.indexOf('check');
  return at > 0 && argv[argv.length - 1] === '--input' ? argv.slice(0, at) : null;
}

/** The `format` argv for options { indent, blankLine, fromLine, toLine } (0-based lines; no
 *  range for the whole text), reading stdin. */
function formatArgv(assembler, { indent, blankLine, fromLine, toLine }, lineCount) {
  const argv = [...assembler, 'format', '--indent', indent === '\t' ? 'tab' : String(indent.length)];
  if (blankLine) argv.push('--blank-line');
  if (fromLine > 0 || toLine < lineCount - 1) argv.push('--range', `${fromLine + 1}:${toLine + 1}`);
  return [...argv, '--input', '-'];
}

/** Where lines [from, to] (0-based, inclusive; `to` past the end means the last line) are in
 *  `text`: [start, end) offsets, the end before the last line's `\r`. */
function lineSpan(text, from, to) {
  const lines = text.split('\n');
  const last = Math.min(to, lines.length - 1);
  const offsetOf = (n) => lines.slice(0, n).reduce((at, line) => at + line.length + 1, 0);
  return [offsetOf(from), offsetOf(last) + lines[last].replace(/\r$/, '').length];
}

/** `text` with lines [from, to] replaced by `lines`: what `format --range` prints. */
function spliceLines(text, from, to, lines) {
  const [start, end] = lineSpan(text, from, to);
  return text.slice(0, start) + lines + text.slice(end);
}

/** Lines [from, to] of `out`, which is `text` with only those lines formatted (what `format
 *  --range` prints), as language.js `format` returns them; null if the lines around them changed. */
function linesOf(text, out, from, to) {
  const [start, end] = lineSpan(text, from, to);
  const before = text.slice(0, start);
  const after = text.slice(end);
  if (out.length < before.length + after.length || !out.startsWith(before) || !out.endsWith(after)) return null;
  return out.slice(before.length, out.length - after.length);
}

/** Formats with the checkout's assembler where it can. */
class CheckoutFormatter {
  /** `run`: checker.run's signature. `retryMs`: how long a probe that ran out of time counts as
   *  "cannot format". */
  constructor({ run = checker.run, retryMs = 60000 } = {}) {
    this.run = run;
    this.probes = new Probes(run, retryMs);
  }

  /** Whether checker `argv` has `format` in `cwd`, asked once per command there. */
  supports(argv, cwd, seconds) {
    const assembler = assemblerOf(argv);
    if (!assembler) return Promise.resolve(false);
    return this.probes.ask([...assembler, 'format', '--help'], cwd, seconds, (r) => r.code === 0);
  }

  /** Forget what the checker in `cwd` said: its sources changed. */
  forget(cwd) {
    this.probes.forget(cwd);
  }

  /** Lines [fromLine, toLine] of `text` formatted by the assembler: { lines } (as language.js
   *  `format` returns them), { refused } (why the assembler kept the text as it is), or null when
   *  it cannot be asked (no `format`, or the run failed), for the editor's own rules to answer. */
  async format(argv, cwd, seconds, text, options) {
    if (!(await this.supports(argv, cwd, seconds))) return null;
    const [command, ...args] = formatArgv(assemblerOf(argv), options, text.split('\n').length);
    const r = await this.run(command, args, cwd, seconds, { input: text });
    if (r.code === CANNOT_FORMAT) return { refused: r.stderr.trim() || 'ipu-as format refused' };
    if (r.code !== 0) return null;
    const lines = linesOf(text, r.stdout, options.fromLine, options.toLine);
    return lines === null ? null : { lines };
  }
}

module.exports = { CANNOT_FORMAT, CheckoutFormatter, assemblerOf, formatArgv, linesOf, spliceLines };
