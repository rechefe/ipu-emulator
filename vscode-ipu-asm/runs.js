'use strict';

// Command lines for running kernels. The kernel manifest is the only source of
// kernel names, cases, options and targets; nothing here knows a naming rule.
// Plain node, no `vscode` (see test/runs.js).

const cp = require('child_process');
const path = require('path');
const { QUIET } = require('./checker');

/** The one label the extension has to know (as checker.js knows the assembler's). */
const MANIFEST_TARGET = '//src/tools/ipu-apps:kernel_manifest';

/** Whether the manifest command failed for want of the target: a checkout predating it. */
function manifestMissing(stderr) {
  return /no such (target|package)[^\n]*(kernel_manifest|ipu-apps)|'kernel_manifest' not declared/.test(String(stderr || ''));
}

const manifestCommand = () => ['bazel', 'run', ...QUIET, MANIFEST_TARGET];

/** Quote one argument for a POSIX shell, only when it needs it. */
function shellQuote(arg) {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(arg) ? arg : `'${String(arg).replace(/'/g, "'\\''")}'`;
}

/** An option's type as the runner parses it: bool, int, float or str. A manifest from
 *  before `type` was listed is read from the default. */
const optionType = (o) => o.type || (typeof o.default === 'boolean' ? 'bool'
  : typeof o.default === 'number' ? (Number.isInteger(o.default) ? 'int' : 'float') : 'str');

/** A case's option arguments, `values` over its defaults, e.g. `--rows 4 --no-wide`. */
function optionArgs(caseEntry, values = {}) {
  return Object.entries((caseEntry && caseEntry.options) || {}).flatMap(([name, o]) => {
    const value = Object.hasOwn(values, name) ? values[name] : o.default;
    return optionType(o) === 'bool' ? [value ? o.flag : o.negated_flag] : [o.flag, shellQuote(String(value))];
  }).join(' ');
}

/** Why `values` do not fit a case's options, or null. */
function invalidValues(caseEntry, values) {
  const fits = { bool: (v) => typeof v === 'boolean', int: Number.isInteger, float: Number.isFinite, str: (v) => typeof v === 'string' };
  const bad = Object.entries(caseEntry.options).find(([name, o]) => Object.hasOwn(values, name) && !fits[optionType(o)](values[name]));
  return bad ? `${bad[0]} must be ${{ bool: 'on or off', int: 'a whole number', float: 'a number', str: 'text' }[optionType(bad[1])]}` : null;
}

/** A kernel's cases: the registry's, with the ones saved in ipuAsm.cases (`{ name: { base,
 *  options } }`) added or over them. Each is `{ base, entry, values, saved, builtIn }` and runs
 *  as `--case <base>` with its values; `default` comes first. A saved case whose base the
 *  registry no longer has is left out. */
function kernelCases(kernel, saved = {}) {
  const cases = {};
  for (const name of Object.keys(kernel.cases)) cases[name] = { base: name, values: {} };
  for (const [name, s] of Object.entries(saved || {})) {
    const base = (s && s.base) || (kernel.cases[name] ? name : 'default');
    if (kernel.cases[base]) cases[name] = { base, values: (s && s.options) || {}, saved: true };
  }
  const names = Object.keys(cases).sort((a, b) => (a === 'default' ? -1 : b === 'default' ? 1 : a.localeCompare(b)));
  return Object.fromEntries(names.map((name) => [name,
    { ...cases[name], entry: kernel.cases[cases[name].base], builtIn: Object.hasOwn(kernel.cases, name) }]));
}

/** The shell command to run, debug, test or benchmark a kernel, or null with no target
 *  for it. `options` is shell text the user saw and may have edited, so it is passed as
 *  typed; `flags` are ipuAsm.bazelFlags. */
function kernelCommand(kernel, action, { caseName, options = '', flags = [] } = {}) {
  const targets = kernel.targets || {};
  const bazel = (verb, target, ...tail) =>
    target ? ['bazel', verb, ...flags.map(shellQuote), target, ...tail].join(' ') : null;
  const passCase = ['--', '--case', shellQuote(caseName || 'default'), options].filter(Boolean).join(' ');
  const actions = {
    run: ['run', targets.run, passCase],
    // The terminal debugger: .bazelrc's `run:debug` config.
    debug: ['run --config=debug', targets.run, passCase],
    test: ['test', targets.test],
    benchmark: ['run', targets.benchmark],
  };
  if (!Object.hasOwn(actions, action)) throw new Error(`unknown action ${action}`);
  return bazel(...actions[action]);
}

const familyCommand = (family, { flags = [] } = {}) => ['bazel', 'test', ...flags.map(shellQuote), family.suite].join(' ');

/** The query argv for `op` and its `name=value` params. `configured` (ipuAsm.queryCommand,
 *  e.g. a virtualenv's script) replaces `bazel run` of the manifest's query target. */
function queryCommand(manifest, op, params, configured = []) {
  // Quotes group a value with spaces (shape="32, 300"); a `name=` left from the
  // prefill, which lists every parameter, is dropped rather than sent empty.
  const words = (params.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [])
    .map((w) => w.replace(/"([^"]*)"|'([^']*)'/g, (_, a, b) => a ?? b))
    .filter((w) => w && !/^[^=]+=$/.test(w));
  const base = configured.length ? configured : ['bazel', 'run', ...QUIET, manifest.query, '--'];
  return [...base, '--json', op, ...words];
}

/** The kernel whose .asm is `fsPath`, matched against the manifest's paths. */
function kernelForFile(manifest, root, fsPath) {
  const target = path.resolve(fsPath);
  const found = Object.entries(manifest ? manifest.kernels : {}).find(([, kernel]) => path.resolve(root, kernel.asm) === target);
  return found ? { name: found[0], kernel: found[1] } : null;
}

/** The directory holding every kernel, so a change beneath it can refresh
 *  the manifest: the deepest common ancestor of the kernel folders. */
function kernelsRoot(manifest) {
  const dirs = Object.values((manifest && manifest.kernels) || {}).map((k) => path.posix.dirname(k.asm).split('/'));
  if (!dirs.length) return null;
  const differ = dirs[0].findIndex((part, i) => !dirs.every((d) => d[i] === part));
  return dirs[0].slice(0, differ < 0 ? undefined : differ).join('/');
}

/** Kernels grouped by family, sorted, for the sidebar. */
function familyTree(manifest) {
  const families = new Map();
  for (const [name, kernel] of Object.entries(manifest.kernels)) {
    if (!families.has(kernel.family)) families.set(kernel.family, []);
    families.get(kernel.family).push(name);
  }
  return [...families.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([family, kernels]) => ({ family, entry: manifest.families[family] || null, kernels: kernels.sort() }));
}

/** A one-line summary of a query verdict (query --json). */
function describeVerdict(verdict) {
  return `${verdict.supported ? 'SUPPORTED' : 'NOT SUPPORTED'}${verdict.app ? ` (${verdict.app})` : ''}: ${verdict.reason}`;
}

/** From `ps -o tpgid=,pgid=`: whether the process's group is its terminal's foreground
 *  group, or null with no terminal or unexpected output. */
function foregroundFrom(psOutput) {
  const [tpgid, pgid] = String(psOutput).trim().split(/\s+/).map(Number);
  return tpgid > 0 && pgid > 0 ? tpgid === pgid : null;
}

/** Whether the shell `pid` is at its prompt: holding its terminal's foreground, no command
 *  runs in front of it, whoever typed it (a background job does not count). Null where
 *  that cannot be told (no `ps`, as on Windows). */
function atPrompt(pid) {
  return new Promise((resolve) => {
    if (!pid) return resolve(null);
    cp.execFile('ps', ['-o', 'tpgid=,pgid=', '-p', String(pid)], { timeout: 2000 }, (err, stdout) => {
      resolve(err ? null : foregroundFrom(stdout));
    });
  });
}

module.exports = {
  MANIFEST_TARGET, atPrompt, describeVerdict, familyCommand, familyTree, foregroundFrom, invalidValues, kernelCases,
  kernelCommand, kernelForFile, kernelsRoot, manifestCommand, manifestMissing, optionArgs, optionType, queryCommand, shellQuote,
};
