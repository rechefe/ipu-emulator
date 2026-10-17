'use strict';

// A stand-in `ipu-as check` that answers from recorded answers (test/record-analysis.js), and
// notes each run, so test/analysis.js needs no Bazel and no Python:
//
//   node fake-checker.js <bundle.json> <runs.log> [--old] check --json [--json-version 2 …] --input -
//
// `--old` makes it an assembler from before contract versions: `--json-versions` and
// `--json-version` are usage errors (exit 2). A text it has no recording for gets no
// diagnostics and no analyses.

const fs = require('fs');

const [bundlePath, logPath, ...args] = process.argv.slice(2);
const old = args[0] === '--old';
const argv = old ? args.slice(1) : args;
fs.appendFileSync(logPath, `${JSON.stringify(argv)}\n`);

if (old && argv.some((a) => a.startsWith('--json-version'))) {
  process.stderr.write('Usage: ipu-as check [OPTIONS]\nError: No such option: --json-version\n');
  process.exit(2);
}
if (argv.includes('--json-versions')) {
  process.stdout.write('[1, 2]\n');
  process.exit(0);
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => (input += d));
process.stdin.on('end', () => {
  const { entries } = JSON.parse(fs.readFileSync(bundlePath, 'utf8'));
  const entry = entries.find((e) => e.text === input);
  const answer = entry ? entry.answer : { schemaVersion: 2, diagnostics: [], tokens: [], words: [], jinja: [] };
  const v2 = argv.includes('--json-version');
  process.stdout.write(JSON.stringify(v2 ? answer : answer.diagnostics));
  process.exit(answer.diagnostics.length ? 1 : 0);
});
