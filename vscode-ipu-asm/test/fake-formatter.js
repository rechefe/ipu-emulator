'use strict';

// A stand-in assembler with `format`, formatting by the editor's rules (language.js) and
// printing what `ipu-as format` prints, and noting each run, so test/formatting.js needs no
// Bazel and no Python:
//
//   node fake-formatter.js <isa.json> <runs.log> [--old] check --json --input -
//   node fake-formatter.js <isa.json> <runs.log> [--old] format [--indent N|tab] [--blank-line] [--range L1:L2] --input -
//
// `--old` makes it an assembler from before `format` (exit 2). A text holding REFUSE is refused
// (exit 3, as ipu-as's safety check does); one holding CRASH fails (exit 1).

const fs = require('fs');

const { Language } = require('../language');
const { spliceLines } = require('../formatting');

const [isaPath, logPath, ...args] = process.argv.slice(2);
const old = args[0] === '--old';
const argv = old ? args.slice(1) : args;
fs.appendFileSync(logPath, `${JSON.stringify(argv)}\n`);

if (argv[0] === 'check') {
  process.stdout.write('[]');
  process.exit(0);
}
if (old) {
  process.stderr.write("Usage: ipu-as [OPTIONS] COMMAND [ARGS]...\nError: No such command 'format'.\n");
  process.exit(2);
}
if (argv.includes('--help')) {
  process.stdout.write('Usage: ipu-as format [OPTIONS]\n');
  process.exit(0);
}

const option = (name) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : undefined);
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => (input += d));
process.stdin.on('end', () => {
  if (input.includes('REFUSE')) {
    process.stderr.write('Error formatting -: formatting would change the program: item 3\n');
    process.exit(3);
  }
  if (input.includes('CRASH')) process.exit(1);
  const lang = new Language(JSON.parse(fs.readFileSync(isaPath, 'utf8')));
  const indent = option('--indent') === 'tab' ? '\t' : ' '.repeat(Number(option('--indent') || 4));
  const options = { indent, blankLine: argv.includes('--blank-line') };
  const range = option('--range');
  if (!range) {
    process.stdout.write(lang.format(input, options));
    return;
  }
  const [from, to] = range.split(':').map((n) => Number(n) - 1);
  process.stdout.write(spliceLines(input, from, to, lang.format(input, { ...options, fromLine: from, toLine: to })));
});
