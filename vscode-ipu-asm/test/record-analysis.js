'use strict';

// Records what a version-2 checker answers for the analysis fixtures, so test/analysis.js and
// test/analysis-agreement.js can run with a fake checker (no Bazel, no Python):
//
//   node test/record-analysis.js <out.json> '<checker argv>' [file.asm ...]
//
// The argv is JSON and ends like the extension's: `["bazel", "run", "//src/tools/ipu-as:ipu-as",
// "--", "check", "--json", "--input"]`. Without files it records the committed set (FILES below and
// fixtures/analysis-snippets.json) to <out.json>; re-record test/fixtures/analysis-v2.json with it
// when the assembler's answers or those files change. With files, each is recorded by its path
// relative to the repository: a bundle of the whole corpus for analysis-agreement.js.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { commandsFor } = require('../analysis');
const checker = require('../checker');

const REPO = path.resolve(__dirname, '..', '..');
/** The committed recordings: the extension's own corpus and a kernel with a `for` (the
 * `macro` kernel, a MobileViT packed program, moved to //kernels, E07-23; the
 * `macro-param` snippet covers macros). */
const FILES = [
  'vscode-ipu-asm/test/corpus/edge_cases.asm',
  ...fs.readdirSync(path.join(__dirname, 'corpus', 'negative')).filter((f) => f.endsWith('.asm')).sort()
    .map((f) => `vscode-ipu-asm/test/corpus/negative/${f}`),
  'vscode-ipu-asm/test/integration/fixture/kernels/demo/demo_kernel/demo_kernel.asm',
  'kernels/sampling/sample_descriptors/sample_descriptors.asm',
];
const SNIPPETS = path.join(__dirname, 'fixtures', 'analysis-snippets.json');
const digest = (text) => crypto.createHash('sha256').update(text).digest('hex');

async function record(argv, name, text) {
  const v2 = commandsFor(argv).v2;
  if (!v2) throw new Error(`not a checker argv ending "check --json --input": ${JSON.stringify(argv)}`);
  const r = await checker.run(v2[0], [...v2.slice(1), '-'], REPO, 600, { input: text });
  const answer = JSON.parse(r.stdout); // exit 1 only means there are diagnostics
  if (answer.schemaVersion !== 2) throw new Error(`${name}: not a version 2 answer: ${r.stdout.slice(0, 200)} ${r.stderr}`);
  return answer;
}

async function main() {
  const [out, argvJson, ...files] = process.argv.slice(2);
  if (!argvJson) {
    console.error("usage: record-analysis.js <out.json> '<checker argv JSON>' [file.asm ...]");
    process.exit(2);
  }
  const argv = JSON.parse(argvJson);
  const entries = [];
  const paths = files.length ? files.map((f) => path.relative(REPO, path.resolve(f))) : FILES;
  for (const file of paths) {
    const text = fs.readFileSync(path.join(REPO, file), 'utf8');
    entries.push({ name: file, file, sha256: digest(text), answer: await record(argv, file, text) });
  }
  if (!files.length) {
    for (const [name, text] of Object.entries(JSON.parse(fs.readFileSync(SNIPPETS, 'utf8')))) {
      entries.push({ name: `snippet:${name}`, text, answer: await record(argv, name, text) });
    }
  }
  fs.writeFileSync(out, `${JSON.stringify({ contract: 2, entries })}\n`);
  console.log(`Recorded ${entries.length} answers to ${out}.`);
}

/** The entries of a recorded bundle, each with its text: a file's is read from the repository
 *  and must be what was recorded (`stale` names those that changed since). */
function load(bundlePath) {
  const bundle = JSON.parse(fs.readFileSync(bundlePath, 'utf8'));
  const snippets = JSON.parse(fs.readFileSync(SNIPPETS, 'utf8'));
  return bundle.entries.map((e) => {
    const text = e.file ? fs.readFileSync(path.join(REPO, e.file), 'utf8') : e.text;
    const stale = e.file ? digest(text) !== e.sha256 : snippets[e.name.replace(/^snippet:/, '')] !== e.text;
    return { ...e, text, stale };
  });
}

if (require.main === module) main().catch((err) => { console.error(err); process.exit(1); });

module.exports = { FILES, REPO, load };
