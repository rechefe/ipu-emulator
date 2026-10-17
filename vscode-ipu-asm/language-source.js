'use strict';

// Where the editor's language data (isa.json: hover, completion, signature help, slot fit,
// formatting) comes from (E08-25). A checkout's own ISA, generated from it, so that on a branch
// that changes the ISA they agree with its assembler's diagnostics; else the isa.json bundled
// with the extension, built from master.
//
//   CheckoutLanguageSource(root)  `bazel run //src/tools/ipu-gen:ipu-gen -- vscode-data <cache>`,
//                                 once per checkout and source change
//   BundledLanguageSource         data/isa.json shipped with the extension
//
// Generating takes seconds (a Bazel build the first time), so the bundled data answers until
// the checkout's is ready.
//
// Imports nothing from `vscode`: test/language-source.js runs it in plain node.

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

const checker = require('./checker');
const { Language } = require('./language');

/** The newest isa.json layout this extension reads (ipu_gen/vscode_data.py SCHEMA_VERSION). */
const SUPPORTED_SCHEMA = 1;

/** The language data generator. */
const GENERATOR_TARGET = '//src/tools/ipu-gen:ipu-gen';

/** What the language data is generated from: the ISA, and the generator. */
const LANGUAGE_SOURCE_DIRS = [
  path.join('src', 'tools', 'ipu-isa', 'src'),
  path.join('src', 'tools', 'ipu-gen', 'src'),
];

/** The settings `ipuAsm.languageSource` takes. */
const SETTINGS = ['auto', 'checkout', 'bundled'];

/** The data shipped with the extension. */
class BundledLanguageSource {
  constructor(file = path.join(__dirname, 'data', 'isa.json')) {
    this.file = file;
    this.kind = 'bundled';
  }

  /** { lang }: a Language over the bundled isa.json; throws if it is missing or unreadable. */
  async load() {
    return { lang: new Language(JSON.parse(await fs.readFile(this.file, 'utf8'))) };
  }
}

/** A checkout's own ISA, generated into `cacheDir` (one directory per checkout) and kept until
 *  its sources change: fresh while every source is no newer than what the cached data was
 *  generated against (checker.js `isFresh`'s rule, with the sources' newest mtime recorded). */
class CheckoutLanguageSource {
  /** `run`: checker.run's signature. `seconds`: the longest a generation may take. */
  constructor(root, { cacheBase, run = checker.run, seconds = 300, flags = [] }) {
    this.root = root;
    this.kind = 'checkout';
    this.dir = path.join(cacheBase, crypto.createHash('sha256').update(root).digest('hex').slice(0, 16));
    this.run = run;
    this.seconds = seconds;
    this.flags = flags;
  }

  /** The argv generating the data into this checkout's cache directory. */
  command() {
    return ['bazel', 'run', ...this.flags, ...checker.QUIET, GENERATOR_TARGET, '--', 'vscode-data', this.dir];
  }

  /** { lang }: a Language over the checkout's ISA, from the cache when it is fresh, else
   *  generated now; { newer }: its isa.json's layout, newer than this extension reads. Throws
   *  (with what went wrong) when it cannot be generated or read. */
  async load() {
    const newest = await checker.newestSourceMtime(this.root, LANGUAGE_SOURCE_DIRS);
    if (newest === 0) throw new Error(`no ISA sources under ${this.root}`);
    const isaFile = path.join(this.dir, 'data', 'isa.json');
    const stampFile = path.join(this.dir, 'stamp.json');
    const stamp = await fs.readFile(stampFile, 'utf8').then(JSON.parse, () => null);
    if (!stamp || stamp.root !== this.root || stamp.sources < newest) {
      await fs.mkdir(this.dir, { recursive: true });
      const [command, ...args] = this.command();
      const r = await this.run(command, args, this.root, this.seconds);
      if (r.code !== 0) {
        const tail = (r.stderr || '').trim().split('\n').slice(-3).join(' ').slice(0, 300);
        throw new Error(`generating the checkout's language data failed: ${tail || `exit ${r.code}`}`);
      }
      // Sampled before the run: an edit made during it is not credited to it.
      await fs.writeFile(stampFile, JSON.stringify({ root: this.root, sources: newest }));
    }
    const isa = JSON.parse(await fs.readFile(isaFile, 'utf8'));
    return (isa.schemaVersion ?? 0) > SUPPORTED_SCHEMA ? { newer: isa.schemaVersion } : { lang: new Language(isa) };
  }
}

/** Which data a document uses: { use: 'checkout' | 'bundled', why, warn }. `why` says why the
 *  bundled data is in use (null for the checkout's); `warn`, if set, is to be shown once.
 *  `setting`: ipuAsm.languageSource. `root`: the document's checkout, or null. `checkout`: what
 *  loading the checkout's data gave ({ lang }, { newer } or { error }), or undefined while it
 *  loads. */
function select({ trusted, setting = 'auto', root, checkout }) {
  const forced = setting === 'checkout';
  const bundled = (why, warn = null) => ({ use: 'bundled', why, warn });
  if (setting === 'bundled') return bundled('ipuAsm.languageSource is "bundled"');
  if (!root) return bundled('the file is in no IPU checkout');
  if (!trusted) return bundled('the workspace is in Restricted Mode');
  if (!checkout) return bundled("the checkout's language data is being generated");
  if (checkout.error) return bundled(checkout.error, forced ? `IPU Assembly: using the bundled ISA. ${checkout.error}` : null);
  if (checkout.newer) {
    const why = `the checkout's isa.json is schema version ${checkout.newer}; this extension reads up to ${SUPPORTED_SCHEMA}`;
    return bundled(why, `IPU Assembly: using the bundled ISA, since ${why}. Update the extension.`);
  }
  return { use: 'checkout', why: null, warn: null };
}

module.exports = { BundledLanguageSource, CheckoutLanguageSource, GENERATOR_TARGET, LANGUAGE_SOURCE_DIRS, SETTINGS, SUPPORTED_SCHEMA, select };
