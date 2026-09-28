'use strict';

// `node test/integration/run.js`: launch a real VS Code with this extension on a temporary copy of
// fixture/ and run suite.js, then layouts.js per layout. Needs the generated files (`bazel run
// //vscode-ipu-asm:gen_vscode`), `@vscode/test-electron` installed without saving, and a display
// (else `xvfb-run -a`). IPU_TEST_CHECK_COMMAND, a JSON argv such as the built ipu-as plus
// `check --json --input`, adds the checker-backed checks.

const fs = require('fs');
const os = require('os');
const path = require('path');

const { runTests } = require('@vscode/test-electron');

const VSCODE_VERSION = '1.139.0'; // pinned so a run is reproducible; IPU_TEST_VSCODE_VERSION overrides it

/** Write `text` to `file`, making its folder. */
function write(file, text, mode) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text, { mode });
}
/** A copy of fixture/ at `dir` marked as an IPU checkout (a Bazel workspace
 *  holding the assembler's package), so the repository gains no nested module. */
function makeCheckout(dir) {
  fs.cpSync(path.join(__dirname, 'fixture'), dir, { recursive: true });
  for (const marker of ['MODULE.bazel', 'src/tools/ipu-as-py/BUILD.bazel']) write(path.join(dir, marker), '');
  return dir;
}
/** `folder`'s .vscode/settings.json, each key prefixed with `ipuAsm.`. */
const writeSettings = (folder, settings) => write(path.join(folder, '.vscode', 'settings.json'),
  JSON.stringify(Object.fromEntries(Object.entries(settings).map(([k, v]) => [`ipuAsm.${k}`, v])), null, 2));

const node = (script) => [process.execPath, '-e', script];
const CLEAN = "process.stdout.write('[]')";
/** A command that notes where it ran in `marker`, then does `then`. */
const recorder = (marker, then) => node(`require('fs').appendFileSync(${JSON.stringify(marker)}, process.cwd() + '\\n'); ${then}`);
// The fixture's manifest: this tests the extension's handling of one; the Python tests test producing it.
const MANIFEST = node("process.stdout.write(require('fs').readFileSync('manifest.json', 'utf8'))");
// What Bazel says in a checkout from before the kernel manifest.
const FAIL_NO_TARGET = `process.stderr.write(${JSON.stringify("ERROR: Skipping '//src/tools/ipu-apps:kernel_manifest': no such target '//src/tools/ipu-apps:kernel_manifest': target 'kernel_manifest' not declared in package 'src/tools/ipu-apps'")}); process.exit(1);`;

const extension = path.resolve(__dirname, '..', '..');
const launch = (workspace, suite, env) => runTests({
  version: process.env.IPU_TEST_VSCODE_VERSION || VSCODE_VERSION,
  // .vscode-test/ is ignored by git and by the vsix.
  cachePath: process.env.IPU_TEST_VSCODE_CACHE || path.join(extension, '.vscode-test'),
  extensionDevelopmentPath: extension,
  extensionTestsPath: path.join(__dirname, suite),
  launchArgs: [workspace, '--disable-extensions', '--disable-workspace-trust'],
  extensionTestsEnv: { IPU_TEST_WORKSPACE: workspace, ...env },
});

async function main() {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'ipu-asm-it-'));
  // Terminals inherit this PATH, so the kernel buttons' `bazel ...` reaches a
  // stand-in and no real Bazel server starts in a temporary folder.
  const bin = path.join(temp, 'bin');
  write(path.join(bin, 'bazel'), '#!/bin/sh\necho "[stand-in bazel] $*"\n', 0o755);
  process.env.PATH = `${bin}${path.delimiter}${process.env.PATH}`;
  try {
    // The checkout itself, opened as the folder: the full suite.
    const workspace = makeCheckout(path.join(temp, 'checkout'));
    const check = process.env.IPU_TEST_CHECK_COMMAND;
    writeSettings(workspace, {
      manifestCommand: MANIFEST,
      queryCommand: [process.execPath, path.join(workspace, 'query.js')],
      // Without one, a checker that cannot start: rename falls back to its lexical check.
      checkCommand: check ? JSON.parse(check) : [path.join(workspace, 'no-such-checker')],
    });
    await launch(workspace, 'suite.js', { IPU_TEST_HAS_CHECKER: check ? '1' : '' });

    // The other ways a folder can relate to a checkout (see layouts.js), each in its own window.
    const parent = path.join(temp, 'parent');
    makeCheckout(path.join(parent, 'work', 'ipu-emulator'));
    writeSettings(parent, { manifestCommand: MANIFEST, checkCommand: recorder(path.join(temp, 'parent-checks'), CLEAN) });
    // Another project with its own .asm and .s: nothing may run.
    const other = path.join(temp, 'other');
    for (const [file, text] of [['MODULE.bazel', ''], ['boot/start.asm', 'section .text\nglobal _start\n_start:\n    mov eax, 1\n    int 0x80\n'],
      ['boot/crt0.s', '.globl _start\n_start:\n    movl $1, %eax\n']]) write(path.join(other, file), text);
    writeSettings(other, { manifestCommand: recorder(path.join(temp, 'other-manifest'), 'process.exit(1)') });
    const old = makeCheckout(path.join(temp, 'old'));
    writeSettings(old, { manifestCommand: node(FAIL_NO_TARGET), checkCommand: node(CLEAN) });
    // Switched between versions while open: the manifest command answers like Bazel for whatever is there.
    const switching = makeCheckout(path.join(temp, 'switch'));
    writeSettings(switching, {
      manifestCommand: node(`const fs = require('fs'); if (fs.existsSync('manifest.json')) process.stdout.write(fs.readFileSync('manifest.json', 'utf8')); else { ${FAIL_NO_TARGET} }`),
      checkCommand: node(CLEAN),
    });
    for (const [layout, folder] of [['parent', parent], ['other', other], ['old', old], ['switch', switching]]) {
      await launch(folder, 'layouts.js', { IPU_TEST_LAYOUT: layout, IPU_TEST_TEMP: temp });
    }
  } catch (err) {
    console.error(`Integration tests failed: ${err.message || err}`);
    process.exit(1);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
}

main();
