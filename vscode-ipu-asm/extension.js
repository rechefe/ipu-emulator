'use strict';

// Surface assembler errors as squiggles.
//
// The TextMate grammar colours tokens but cannot report errors — it assigns
// scopes to spans and has no notion of what the grammar expects next. Validity
// is not a lexical property: whether a comma is legal depends on the parser's
// state, not on the characters. So the only thing that can answer "is this
// valid" is the assembler, which this runs.
//
// It shells out to `ipu-as check --json` rather than speaking LSP, which keeps
// the extension dependency-free: no vscode-languageclient, no bundled
// node_modules, and the packaged vsix stays a few kilobytes.

const cp = require('child_process');
const fs = require('fs');
const path = require('path');
const vscode = require('vscode');

const checker = require('./checker');

const LANGUAGE_ID = 'ipu-asm';

// Mnemonics contain dots (ACC.ADD.FIRST, MULT.RC.VV). The editor's default word
// pattern stops at the first dot, which would look up "ACC" and find nothing.
const WORD_PATTERN = /[A-Za-z0-9_][A-Za-z0-9_.]*/;
const CONFIG_SECTION = 'ipuAsm';

/** Debounce so a burst of keystrokes runs the checker once. */
const DEBOUNCE_MS = 400;

let diagnostics;
let statusItem;
/** Debounce timers, keyed by document: a single shared timer let an edit in
 *  any other file cancel a pending .asm check. */
const pending = new Map();
let warnedOnce = false;
let lastError = null;
const running = new Map();

/** Explain a checker failure in terms of what the user can do about it. */
function explain(stderr, command) {
  // The most likely cause by far: `check` is added by the extension's own
  // change, so a workspace on a branch predating it has an ipu-as without it.
  if (/no such command/i.test(stderr)) {
    return (
      `the assembler in this workspace has no "check" subcommand, so ` +
      `diagnostics cannot run. It is added alongside this extension — a branch ` +
      `predating that will not have it.`
    );
  }
  if (/not found|ENOENT/i.test(stderr)) {
    return `"${command}" could not be run. Set ipuAsm.checkCommand to point at a working assembler.`;
  }
  const tail = stderr.trim().split('\n').slice(-3).join(' ').slice(0, 300);
  return tail || `"${command}" produced no output. See the Developer Tools console.`;
}

function markUnavailable(reason) {
  statusItem.text = '$(warning) IPU: checks off';
  statusItem.tooltip = `IPU Assembly diagnostics are not running:\n\n${reason}`;
  statusItem.command = 'ipuAsm.showCheckerStatus';
  statusItem.show();

  // Once per session. Repeating it on every keystroke would be worse than the
  // silence it replaces.
  if (!warnedOnce) {
    warnedOnce = true;
    vscode.window
      .showWarningMessage(`IPU Assembly: diagnostics are off — ${reason}`, 'Details')
      .then((choice) => {
        if (choice === 'Details') vscode.commands.executeCommand('ipuAsm.showCheckerStatus');
      });
  }
}

function markAvailable() {
  statusItem.text = '$(check) IPU';
  statusItem.tooltip = 'IPU Assembly diagnostics are running.';
  statusItem.command = undefined;
  statusItem.show();
}

// ---------------------------------------------------------------------------
// Hover
// ---------------------------------------------------------------------------

// Assembly is case-insensitive -- softmax_rows.asm mixes `ACC.ADD.FIRST` and
// `acc.sub` -- so every lookup goes through a lower-cased key.
const instructionIndex = new Map();
const registerIndex = new Set();
let slots = {};

function loadHoverData() {
  const dataPath = path.join(__dirname, 'data', 'isa.json');
  let isa;
  try {
    isa = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
  } catch (err) {
    // Hover is a convenience; losing it must not take highlighting with it.
    console.warn(`[ipu-asm] no hover data (${err.message})`);
    return;
  }
  for (const [mnemonic, forms] of Object.entries(isa.instructions)) {
    instructionIndex.set(mnemonic.toLowerCase(), { mnemonic, forms });
  }
  for (const name of isa.registers) registerIndex.add(name.toLowerCase());
  slots = isa.slots || {};
}

function describeSlot(slot) {
  const meta = slots[slot];
  return meta && meta.hardware === false ? `${slot} (simulation-only)` : slot;
}

function renderInstruction(mnemonic, forms) {
  const doc = (forms[0] && forms[0].doc) || {};
  const out = [`**${doc.title || mnemonic}**`, ''];

  // A mnemonic can occupy several slots -- NOP exists in all nine, each with its
  // own summary -- so list every form rather than silently showing the first.
  for (const form of forms) {
    const label = form.pseudo ? `pseudo → ${form.expandsTo}` : describeSlot(form.slot);
    const summary = (form.doc && form.doc.summary) || '';
    out.push(`- \`${label}\`${summary ? ` — ${summary}` : ''}`);
  }

  if (doc.syntax) out.push('', '```ipu-asm', doc.syntax, '```');

  if (Array.isArray(doc.operands) && doc.operands.length) {
    out.push('', '**Operands**', '');
    // Already markdown in instruction_spec.py -- pass straight through.
    for (const operand of doc.operands) out.push(`- ${operand}`);
  }
  if (doc.operation) out.push('', `**Operation** — ${doc.operation}`);
  if (doc.example) out.push('', '**Example**', '', '```ipu-asm', doc.example, '```');
  if (doc.notes) out.push('', `**Notes** — ${doc.notes}`);

  return out.join('\n');
}

function renderRegister(name) {
  const upper = name.toUpperCase();
  if (/^lrd/.test(name)) {
    const n = Number(name.slice(3));
    return `**\`${upper}\`** — register pair: \`LR${n + 1}\`:\`LR${n}\` as one 8-byte-element value.`;
  }
  if (/^lr/.test(name)) return `**\`${upper}\`** — loop register (read/write).`;
  if (/^cr/.test(name)) return `**\`${upper}\`** — constant register (read-only in assembly; set by the host harness).`;
  return `**\`${upper}\`** — multiply-stage register.`;
}

const hoverProvider = {
  provideHover(document, position) {
    const range = document.getWordRangeAtPosition(position, WORD_PATTERN);
    if (!range) return undefined;
    const key = document.getText(range).toLowerCase();

    const instruction = instructionIndex.get(key);
    if (instruction) {
      return new vscode.Hover(
        new vscode.MarkdownString(renderInstruction(instruction.mnemonic, instruction.forms)),
        range
      );
    }
    if (registerIndex.has(key)) {
      return new vscode.Hover(new vscode.MarkdownString(renderRegister(key)), range);
    }
    return undefined;
  },
};

/** Resolved checker binary per workspace folder: a path once the build has
 *  finished, null when this workspace has no Bazel or the build failed. Absent
 *  means resolution has not finished, which is not the same as failed -- both
 *  take the Bazel path, but only the second stops trying. */
const checkerBinaries = new Map();
const resolvingFolders = new Set();
/** Source stamp each folder's checker was last built against. See isFresh:
 *  Bazel rebuilds on content, so a re-dated but unchanged source leaves the
 *  binary's mtime behind forever, and this is what keeps that from pinning
 *  every check to the slow path. */
const checkerStamps = new Map();

/** Run a command to completion, resolving to its stdout or an error. */
function runToCompletion(command, args, cwd) {
  return new Promise((resolve) => {
    let child;
    try {
      child = cp.spawn(command, args, { cwd });
    } catch (err) {
      resolve({ error: err.message });
      return;
    }
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => (stdout += d));
    child.stderr.on('data', (d) => (stderr += d));
    child.on('error', (err) => resolve({ error: err.message }));
    child.on('close', (code) =>
      resolve(code === 0 ? { stdout } : { error: stderr.trim() || `exit ${code}` })
    );
  });
}

/** Build the checker once and remember where Bazel put it.
 *
 *  Never awaited by a check: a cold Bazel server takes ten seconds or more to
 *  answer, and a check that waited for it would be slower than the path it is
 *  replacing. Checks keep using `bazel run` until this lands. */
async function resolveChecker(root) {
  // Sampled before the build, so an edit made while it runs invalidates the
  // result rather than being swallowed by it.
  const stamp = checker.newestSourceMtime(root);
  const build = await runToCompletion(
    'bazel',
    ['build', '--ui_event_filters=-info,-stdout,-stderr', '--noshow_progress', checker.CHECKER_TARGET],
    root
  );
  if (build.error) return null;

  // bazel-bin is a convenience symlink: absent after a clean, renamed by
  // --symlink_prefix, elsewhere under a different --config. Ask instead.
  const info = await runToCompletion('bazel', ['info', 'bazel-bin'], root);
  if (info.error) return null;

  const binary = checker.checkerBinary(info.stdout.trim(), process.platform);
  if (!fs.existsSync(binary)) return null;
  checkerStamps.set(root, Math.max(checkerStamps.get(root) || 0, stamp));
  return binary;
}

function ensureChecker(root) {
  if (!root || checkerBinaries.has(root) || resolvingFolders.has(root)) return;
  resolvingFolders.add(root);
  resolveChecker(root).then(
    (binary) => {
      resolvingFolders.delete(root);
      checkerBinaries.set(root, binary);
      if (!binary) console.warn('[ipu-asm] no prebuilt checker; using `bazel run` per check');
    },
    (err) => {
      resolvingFolders.delete(root);
      checkerBinaries.set(root, null);
      console.warn(`[ipu-asm] could not resolve the checker (${err.message})`);
    }
  );
}

/** Drop a resolution whose binary no longer runs. */
function forgetChecker(command) {
  for (const [root, binary] of checkerBinaries) {
    if (binary === command) checkerBinaries.delete(root);
  }
}

function checkCommand(folder) {
  const configured = vscode.workspace
    .getConfiguration(CONFIG_SECTION, folder && folder.uri)
    .get('checkCommand');
  if (Array.isArray(configured) && configured.length) return configured;

  const root = folder && folder.uri.fsPath;
  if (!root) return checker.bazelCommand();
  ensureChecker(root);
  return checker.checkCommandFor(
    checkerBinaries.get(root) || null,
    root,
    checkerStamps.get(root) || 0
  );
}

/** Run the checker over a document's current buffer; resolves to diagnostics. */
function runCheck(document) {
  const folder = vscode.workspace.getWorkspaceFolder(document.uri);
  const cwd = folder ? folder.uri.fsPath : path.dirname(document.uri.fsPath);
  const argv = checkCommand(folder);
  const [command, ...args] = argv;
  // The fallback rebuilds on its way to checking, so a run that comes back with
  // diagnostics has also brought the binary up to date with these sources.
  // Sampled before the run, so an edit made during it is not credited to it.
  const rebuilding = Boolean(folder) && checker.isBazelCommand(argv);
  const stamp = rebuilding ? checker.newestSourceMtime(cwd) : 0;
  const key = document.uri.toString();
  // Check what is on screen, not what is on disk: this runs while typing, and
  // a path would make every squiggle describe the last save instead.
  const text = document.getText();

  return new Promise((resolve) => {
    let child;
    try {
      child = cp.spawn(command, [...args, '-'], { cwd });
    } catch (err) {
      resolve({ error: `could not start ${command}: ${err.message}`, command });
      return;
    }

    // Supersede an in-flight run for the same document rather than racing it.
    // The kill makes that child close with no output, which must not be read as
    // the checker failing — it would clear diagnostics and flip the status bar
    // on nothing but a keystroke.
    const previous = running.get(key);
    if (previous) {
      previous.superseded = true;
      previous.kill();
    }
    child.superseded = false;
    running.set(key, child);

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => (stdout += d));
    child.stderr.on('data', (d) => (stderr += d));

    child.on('error', (err) => {
      if (child.superseded) return resolve({ superseded: true });
      // A prebuilt checker that will not start is a stale resolution, not a
      // broken toolchain: `bazel clean` deletes the binary out from under us.
      // Forgetting it sends the next check down the Bazel path, which rebuilds.
      forgetChecker(command);
      resolve({ error: `${command}: ${err.message}`, command });
    });
    child.on('close', () => {
      // Only clear the map entry if it is still ours; a newer run may have
      // replaced it already.
      if (running.get(key) === child) running.delete(key);
      if (child.superseded) return resolve({ superseded: true });
      // Exit code 1 just means "found problems", so it is not an error here.
      try {
        const found = JSON.parse(stdout);
        if (rebuilding) {
          checkerStamps.set(cwd, Math.max(checkerStamps.get(cwd) || 0, stamp));
          ensureChecker(cwd);
        }
        resolve({ found });
      } catch {
        resolve({ error: explain(stderr || stdout, command), command });
      }
    });

    child.stdin.on('error', () => {}); // a killed child closes stdin first
    child.stdin.end(text);
  });
}

function toVscodeDiagnostic(raw) {
  const range = new vscode.Range(
    Math.max(raw.line, 0),
    Math.max(raw.column, 0),
    Math.max(raw.end_line, 0),
    Math.max(raw.end_column, 0)
  );
  const message = raw.approximate
    ? `${raw.message}\n(position approximate)`
    : raw.message;

  const diagnostic = new vscode.Diagnostic(
    range,
    message,
    vscode.DiagnosticSeverity.Error
  );
  diagnostic.source = `ipu-as (${raw.stage})`;
  return diagnostic;
}

async function refresh(document) {
  if (!document || document.languageId !== LANGUAGE_ID) return;

  const result = await runCheck(document);

  // A run we cancelled ourselves says nothing about the toolchain or the code.
  if (result.superseded) return;

  if (result.error) {
    // A broken toolchain must not paint the file red — that would be a
    // diagnostic about the environment masquerading as one about the code. But
    // it must not be silent either: "checker unavailable" and "your code is
    // fine" looked identical before, which is worse than either.
    console.warn(`[ipu-asm] ${result.error}`);
    lastError = result.error;
    markUnavailable(result.error);
    diagnostics.delete(document.uri);
    return;
  }

  lastError = null;
  markAvailable();
  diagnostics.set(document.uri, result.found.map(toVscodeDiagnostic));
}

function scheduleRefresh(document) {
  if (!document || document.languageId !== LANGUAGE_ID) return;
  const key = document.uri.toString();
  clearTimeout(pending.get(key));
  pending.set(
    key,
    setTimeout(() => {
      pending.delete(key);
      refresh(document);
    }, DEBOUNCE_MS)
  );
}

function activate(context) {
  loadHoverData();

  // Start building the checker now rather than on the first keystroke: the
  // build is slow exactly once, and nothing waits on it.
  for (const folder of vscode.workspace.workspaceFolders || []) {
    ensureChecker(folder.uri.fsPath);
  }

  diagnostics = vscode.languages.createDiagnosticCollection(LANGUAGE_ID);
  statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  context.subscriptions.push(
    diagnostics,
    statusItem,
    vscode.languages.registerHoverProvider(LANGUAGE_ID, hoverProvider)
  );

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(refresh),
    vscode.workspace.onDidSaveTextDocument(refresh),
    vscode.workspace.onDidChangeTextDocument((e) => scheduleRefresh(e.document)),
    vscode.workspace.onDidCloseTextDocument((d) => {
      clearTimeout(pending.get(d.uri.toString()));
      pending.delete(d.uri.toString());
      diagnostics.delete(d.uri);
    }),
    vscode.commands.registerCommand('ipuAsm.check', () => {
      const editor = vscode.window.activeTextEditor;
      if (editor) refresh(editor.document);
    }),
    vscode.commands.registerCommand('ipuAsm.showCheckerStatus', () => {
      vscode.window.showInformationMessage(
        lastError
          ? `IPU Assembly diagnostics are not running.\n\n${lastError}`
          : 'IPU Assembly diagnostics are running normally.',
        { modal: true }
      );
    })
  );

  vscode.workspace.textDocuments.forEach(refresh);
}

function deactivate() {
  for (const timer of pending.values()) clearTimeout(timer);
  pending.clear();
  for (const child of running.values()) {
    child.superseded = true;
    child.kill();
  }
}

module.exports = { activate, deactivate };
