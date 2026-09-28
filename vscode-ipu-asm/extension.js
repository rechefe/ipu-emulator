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

const fs = require('fs');
const path = require('path');
const vscode = require('vscode');

const checker = require('./checker');
const project = require('./project');
const editing = require('./editing');
const kernels = require('./kernels');
const { timeoutSeconds } = kernels;
const { Language } = require('./language');

const LANGUAGE_ID = 'ipu-asm';

/** Debounce so a burst of keystrokes runs the checker once. */
const DEBOUNCE_MS = 400;

let diagnostics;
let statusItem;
/** Debounce timers, keyed by document: a single shared timer let an edit in
 *  any other file cancel a pending .asm check. */
const pending = new Map();
/** The document version each document was last checked at: an unchanged text
 *  (a save, or undo back to it) is not checked again. */
const checked = new Map();
let warnedOnce = false;
let lastError = null;
const running = new Map();

/** What the status item shows ('' when hidden). Each assignment to it is a message to the
 *  window -- across the connection, in a remote one -- so an unchanged state sends none. */
let status = '';
function showStatus(text, tooltip, command) {
  if (status === text + tooltip) return;
  status = text + tooltip;
  Object.assign(statusItem, { text, tooltip, command }).show();
}

function markUnavailable(reason) {
  showStatus('$(warning) IPU: checks off', `IPU Assembly diagnostics are not running:\n\n${reason}`, 'ipuAsm.showCheckerStatus');

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
  showStatus('$(check) IPU', 'IPU Assembly diagnostics are running.', undefined);
}

// ---------------------------------------------------------------------------
// Hover
// ---------------------------------------------------------------------------

/** language.js over isa.json (built with the extension, so always matching), or null. */
let lang = null;
/** Hover's word: the assembler's own token shape (mnemonics have dots). */
let wordPattern = null;
/** Numbers rename checks, so concurrent ones never supersede each other. */
let renameChecks = 0;
let kernelApi = null;

function loadHoverData() {
  try {
    lang = new Language(JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'isa.json'), 'utf8')));
    wordPattern = new RegExp(lang.isa.lexical.token);
  } catch (err) {
    // A convenience: losing it must not take highlighting with it.
    console.warn(`[ipu-asm] no language data (${err.message})`);
  }
}

function describeSlot(slot) {
  const meta = (lang.isa.slots || {})[slot];
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

/** A register's hover, from the simulator's definitions (isa.json `registerDocs`). */
function renderRegister(name) {
  const doc = lang.isa.registerDocs[name] || {};
  const lines = [`**\`${name.toUpperCase()}\`** — ${doc.about || 'register'}`];
  if (doc.pair) lines.push(`\`${doc.pair.map((r) => r.toUpperCase()).join('`:`')}\` as one value.`);
  if (doc.fixed !== undefined) lines.push(`Fixed at ${doc.fixed} (read-only).`);
  return lines.join('\n\n');
}

const hoverProvider = {
  provideHover(document, position) {
    const range = document.getWordRangeAtPosition(position, wordPattern);
    if (!range) return undefined;
    const key = document.getText(range).toLowerCase();
    const instruction = lang.lookup(key);
    const text = instruction ? renderInstruction(instruction.mnemonic, instruction.forms)
      : lang.registers.has(key) ? renderRegister(key) : null;
    return text ? new vscode.Hover(new vscode.MarkdownString(text), range) : undefined;
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

/** Build the checker once and remember where Bazel put it.
 *
 *  Never awaited by a check: a cold Bazel server takes ten seconds or more to
 *  answer, and a check that waited for it would be slower than the path it is
 *  replacing. Checks keep using `bazel run` until this lands. */
async function resolveChecker(root) {
  // Sampled before the build, so an edit made while it runs invalidates the
  // result rather than being swallowed by it.
  const stamp = checker.newestSourceMtime(root);
  const seconds = timeoutSeconds(vscode.workspace.getWorkspaceFolder(vscode.Uri.file(root)));
  const build = await checker.run('bazel', ['build', ...checker.QUIET, checker.CHECKER_TARGET], root, seconds);
  // A build that timed out (behind another command's lock) says nothing about
  // the checker: undefined leaves it to be tried again, null gives up.
  if (build.code !== 0) return build.timedOut ? undefined : null;

  // bazel-bin is a convenience symlink: absent after a clean, renamed by
  // --symlink_prefix, elsewhere under a different --config. Ask instead.
  const info = await checker.run('bazel', ['info', 'bazel-bin'], root, seconds);
  if (info.code !== 0) return info.timedOut ? undefined : null;

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
      if (binary === undefined) return; // timed out: the next check tries again
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

/** The checkout holding a document's file; for an unsaved one, the first open. */
function checkoutFor(uri) {
  if (uri.scheme === 'file') return project.checkoutOf(uri.fsPath);
  const [first] = kernelApi.indexes.keys();
  return first || null;
}

/** ipuAsm.checkCommand if set, else checkout `root`'s checker; null with neither, since
 *  outside an IPU checkout a .asm is some other assembly and nothing is run for it. */
function checkCommand(folder, root) {
  const configured = kernels.setting(folder, 'checkCommand', []);
  if (configured.length) return configured;

  if (!root) return null;
  ensureChecker(root);
  return checker.checkCommandFor(
    checkerBinaries.get(root) || null,
    root,
    checkerStamps.get(root) || 0
  );
}

/** Check `text` as the file at `uri`. A newer run with the same `key`, or cancelling
 *  `token`, supersedes it; one past ipuAsm.timeoutSeconds is stopped as an error. */
async function runCheckText(uri, text, key, token) {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  const root = checkoutFor(uri);
  const cwd = root || (folder ? folder.uri.fsPath : path.dirname(uri.fsPath));
  const argv = checkCommand(folder, root);
  if (!argv) return { outside: true };
  if (token && token.isCancellationRequested) return { superseded: true };
  const [command, ...args] = argv;
  // The fallback rebuilds on its way to checking, so a run that comes back with
  // diagnostics has also brought the binary up to date with these sources.
  // Sampled before the run, so an edit made during it is not credited to it.
  const rebuilding = Boolean(root) && checker.isBazelCommand(argv);
  const stamp = rebuilding ? checker.newestSourceMtime(cwd) : 0;

  // A superseded or cancelled run closes with no output, which must not read as the
  // checker failing: that would clear diagnostics and flip the status bar on a keystroke.
  let child = null;
  const stop = (c) => { c.superseded = true; c.stop(); };
  const cancelled = token && token.onCancellationRequested(() => child && stop(child));
  const r = await checker.run(command, [...args, '-'], cwd, timeoutSeconds(folder), {
    input: text,
    onSpawn: (c) => {
      child = c;
      if (running.has(key)) stop(running.get(key));
      running.set(key, c);
    },
  });
  if (cancelled) cancelled.dispose();
  if (child && running.get(key) === child) running.delete(key);
  if (child && child.superseded) return { superseded: true };
  if (r.spawnError) {
    // A prebuilt checker that will not start is a stale resolution, not a
    // broken toolchain: `bazel clean` deletes the binary out from under us.
    // Forgetting it sends the next check down the Bazel path, which rebuilds.
    forgetChecker(command);
    return { error: `${command}: ${r.stderr}`, command };
  }
  if (r.timedOut) return { error: r.stderr, command };
  // Exit code 1 just means "found problems", so it is not an error here.
  try {
    const found = JSON.parse(r.stdout);
    if (rebuilding) {
      checkerStamps.set(cwd, Math.max(checkerStamps.get(cwd) || 0, stamp));
      ensureChecker(cwd);
    }
    return { found };
  } catch {
    return { error: checker.explainFailure(r.stderr || r.stdout, command), command };
  }
}

function toVscodeDiagnostic(raw) {
  const at = (n) => Math.max(n, 0);
  const message = raw.approximate ? `${raw.message}\n(position approximate)` : raw.message;
  const diagnostic = new vscode.Diagnostic(
    new vscode.Range(at(raw.line), at(raw.column), at(raw.end_line), at(raw.end_column)),
    message,
    vscode.DiagnosticSeverity.Error
  );
  diagnostic.source = `ipu-as (${raw.stage})`;
  return diagnostic;
}

/** Check a document's current buffer (what is on screen, not on disk), unless
 *  this very version was checked already. */
async function refresh(document, force = false) {
  if (!document || document.languageId !== LANGUAGE_ID) return;
  const key = document.uri.toString();
  const version = document.version;
  if (!force && checked.get(key) === version) return;

  const result = await runCheckText(document.uri, document.getText(), key);

  // A run we cancelled ourselves says nothing about the toolchain or the code,
  // and a file closed while it ran has nothing left to mark.
  if (result.superseded || document.isClosed) return;
  checked.set(key, version);

  // Not an IPU checkout's file (another project's assembly): nothing to say.
  if (result.outside) {
    diagnostics.delete(document.uri);
    statusItem.hide();
    status = '';
    return;
  }

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

  diagnostics = vscode.languages.createDiagnosticCollection(LANGUAGE_ID);
  statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  context.subscriptions.push(diagnostics, statusItem);

  kernelApi = kernels.register(context);
  // Build the checker now, not on the first keystroke (slow exactly once, and nothing waits
  // on it); after activation returns, since starting a process blocks briefly.
  setImmediate(() => {
    for (const root of kernelApi.indexes.keys()) ensureChecker(root);
    vscode.workspace.textDocuments.forEach((d) => refresh(d));
  });

  let editingApi = null;
  if (lang) {
    context.subscriptions.push(vscode.languages.registerHoverProvider(LANGUAGE_ID, hoverProvider));
    editingApi = editing.register(context, {
      lang,
      renderInstruction,
      renderRegister,
      checkText: (uri, text, token) => runCheckText(uri, text, `${uri.toString()}#rename-${++renameChecks}`, token),
    });
  }

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument(refresh),
    vscode.workspace.onDidSaveTextDocument(refresh),
    // No content change (the dirty flag flipped): nothing new to check.
    vscode.workspace.onDidChangeTextDocument((e) => e.contentChanges.length && scheduleRefresh(e.document)),
    vscode.workspace.onDidCloseTextDocument((d) => {
      clearTimeout(pending.get(d.uri.toString()));
      pending.delete(d.uri.toString());
      checked.delete(d.uri.toString());
      diagnostics.delete(d.uri);
    }),
    vscode.commands.registerCommand('ipuAsm.check', () => {
      const editor = vscode.window.activeTextEditor;
      if (editor) refresh(editor.document, true);
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
  // For tests: the kernels and editing APIs, and why diagnostics are off, if they are.
  return { kernels: kernelApi, editing: editingApi, checkerError: () => lastError };
}

function deactivate() {
  for (const timer of pending.values()) clearTimeout(timer);
  pending.clear();
  for (const child of running.values()) {
    child.superseded = true;
    child.stop();
  }
}

module.exports = { activate, deactivate };
