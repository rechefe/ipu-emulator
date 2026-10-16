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
const trust = require('./trust');

const LANGUAGE_ID = 'ipu-asm';

/** Debounce so a burst of keystrokes runs the checker once. */
const DEBOUNCE_MS = 400;

/** Schemes whose documents are files on disk at `uri.fsPath`: a git diff view's `git:`
 *  side names the working-tree file it shows a version of. */
const FILE_SCHEMES = new Set(['file', 'git']);

/** One checkout's checking state. Each checkout has its own toolchain, so one checkout's
 *  broken checker says nothing about another's. Files in no checkout (checked with
 *  ipuAsm.checkCommand) share one. */
class CheckState {
  constructor() {
    /** Debounce timers, keyed by document: a single shared timer let an edit in
     *  any other file cancel a pending .asm check. */
    this.pending = new Map();
    /** The document version each document was last checked at: an unchanged text
     *  (a save, or undo back to it) is not checked again. A check that failed is not
     *  recorded, so the same text is checked again once the toolchain is back. */
    this.checked = new Map();
    /** The checker process per run key: a newer run with the same key stops it. */
    this.running = new Map();
    /** Why the last check failed, or null when it succeeded (or none has finished). */
    this.lastError = null;
    /** Whether the last check succeeded. */
    this.ok = false;
    /** The failure pop-up is shown once per checkout: on every keystroke it would be
     *  worse than the silence it replaces. */
    this.warnedOnce = false;
    /** The resolved checker binary: a path once the build has finished, null when this
     *  checkout has no Bazel or the build failed. Undefined means resolution has not
     *  finished, which is not the same as failed -- both take the Bazel path, but only
     *  the second stops trying. */
    this.binary = undefined;
    this.resolving = false;
    /** Source stamp the checker was last built against. See checker.isFresh: Bazel
     *  rebuilds on content, so a re-dated but unchanged source leaves the binary's mtime
     *  behind forever, and this is what keeps that from pinning every check to the slow
     *  path. */
    this.stamp = 0;
  }
}

/** The status bar item, sent to the window only when what it shows changes: each
 *  assignment to it is a message to the window -- across the connection, in a remote one. */
class StatusBar {
  constructor() {
    this.item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    this.shown = '';
  }
  show(text, tooltip, command) {
    if (this.shown === text + tooltip) return;
    this.shown = text + tooltip;
    Object.assign(this.item, { text, tooltip, command }).show();
  }
  hide() {
    if (!this.shown) return;
    this.shown = '';
    this.item.hide();
  }
  dispose() {
    this.item.dispose();
  }
}

// ---------------------------------------------------------------------------
// Hover
// ---------------------------------------------------------------------------

/** language.js over isa.json (built with the extension, so always matching), or null. */
function loadLanguage() {
  try {
    return new Language(JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'isa.json'), 'utf8')));
  } catch (err) {
    // A convenience: losing it must not take highlighting with it.
    console.warn(`[ipu-asm] no language data (${err.message})`);
    return null;
  }
}

function describeSlot(lang, slot) {
  const meta = (lang.isa.slots || {})[slot];
  return meta && meta.hardware === false ? `${slot} (simulation-only)` : slot;
}

function renderInstruction(lang, mnemonic, forms) {
  const doc = (forms[0] && forms[0].doc) || {};
  const out = [`**${doc.title || mnemonic}**`, ''];

  // A mnemonic can occupy several slots -- NOP exists in all nine, each with its
  // own summary -- so list every form rather than silently showing the first.
  for (const form of forms) {
    const label = form.pseudo ? `pseudo → ${form.expandsTo}` : describeSlot(lang, form.slot);
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
function renderRegister(lang, name) {
  const doc = lang.isa.registerDocs[name] || {};
  const lines = [`**\`${name.toUpperCase()}\`** — ${doc.about || 'register'}`];
  if (doc.pair) lines.push(`\`${doc.pair.map((r) => r.toUpperCase()).join('`:`')}\` as one value.`);
  if (doc.fixed !== undefined) lines.push(`Fixed at ${doc.fixed} (read-only).`);
  return lines.join('\n\n');
}

function hoverProvider(lang) {
  // Hover's word: the assembler's own token shape (mnemonics have dots).
  const wordPattern = new RegExp(lang.isa.lexical.token);
  return {
    provideHover(document, position) {
      const range = document.getWordRangeAtPosition(position, wordPattern);
      if (!range) return undefined;
      const key = document.getText(range).toLowerCase();
      const instruction = lang.lookup(key);
      const text = instruction ? renderInstruction(lang, instruction.mnemonic, instruction.forms)
        : lang.registers.has(key) ? renderRegister(lang, key) : null;
      return text ? new vscode.Hover(new vscode.MarkdownString(text), range) : undefined;
    },
  };
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

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

/** The checker, run on documents as they change, with a CheckState per checkout. */
class Checking {
  constructor(kernelApi) {
    this.kernelApi = kernelApi;
    this.diagnostics = vscode.languages.createDiagnosticCollection(LANGUAGE_ID);
    this.status = new StatusBar();
    /** CheckState by checkout root ('' for files in none). */
    this.states = new Map();
  }

  stateOf(root) {
    const key = root || '';
    if (!this.states.has(key)) this.states.set(key, new CheckState());
    return this.states.get(key);
  }

  /** The checkout holding a document's file. An unsaved one has no file: it belongs to the
   *  window's checkout if there is exactly one, else to none. Other schemes have a
   *  checkout only when they name a file on disk. */
  checkoutFor(uri) {
    if (FILE_SCHEMES.has(uri.scheme)) return project.checkoutOf(uri.fsPath);
    if (uri.scheme !== 'untitled') return null;
    const roots = [...this.kernelApi.indexes.keys()];
    return roots.length === 1 ? roots[0] : null;
  }

  /** Build the checker once and remember where Bazel put it.
   *
   *  Never awaited by a check: a cold Bazel server takes ten seconds or more to
   *  answer, and a check that waited for it would be slower than the path it is
   *  replacing. Checks keep using `bazel run` until this lands. */
  async resolveChecker(root) {
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
    const state = this.stateOf(root);
    state.stamp = Math.max(state.stamp, stamp);
    return binary;
  }

  ensureChecker(root) {
    if (!root || !trust.mayRun()) return;
    const state = this.stateOf(root);
    if (state.binary !== undefined || state.resolving) return;
    state.resolving = true;
    this.resolveChecker(root).then(
      (binary) => {
        state.resolving = false;
        if (binary === undefined) return; // timed out: the next check tries again
        state.binary = binary;
        if (!binary) console.warn('[ipu-asm] no prebuilt checker; using `bazel run` per check');
      },
      (err) => {
        state.resolving = false;
        state.binary = null;
        console.warn(`[ipu-asm] could not resolve the checker (${err.message})`);
      }
    );
  }

  /** ipuAsm.checkCommand if set, else checkout `root`'s checker; null with neither, since
   *  outside an IPU checkout a .asm is some other assembly and nothing is run for it. */
  checkCommand(folder, root) {
    const configured = kernels.setting(folder, 'checkCommand', []);
    if (configured.length) return configured;

    if (!root) return null;
    this.ensureChecker(root);
    const state = this.stateOf(root);
    return checker.checkCommandFor(state.binary || null, root, state.stamp);
  }

  /** Check `text` as the file at `uri`. A newer run with the same `key`, or cancelling
   *  `token`, supersedes it; one past ipuAsm.timeoutSeconds is stopped as an error. */
  runCheckText(uri, text, key, token) {
    return this.checkIn(this.checkoutFor(uri), uri, text, key, token);
  }

  /** runCheckText, in checkout `root` (null: none). */
  async checkIn(root, uri, text, key, token) {
    // In Restricted Mode the folder's settings and BUILD files would choose what runs.
    if (!trust.mayRun()) return { restricted: true, error: trust.RESTRICTED };
    const folder = vscode.workspace.getWorkspaceFolder(uri);
    // A document with neither a checkout nor a file has nowhere to be checked from.
    if (!root && !folder && !FILE_SCHEMES.has(uri.scheme)) return { outside: true };
    const cwd = root || (folder ? folder.uri.fsPath : path.dirname(uri.fsPath));
    const argv = this.checkCommand(folder, root);
    if (!argv) return { outside: true };
    if (token && token.isCancellationRequested) return { superseded: true };
    const [command, ...args] = argv;
    const state = this.stateOf(root);
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
        if (state.running.has(key)) stop(state.running.get(key));
        state.running.set(key, c);
      },
    });
    if (cancelled) cancelled.dispose();
    if (child && state.running.get(key) === child) state.running.delete(key);
    if (child && child.superseded) return { superseded: true };
    if (r.spawnError) {
      // A prebuilt checker that will not start is a stale resolution, not a
      // broken toolchain: `bazel clean` deletes the binary out from under us.
      // Forgetting it sends the next check down the Bazel path, which rebuilds.
      if (state.binary === command) state.binary = undefined;
      return { error: `${command}: ${r.stderr}`, command };
    }
    if (r.timedOut) return { error: r.stderr, command };
    // Exit code 1 just means "found problems", so it is not an error here.
    try {
      const found = JSON.parse(r.stdout);
      if (rebuilding) {
        state.stamp = Math.max(state.stamp, stamp);
        this.ensureChecker(root);
      }
      return { found };
    } catch {
      return { error: checker.explainFailure(r.stderr || r.stdout, command), command };
    }
  }

  /** Check a document's current buffer (what is on screen, not on disk), unless
   *  this very version was checked already. */
  async refresh(document, force = false) {
    if (!document || document.languageId !== LANGUAGE_ID) return;
    if (!trust.mayRun()) {
      this.showStatus();
      return;
    }
    const key = document.uri.toString();
    const version = document.version;
    const root = this.checkoutFor(document.uri);
    const state = this.stateOf(root);
    if (!force && state.checked.get(key) === version) return;

    const result = await this.checkIn(root, document.uri, document.getText(), key);

    // A run we cancelled ourselves says nothing about the toolchain or the code,
    // and a file closed while it ran has nothing left to mark.
    if (result.superseded || document.isClosed) return;

    if (result.error) {
      // A broken toolchain must not paint the file red — that would be a
      // diagnostic about the environment masquerading as one about the code. But
      // it must not be silent either: "checker unavailable" and "your code is
      // fine" looked identical before, which is worse than either.
      console.warn(`[ipu-asm] ${result.error}`);
      Object.assign(state, { lastError: result.error, ok: false });
      this.diagnostics.delete(document.uri);
      this.warnOnce(state);
      this.showStatus();
      return;
    }
    state.checked.set(key, version);

    // Not an IPU checkout's file (another project's assembly): nothing to say.
    if (result.outside) {
      this.diagnostics.delete(document.uri);
    } else {
      Object.assign(state, { lastError: null, ok: true });
      this.diagnostics.set(document.uri, result.found.map(toVscodeDiagnostic));
    }
    this.showStatus();
  }

  scheduleRefresh(document) {
    if (!document || document.languageId !== LANGUAGE_ID) return;
    const key = document.uri.toString();
    const { pending } = this.stateOf(this.checkoutFor(document.uri));
    clearTimeout(pending.get(key));
    pending.set(
      key,
      setTimeout(() => {
        pending.delete(key);
        this.refresh(document);
      }, DEBOUNCE_MS)
    );
  }

  /** Forget a closed document: its pending check, its checked version, its diagnostics. */
  forget(document) {
    const key = document.uri.toString();
    for (const state of this.states.values()) {
      clearTimeout(state.pending.get(key));
      state.pending.delete(key);
      state.checked.delete(key);
    }
    this.diagnostics.delete(document.uri);
  }

  /** The status of the active editor's checkout (unchanged when the editor holds no IPU
   *  assembly). */
  showStatus() {
    if (!trust.mayRun()) {
      this.status.show('$(shield) IPU: Restricted Mode', `IPU Assembly: ${trust.RESTRICTED}`, 'workbench.trust.manage');
      return;
    }
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== LANGUAGE_ID) return;
    const state = this.states.get(this.checkoutFor(editor.document.uri) || '');
    if (state && state.lastError) {
      this.status.show('$(warning) IPU: checks off', `IPU Assembly diagnostics are not running:\n\n${state.lastError}`, 'ipuAsm.showCheckerStatus');
    } else if (state && state.ok) {
      this.status.show('$(check) IPU', 'IPU Assembly diagnostics are running.', undefined);
    } else {
      // Nothing checked in this checkout yet, or a file in none.
      this.status.hide();
    }
  }

  warnOnce(state) {
    if (state.warnedOnce) return;
    state.warnedOnce = true;
    vscode.window
      .showWarningMessage(`IPU Assembly: diagnostics are off — ${state.lastError}`, 'Details')
      .then((choice) => {
        if (choice === 'Details') vscode.commands.executeCommand('ipuAsm.showCheckerStatus');
      });
  }

  /** Why diagnostics are off for `uri`'s checkout, or for the active editor's; without
   *  either, the first checkout's whose are. Null when they are running. */
  errorFor(uri) {
    const editor = vscode.window.activeTextEditor;
    const at = uri || (editor && editor.document.uri);
    if (at) return (this.states.get(this.checkoutFor(at) || '') || {}).lastError || null;
    const failing = [...this.states.values()].find((s) => s.lastError);
    return failing ? failing.lastError : null;
  }

  /** Build the checkers now, not on the first keystroke (slow exactly once, and nothing
   *  waits on it), and check what is open. */
  start() {
    for (const root of this.kernelApi.indexes.keys()) this.ensureChecker(root);
    vscode.workspace.textDocuments.forEach((d) => this.refresh(d));
  }

  dispose() {
    for (const state of this.states.values()) {
      for (const timer of state.pending.values()) clearTimeout(timer);
      state.pending.clear();
      for (const child of state.running.values()) {
        child.superseded = true;
        child.stop();
      }
    }
    this.diagnostics.dispose();
    this.status.dispose();
  }
}

function activate(context) {
  const lang = loadLanguage();
  const kernelApi = kernels.register(context);
  const checking = new Checking(kernelApi);
  context.subscriptions.push(checking);

  checking.showStatus();
  // After activation returns, since starting a process blocks briefly.
  setImmediate(() => checking.start());

  let editingApi = null;
  if (lang) {
    /** Numbers rename checks, so concurrent ones never supersede each other. */
    let renameChecks = 0;
    context.subscriptions.push(vscode.languages.registerHoverProvider(LANGUAGE_ID, hoverProvider(lang)));
    editingApi = editing.register(context, {
      lang,
      renderInstruction: (mnemonic, forms) => renderInstruction(lang, mnemonic, forms),
      renderRegister: (name) => renderRegister(lang, name),
      checkText: (uri, text, token) => checking.runCheckText(uri, text, `${uri.toString()}#rename-${++renameChecks}`, token),
    });
  }

  context.subscriptions.push(
    // Out of Restricted Mode: what it kept from running starts now.
    vscode.workspace.onDidGrantWorkspaceTrust(() => {
      checking.status.hide();
      checking.start();
    }),
    vscode.workspace.onDidOpenTextDocument((d) => checking.refresh(d)),
    vscode.workspace.onDidSaveTextDocument((d) => checking.refresh(d)),
    // No content change (the dirty flag flipped): nothing new to check.
    vscode.workspace.onDidChangeTextDocument((e) => e.contentChanges.length && checking.scheduleRefresh(e.document)),
    vscode.workspace.onDidCloseTextDocument((d) => checking.forget(d)),
    vscode.window.onDidChangeActiveTextEditor(() => checking.showStatus()),
    vscode.commands.registerCommand('ipuAsm.check', () => {
      const editor = vscode.window.activeTextEditor;
      if (editor) checking.refresh(editor.document, true);
    }),
    vscode.commands.registerCommand('ipuAsm.showCheckerStatus', () => {
      const error = checking.errorFor();
      vscode.window.showInformationMessage(
        error
          ? `IPU Assembly diagnostics are not running.\n\n${error}`
          : 'IPU Assembly diagnostics are running normally.',
        { modal: true }
      );
    })
  );
  // For tests: the kernels and editing APIs, and why diagnostics are off (for a document,
  // the active editor's, or any), if they are.
  return { kernels: kernelApi, editing: editingApi, checkerError: (uri) => checking.errorFor(uri) };
}

module.exports = { activate };
