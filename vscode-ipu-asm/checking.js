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
//
// A checker that speaks version 2 of that output answers the editing providers
// too, from the same run (analysis.js): one check per document version.

const path = require('path');
const vscode = require('vscode');

const { AnalysisClient, Positions, parseCheck } = require('./analysis');
const { CheckoutFormatter } = require('./formatting');
const { V2Semantics } = require('./analysis-semantics');
const checker = require('./checker');
const { fallbackOf } = require('./fallback');
const project = require('./project');
const { LANGUAGE_ID } = require('./providers/shared');
const { featureOn, setting, timeoutSeconds } = require('./settings');
const trust = require('./trust');

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
    /** The checker command the binary's freshness picked (checker.checkCommandFor), until a
     *  source, the binary or the stamp changes: a check does not walk the sources each time. */
    this.command = undefined;
    this.generation = 0;
    /** File-system watchers on the checker's sources (Checking.watchSources). */
    this.watchers = [];
  }

  /** Set what the checker is picked from (`binary`, `stamp`), so it is picked again. */
  update(fields) {
    Object.assign(this, fields);
    this.invalidate();
  }

  /** Forget the picked command: something it was picked from has changed. */
  invalidate() {
    this.command = undefined;
    this.generation++;
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

/** A word that does not fit its slots and no diagnostic covers: information, not an error,
 *  since the error is the assembler's to report (and it reports the first). */
function slotFitDiagnostic(word) {
  const r = word.range;
  const diagnostic = new vscode.Diagnostic(new vscode.Range(r.line, r.character, r.endLine, r.endCharacter),
    `${word.error} (this word: ${word.mnemonics.join('; ')})`, vscode.DiagnosticSeverity.Information);
  diagnostic.source = 'ipu-as (slot fit)';
  return diagnostic;
}

/** Where the checker runs for a document in checkout `root` (null: none): { folder, cwd }, or
 *  null for a document with neither a checkout nor a file, which has nowhere to run from. */
function placeOf(root, uri) {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (!root && !folder && !FILE_SCHEMES.has(uri.scheme)) return null;
  return { folder, cwd: root || (folder ? folder.uri.fsPath : path.dirname(uri.fsPath)) };
}

/** The checker, run on documents as they change, with a CheckState per checkout. */
class Checking {
  constructor(kernelApi) {
    this.kernelApi = kernelApi;
    this.diagnostics = vscode.languages.createDiagnosticCollection(LANGUAGE_ID);
    this.status = new StatusBar();
    /** CheckState by checkout root ('' for files in none). */
    this.states = new Map();
    /** The checkout each document was last found in (null: none), by document Uri, for what
     *  must answer at once (the status bar, a keystroke's debounce). */
    this.roots = new Map();
    /** The latest refresh of each document, by document Uri: what one that was superseded
     *  resolves to, so a refresh resolves once the check that counts is done. */
    this.latest = new Map();
    /** Which contract version each checker speaks, and each document's latest check. */
    this.analysis = new AnalysisClient();
    /** Which checkers have `format`, and running it. */
    this.formatter = new CheckoutFormatter();
    /** Each document's language data (languages.js), once the bundled data is read. */
    this.languages = null;
  }

  stateOf(root) {
    const key = root || '';
    if (!this.states.has(key)) {
      const state = new CheckState();
      if (root) state.watchers = this.watchSources(root, state);
      this.states.set(key, state);
    }
    return this.states.get(key);
  }

  /** Watch checkout `root`'s checker sources: a change makes `state` pick its checker again.
   *  Relative to the checkout, so a checkout opened by a subfolder, or not opened at all
   *  (a file from it), is watched as well. */
  watchSources(root, state) {
    const glob = `**/*{${checker.SOURCE_EXTENSIONS.join(',')}}`;
    return checker.SOURCE_DIRS.map((dir) => {
      const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(vscode.Uri.file(path.join(root, dir)), glob));
      // Changed sources make a checker that may speak other contract versions: ask it again.
      const changed = () => {
        state.invalidate();
        this.analysis.forgetVersions(root);
        this.formatter.forget(root);
        if (this.languages) this.languages.invalidate(root);
      };
      for (const event of [watcher.onDidCreate, watcher.onDidChange, watcher.onDidDelete]) event(changed);
      return watcher;
    });
  }

  /** The checkout holding a document's file. An unsaved one has no file: it belongs to the
   *  window's checkout if there is exactly one, else to none. Other schemes have a
   *  checkout only when they name a file on disk. */
  async checkoutFor(uri) {
    const root = FILE_SCHEMES.has(uri.scheme) ? await project.checkoutOf(uri.fsPath) : this.knownCheckout(uri);
    this.roots.set(uri.toString(), root);
    return root;
  }

  /** checkoutFor, as last found: null for a document not looked at yet. */
  knownCheckout(uri) {
    if (uri.scheme === 'untitled') {
      const roots = [...this.kernelApi.indexes.keys()];
      return roots.length === 1 ? roots[0] : null;
    }
    return this.roots.get(uri.toString()) || null;
  }

  /** Build the checker once and remember where Bazel put it.
   *
   *  Never awaited by a check: a cold Bazel server takes ten seconds or more to
   *  answer, and a check that waited for it would be slower than the path it is
   *  replacing. Checks keep using `bazel run` until this lands. */
  async resolveChecker(root) {
    // Sampled before the build, so an edit made while it runs invalidates the
    // result rather than being swallowed by it.
    const stamp = await checker.newestSourceMtime(root);
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
    if (!(await project.exists(binary))) return null;
    const state = this.stateOf(root);
    state.update({ stamp: Math.max(state.stamp, stamp) });
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
        state.update({ binary });
        if (!binary) console.warn('[ipu-asm] no prebuilt checker; using `bazel run` per check');
      },
      (err) => {
        state.resolving = false;
        state.update({ binary: null });
        console.warn(`[ipu-asm] could not resolve the checker (${err.message})`);
      }
    );
  }

  /** ipuAsm.checkCommand if set, else checkout `root`'s checker; null with neither, since
   *  outside an IPU checkout a .asm is some other assembly and nothing is run for it. */
  async checkCommand(folder, root) {
    const configured = setting(folder, 'checkCommand', []);
    if (configured.length) return configured;

    if (!root) return null;
    this.ensureChecker(root);
    const state = this.stateOf(root);
    if (state.command) return state.command;
    const { generation } = state;
    const command = await checker.checkCommandFor(state.binary || null, root, state.stamp);
    // Kept only if nothing it was picked from changed while it was being picked.
    if (state.generation === generation) state.command = command;
    return command;
  }

  /** Check `text` as the file at `uri`. A newer run with the same `key`, or cancelling
   *  `token`, supersedes it; one past ipuAsm.timeoutSeconds is stopped as an error. */
  async runCheckText(uri, text, key, token) {
    return this.checkIn(await this.checkoutFor(uri), uri, text, key, token);
  }

  /** runCheckText, in checkout `root` (null: none): diagnostics only. */
  checkIn(root, uri, text, key, token) {
    return this.runIn(root, uri, text, key, token, async (argv) => argv);
  }

  /** checkIn for a document's own text: a checker that speaks version 2 of the contract is
   *  asked for every analysis too (`answer` in the result). */
  analyseIn(root, uri, text, key) {
    return this.runIn(root, uri, text, key, undefined, async (argv, cwd, seconds) => (await this.analysis.plan(argv, cwd, seconds)).argv);
  }

  /** A document's lines formatted by its checkout's assembler (formatting.js
   *  CheckoutFormatter.format): null where it cannot be asked, for the editor's rules. */
  async formatWithChecker(document, options, token) {
    if (!trust.mayRun()) return null;
    const { uri } = document;
    const root = await this.checkoutFor(uri);
    const place = placeOf(root, uri);
    const argv = place && await this.checkCommand(place.folder, root);
    if (!argv) return null;
    return this.formatter.format(argv, place.cwd, timeoutSeconds(place.folder), document.getText(), options, token);
  }

  /** Run the checker on `text`; `planFor(argv, cwd, seconds)` gives the argv to run it as. */
  async runIn(root, uri, text, key, token, planFor) {
    // In Restricted Mode the folder's settings and BUILD files would choose what runs.
    if (!trust.mayRun()) return { error: trust.RESTRICTED };
    const place = placeOf(root, uri);
    if (!place) return { outside: true };
    const { folder, cwd } = place;
    const argv = await this.checkCommand(folder, root);
    if (!argv) return { outside: true };
    if (token && token.isCancellationRequested) return { superseded: true };
    const [command, ...args] = await planFor(argv, cwd, timeoutSeconds(folder));
    const state = this.stateOf(root);
    // The fallback rebuilds on its way to checking, so a run that comes back with
    // diagnostics has also brought the binary up to date with these sources.
    // Sampled before the run, so an edit made during it is not credited to it.
    const rebuilding = Boolean(root) && checker.isBazelCommand(argv);
    const stamp = rebuilding ? await checker.newestSourceMtime(cwd) : 0;

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
      if (state.binary === command) state.update({ binary: undefined });
      return { error: `${command}: ${r.stderr}`, command };
    }
    if (r.timedOut) return { error: r.stderr, command };
    // Exit code 1 just means "found problems", so it is not an error here.
    try {
      const { found, answer } = parseCheck(r.stdout, text);
      if (rebuilding) {
        state.update({ stamp: Math.max(state.stamp, stamp) });
        this.ensureChecker(root);
      }
      return { found, answer };
    } catch {
      return { error: checker.explainFailure(r.stderr || r.stdout, command), command };
    }
  }

  /** Check a document's current buffer (what is on screen, not on disk), unless
   *  this very version was checked already. Resolves once that check is done, or
   *  the one that superseded it. */
  refresh(document, force = false) {
    const version = document ? document.version : undefined;
    // However it ends, whoever waits for this version's analysis has its answer then.
    // (Not for a document closed since: forget() has settled it, and must not be undone.)
    const done = this.refreshOnce(document, force)
      .finally(() => document && !document.isClosed && this.analysis.release(document.uri.toString(), version));
    if (document) this.latest.set(document.uri.toString(), done);
    return done;
  }

  async refreshOnce(document, force) {
    if (!document || document.languageId !== LANGUAGE_ID) return;
    if (!trust.mayRun()) {
      this.showStatus();
      return;
    }
    if (!featureOn(document.uri, 'diagnostics')) {
      this.switchedOff(document);
      this.showStatus();
      return;
    }
    const key = document.uri.toString();
    const version = document.version;
    const root = await this.checkoutFor(document.uri);
    const state = this.stateOf(root);
    if (!force && state.checked.get(key) === version) return;

    const text = document.getText();
    const result = await this.analyseIn(root, document.uri, text, key);

    // A run we cancelled ourselves says nothing about the toolchain or the code,
    // and a file closed, or its diagnostics switched off, while it ran has nothing left to mark.
    if (document.isClosed || !featureOn(document.uri, 'diagnostics')) return;
    if (result.superseded) {
      await this.latest.get(key);
      return;
    }

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
      this.analysis.store(key, version, text, result.found, result.answer || null);
      // The contract counts columns in code points, VS Code in UTF-16 units.
      const positions = result.answer ? result.answer.positions : new Positions(text);
      const shown = result.found.map((d) => toVscodeDiagnostic(positions.diagnostic(d)));
      // The assembler stops at a word's first misfit; the analysis fits every word.
      for (const w of result.answer ? result.answer.misfits() : []) shown.push(slotFitDiagnostic(w));
      this.diagnostics.set(document.uri, shown);
    }
    this.showStatus();
  }

  scheduleRefresh(document) {
    if (!document || document.languageId !== LANGUAGE_ID) return;
    const key = document.uri.toString();
    // Cleared wherever it was put: a document's checkout is not known until its first check.
    for (const state of this.states.values()) clearTimeout(state.pending.get(key));
    const { pending } = this.stateOf(this.knownCheckout(document.uri));
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
    this.roots.delete(key);
    this.latest.delete(key);
    this.analysis.forget(key);
    this.diagnostics.delete(document.uri);
  }

  /** Diagnostics switched off (ipuAsm.diagnostics) for a document: its checks are dropped, so
   *  the editing help reads the editor's own rules, and the next check after switching them
   *  back on runs afresh. */
  switchedOff(document) {
    for (const state of this.states.values()) state.checked.delete(document.uri.toString());
    this.analysis.forget(document.uri.toString());
    this.diagnostics.delete(document.uri);
  }

  /** The kept check of `text` as the document at `uri`, if its latest check was of that very
   *  text: a rename compares against it rather than checking the same text again. */
  keptCheck(uri, text) {
    const entry = this.analysis.cached(uri.toString(), text);
    return entry ? Promise.resolve({ found: entry.found }) : null;
  }

  /** What the providers know about `document` (see editing.js): the answer for its text now
   *  (waiting for this version's check, if one is due and the last one had an answer), or with
   *  `stale`, the latest answer at once; else the editor's own rules, as the first paint. */
  async semanticsOf(lang, document, { stale = false } = {}) {
    const key = document.uri.toString();
    const text = document.getText();
    const answer = stale ? this.analysis.latest(key) : await this.analysis.current(key, document.version, text, { first: false });
    return answer ? new V2Semantics(lang, answer, text) : fallbackOf(lang).semantics(text);
  }

  /** The status of the active editor's checkout (unchanged when the editor holds no IPU
   *  assembly). */
  showStatus() {
    if (!trust.mayRun()) {
      // Editing help, which still works, uses the bundled ISA: the checkout's is not generated.
      const suffix = this.languages ? ' (bundled ISA)' : '';
      this.status.show(`$(shield) IPU: Restricted Mode${suffix}`, `IPU Assembly: ${trust.RESTRICTED}`, 'workbench.trust.manage');
      return;
    }
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== LANGUAGE_ID) return;
    if (!featureOn(editor.document.uri, 'diagnostics')) {
      this.status.hide();
      return;
    }
    const state = this.states.get(this.knownCheckout(editor.document.uri) || '');
    // Hover, completion and the other editing help: the checkout's ISA, or the bundled one.
    const chosen = this.languages && this.languages.chosen(editor.document);
    const bundled = chosen && chosen.use === 'bundled';
    const suffix = bundled ? ' (bundled ISA)' : '';
    const isa = bundled ? `\n\nEditing help uses the ISA bundled with the extension: ${chosen.why}.` : '';
    if (state && state.lastError) {
      this.status.show(`$(warning) IPU: checks off${suffix}`, `IPU Assembly diagnostics are not running:\n\n${state.lastError}${isa}`, 'ipuAsm.showCheckerStatus');
    } else if (state && state.ok) {
      this.status.show(`$(check) IPU${suffix}`, `IPU Assembly diagnostics are running.${isa}`, undefined);
    } else if (bundled) {
      // Nothing checked in this checkout yet, or a file in none.
      this.status.show(`$(book) IPU${suffix}`, isa.trim(), undefined);
    } else {
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
    if (at) return (this.states.get(this.knownCheckout(at) || '') || {}).lastError || null;
    const failing = [...this.states.values()].find((s) => s.lastError);
    return failing ? failing.lastError : null;
  }

  /** Check what is open, and build the checkers now, not on the first keystroke (slow
   *  exactly once, and nothing waits on it). Once the open folders' checkouts are known, so
   *  an unsaved buffer is checked in the window's only one. */
  async start() {
    if (!trust.mayRun()) {
      this.showStatus();
      return;
    }
    this.status.hide(); // the Restricted Mode notice, when trust was just granted
    // A failed scan still leaves the open documents to check.
    await this.kernelApi.ready.catch((err) => console.warn(`[ipu-asm] could not find the checkouts (${err.message})`));
    for (const root of this.kernelApi.indexes.keys()) this.ensureChecker(root);
    vscode.workspace.textDocuments.forEach((d) => this.refresh(d));
  }

  dispose() {
    for (const state of this.states.values()) {
      state.watchers.forEach((w) => w.dispose());
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

module.exports = { Checking };
