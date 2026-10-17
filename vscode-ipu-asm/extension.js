'use strict';

// The extension's entry point: the assembler's diagnostics (checking.js), the kernel buttons
// and sidebar (kernels.js), and the editing help (editing.js) over each document's language
// data (languages.js).

const vscode = require('vscode');

const { Checking } = require('./checking');
const editing = require('./editing');
const kernels = require('./kernels');
const { Languages } = require('./languages');
const { BundledLanguageSource } = require('./language-source');

/** language.js over the bundled isa.json (built with the extension, so always matching), or null. */
async function loadLanguage() {
  try {
    return (await new BundledLanguageSource().load()).lang;
  } catch (err) {
    // A convenience: losing it must not take highlighting with it.
    console.warn(`[ipu-asm] no language data (${err.message})`);
    return null;
  }
}

/** Activate: diagnostics and the kernels at once; the editing help once the language data
 *  is read. Nothing here touches the disk synchronously: the extension host has one thread. */
async function activate(context) {
  const kernelApi = kernels.register(context);
  const checking = new Checking(kernelApi);
  context.subscriptions.push(checking);

  // After activation returns, since starting a process blocks briefly.
  setImmediate(() => checking.start());

  context.subscriptions.push(
    // Out of Restricted Mode: what it kept from running starts now.
    vscode.workspace.onDidGrantWorkspaceTrust(() => checking.start()),
    vscode.workspace.onDidOpenTextDocument((d) => checking.refresh(d)),
    vscode.workspace.onDidSaveTextDocument((d) => checking.refresh(d)),
    // No content change (the dirty flag flipped): nothing new to check.
    vscode.workspace.onDidChangeTextDocument((e) => e.contentChanges.length && checking.scheduleRefresh(e.document)),
    vscode.workspace.onDidCloseTextDocument((d) => checking.forget(d)),
    vscode.window.onDidChangeActiveTextEditor(() => checking.showStatus()),
    // Resolves once the check is done, so a caller (a test) can wait on it.
    vscode.commands.registerCommand('ipuAsm.check', () => {
      const editor = vscode.window.activeTextEditor;
      return editor ? checking.refresh(editor.document, true) : undefined;
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

  let editingApi = null;
  const bundled = await loadLanguage();
  if (bundled) {
    const languages = new Languages(context, {
      bundled,
      checkoutOf: (uri) => checking.knownCheckout(uri),
      findCheckout: (uri) => checking.checkoutFor(uri),
    });
    checking.languages = languages;
    context.subscriptions.push(languages.onDidChange(() => checking.showStatus()));
    const langOf = (document) => languages.of(document);
    /** Numbers rename checks, so concurrent ones never supersede each other. */
    let renameChecks = 0;
    editingApi = editing.register(context, {
      langOf,
      onLanguage: (listener) => languages.onDidChange(listener),
      // The document's own text was checked already (unless its check failed): not again.
      checkText: (uri, text, token) => checking.keptCheck(uri, text)
        || checking.runCheckText(uri, text, `${uri.toString()}#rename-${++renameChecks}`, token),
      semantics: (document, options) => checking.semanticsOf(langOf(document), document, options),
      answer: (document) => checking.analysis.current(document.uri.toString(), document.version, document.getText()),
      onAnswer: (listener) => checking.analysis.onStore(listener),
      format: (document, options, token) => checking.formatWithChecker(document, options, token),
    });
  }
  // For tests: the kernels and editing APIs, and why diagnostics are off (for a document,
  // the active editor's, or any), if they are.
  return { kernels: kernelApi, editing: editingApi, checkerError: (uri) => checking.errorFor(uri), analysis: checking.analysis, languages: checking.languages };
}

module.exports = { activate };