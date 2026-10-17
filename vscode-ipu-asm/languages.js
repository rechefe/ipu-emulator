'use strict';

// Each document's language data (E08-25): its checkout's ISA once generated, else the bundled
// one (language-source.js says which, and why). The bundled data answers at once; a checkout's
// is generated in the background and swapped in when ready, and `onDidChange` tells the
// providers whose answers may have changed to refresh.

const os = require('os');
const path = require('path');
const vscode = require('vscode');

const { CheckoutLanguageSource, SETTINGS, select } = require('./language-source');
const { setting, timeoutSeconds } = require('./kernels');
const trust = require('./trust');

class Languages {
  /** `bundled`: the Language over data/isa.json. `checkoutOf(uri)`: the document's checkout as
   *  last found (null if none or not looked at yet); `findCheckout(uri)`: look it up. */
  constructor(context, { bundled, checkoutOf, findCheckout }) {
    this.bundled = bundled;
    this.checkoutOf = checkoutOf;
    this.findCheckout = findCheckout;
    this.cacheBase = context.globalStorageUri
      ? path.join(context.globalStorageUri.fsPath, 'language')
      : path.join(os.tmpdir(), 'ipu-asm-language');
    /** Checkout root -> { source, result, loading, again, warned }. */
    this.entries = new Map();
    /** Documents (by Uri) whose checkout is being looked up. */
    this.looking = new Set();
    this.changed = new vscode.EventEmitter();
    /** Fired with a checkout root (or null: any) whose documents' language may have changed. */
    this.onDidChange = this.changed.event;
    context.subscriptions.push(
      this.changed,
      vscode.workspace.onDidChangeConfiguration((e) => e.affectsConfiguration('ipuAsm.languageSource') && this.changed.fire(null)),
      vscode.workspace.onDidGrantWorkspaceTrust(() => this.changed.fire(null))
    );
  }

  /** The Language a document's providers use. */
  of(document) {
    return this.chosen(document).lang;
  }

  /** What `document` uses: { lang, use: 'checkout' | 'bundled', why }. Starts generating its
   *  checkout's data if it may be used and is not there yet. */
  chosen(document) {
    const { uri } = document;
    const folder = vscode.workspace.getWorkspaceFolder(uri);
    const raw = setting(folder, 'languageSource', 'auto');
    const value = SETTINGS.includes(raw) ? raw : 'auto';
    const trusted = trust.mayRun();
    const root = this.checkoutOf(uri);
    if (!root && trusted && value !== 'bundled') this.lookUp(uri);
    const entry = root && trusted && value !== 'bundled' ? this.ensure(root, folder) : null;
    const choice = select({ trusted, setting: value, root, checkout: entry ? entry.result : undefined });
    if (choice.warn && entry && !entry.warned) {
      entry.warned = true;
      vscode.window.showWarningMessage(choice.warn);
    }
    return { lang: choice.use === 'checkout' ? entry.result.lang : this.bundled, use: choice.use, why: choice.why };
  }

  /** A document not looked at yet: find its checkout, and refresh once it is known. */
  lookUp(uri) {
    const key = uri.toString();
    if (this.looking.has(key)) return;
    this.looking.add(key);
    Promise.resolve(this.findCheckout(uri)).then((root) => {
      if (root) this.changed.fire(root);
    }, () => {});
  }

  /** `root`'s entry, its data loading if it has none yet or its sources changed. */
  ensure(root, folder) {
    if (!this.entries.has(root)) {
      const source = new CheckoutLanguageSource(root, { cacheBase: this.cacheBase, seconds: timeoutSeconds(folder) });
      this.entries.set(root, { source, result: undefined, loading: null, again: true, warned: false });
    }
    const entry = this.entries.get(root);
    if (entry.again && !entry.loading) {
      entry.again = false;
      entry.loading = entry.source.load().catch((err) => ({ error: err.message })).then((result) => {
        // A failure after a success keeps the data it had: the sources are mid-edit.
        if (!(result.error && entry.result && entry.result.lang)) entry.result = result;
        entry.loading = null;
        if (result.error) console.warn(`[ipu-asm] ${result.error}`);
        this.changed.fire(root);
      });
    }
    return entry;
  }

  /** `root`'s sources changed: generate its data again the next time a document asks. Its
   *  current data answers until then. */
  invalidate(root) {
    const entry = this.entries.get(root);
    if (entry) entry.again = true;
    this.changed.fire(root);
  }
}

module.exports = { Languages };
