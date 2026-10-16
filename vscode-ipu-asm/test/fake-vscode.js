'use strict';

// A stand-in for the `vscode` module, enough to activate the extension in plain node: the
// parts the tests look at are recorded (`fake.log`), the rest are inert. VS Code trusts every
// workspace while it runs extension tests, so Restricted Mode can only be tested this way.
//
//   const fake = require('./fake-vscode').install({ trusted: false, folders: [dir] });
//   const api = fake.activate();

const Module = require('module');
const path = require('path');

const EXTENSION = path.resolve(__dirname, '..');

class EventEmitter {
  constructor() {
    this.listeners = new Set();
    this.event = (listener) => {
      this.listeners.add(listener);
      return { dispose: () => this.listeners.delete(listener) };
    };
  }
  fire(value) {
    for (const listener of [...this.listeners]) listener(value);
  }
  dispose() {
    this.listeners.clear();
  }
}

class Uri {
  constructor(scheme, fsPath) {
    this.scheme = scheme;
    this.fsPath = fsPath;
    this.path = fsPath;
  }
  static file(fsPath) {
    return new Uri('file', fsPath);
  }
  toString() {
    return `${this.scheme}://${this.fsPath}`;
  }
}

/** Any class or enum the extension names and the tests do not look at: constructible, and
 *  each of its members is its own name. */
const stub = (name) => new Proxy(function Stub(...args) { this.args = args; }, {
  get: (target, key) => (key in target ? target[key] : `${name}.${String(key)}`),
});

const disposable = () => ({ dispose() {} });

/** Install the fake as `vscode` for every module required from now on. `folders` are the
 *  workspace folders' paths; `settings` the `ipuAsm.*` values each folder sees. */
function install({ trusted = true, folders = [], settings = {} } = {}) {
  const log = { statuses: [], warnings: [], errors: [], terminals: [], commands: new Map() };
  const events = new Proxy({}, { get: (all, key) => (all[key] = all[key] || new EventEmitter()) });
  const on = (name) => events[name].event;
  const workspaceFolders = folders.map((dir, index) => ({ uri: Uri.file(dir), name: path.basename(dir), index }));
  const documents = [];
  const statusItem = {
    visible: false,
    show() { this.visible = true; log.statuses.push(this.text); },
    hide() { this.visible = false; },
    dispose() {},
  };

  const vscode = {
    EventEmitter,
    Uri,
    workspace: {
      get isTrusted() { return trusted; },
      workspaceFolders,
      textDocuments: documents,
      getWorkspaceFolder: (uri) => workspaceFolders.find((f) => uri.scheme === 'file'
        && (uri.fsPath === f.uri.fsPath || uri.fsPath.startsWith(f.uri.fsPath + path.sep))),
      getConfiguration: () => ({
        get: (key, fallback) => (Object.hasOwn(settings, key) ? settings[key] : fallback),
        inspect: () => ({}),
        update: async () => {},
      }),
      createFileSystemWatcher: () => ({
        onDidCreate: disposable, onDidChange: disposable, onDidDelete: disposable, dispose() {},
      }),
    },
    window: {
      activeTextEditor: undefined,
      visibleTextEditors: [],
      createStatusBarItem: () => statusItem,
      createOutputChannel: () => ({ appendLine() {}, show() {}, dispose() {} }),
      createTreeView: () => ({ visible: false, onDidChangeVisibility: disposable, reveal: async () => {}, dispose() {} }),
      createTextEditorDecorationType: disposable,
      createTerminal: (options) => {
        const terminal = { ...options, sent: [], show() {}, sendText(text) { this.sent.push(text); } };
        log.terminals.push(terminal);
        return terminal;
      },
      registerWebviewViewProvider: disposable,
      showWarningMessage: async (text) => { log.warnings.push(text); },
      showErrorMessage: async (text) => { log.errors.push(text); },
      showInformationMessage: async () => {},
      withProgress: (_options, task) => task(),
    },
    languages: {
      createDiagnosticCollection: () => {
        const map = new Map();
        return { set: (uri, d) => map.set(uri.toString(), d), delete: (uri) => map.delete(uri.toString()), get: (uri) => map.get(uri.toString()), dispose() {} };
      },
    },
    commands: {
      registerCommand: (name, run) => {
        log.commands.set(name, run);
        return disposable();
      },
      executeCommand: async (name, ...args) => (log.commands.has(name) ? log.commands.get(name)(...args) : undefined),
    },
  };
  // Every `workspace.onDid…` and `window.onDid…` event is an emitter the test can fire.
  for (const area of [vscode.workspace, vscode.window]) {
    for (const name of ['onDidGrantWorkspaceTrust', 'onDidOpenTextDocument', 'onDidSaveTextDocument', 'onDidChangeTextDocument',
      'onDidCloseTextDocument', 'onDidChangeWorkspaceFolders', 'onDidChangeConfiguration', 'onDidChangeWindowState',
      'onDidChangeActiveTextEditor', 'onDidCloseTerminal', 'onDidChangeVisibleTextEditors']) {
      if (!(name in area)) area[name] = on(name);
    }
  }
  vscode.languages = new Proxy(vscode.languages, { get: (t, key) => (key in t ? t[key] : disposable) });
  const module = new Proxy(vscode, { get: (t, key) => (key in t ? t[key] : stub(String(key))) });

  // A fresh copy of the extension's modules, so each install starts from clean module state.
  for (const file of Object.keys(require.cache)) {
    if (file.startsWith(EXTENSION + path.sep) && !file.includes(`${path.sep}test${path.sep}`)) delete require.cache[file];
  }
  const resolve = Module._resolveFilename;
  Module._resolveFilename = function resolveVscode(request, ...rest) {
    return request === 'vscode' ? 'vscode' : resolve.call(this, request, ...rest);
  };
  require.cache.vscode = { id: 'vscode', filename: 'vscode', loaded: true, exports: module };

  const context = { subscriptions: [], workspaceState: { get() {}, async update() {} } };
  return {
    vscode,
    log,
    statusItem,
    context,
    events,
    /** Open a document holding `text` at `uri`, as VS Code reports one opening. */
    open(uri, text, languageId = 'ipu-asm') {
      const doc = { uri, languageId, version: 1, isClosed: false, getText: () => text };
      documents.push(doc);
      events.onDidOpenTextDocument.fire(doc);
      return doc;
    },
    /** Trust the workspace, as the user granting it does. */
    trust() {
      trusted = true;
      events.onDidGrantWorkspaceTrust.fire();
    },
    activate() {
      return require(path.join(EXTENSION, 'extension.js')).activate(context);
    },
    deactivate() {
      for (const d of context.subscriptions) if (d && d.dispose) d.dispose();
      require(path.join(EXTENSION, 'extension.js')).deactivate();
    },
  };
}

module.exports = { install, Uri };
