'use strict';

// A stand-in for the `vscode` module, enough to activate the extension in plain node: the
// parts the tests look at are recorded (`fake.log`), the rest are inert. No API grants
// workspace trust from inside VS Code, so leaving Restricted Mode can only be tested this way.
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

class Position {
  constructor(line, character) {
    Object.assign(this, { line, character });
  }
}
class Range {
  constructor(a, b, c, d) {
    [this.start, this.end] = a instanceof Position ? [a, b] : [new Position(a, b), new Position(c, d)];
  }
}
/** The VS Code classes the provider tests look into: plain records of what they were given. */
const records = {
  Position,
  Range,
  TextEdit: {
    replace: (range, newText) => ({ range, newText }),
    insert: (position, newText) => ({ range: new Range(position, position), newText }),
  },
  Diagnostic: class Diagnostic { constructor(range, message, severity) { Object.assign(this, { range, message, severity }); } },
  Location: class Location { constructor(uri, range) { Object.assign(this, { uri, range }); } },
  Hover: class Hover { constructor(contents, range) { Object.assign(this, { contents, range }); } },
  MarkdownString: class MarkdownString { constructor(value = '') { this.value = value; } },
  SnippetString: class SnippetString { constructor(value = '') { this.value = value; } },
  CompletionItem: class CompletionItem { constructor(label, kind) { Object.assign(this, { label, kind }); } },
  InlayHint: class InlayHint { constructor(position, label) { Object.assign(this, { position, label }); } },
  DocumentSymbol: class DocumentSymbol {
    constructor(name, detail, kind, range, selectionRange) { Object.assign(this, { name, detail, kind, range, selectionRange }); }
  },
  WorkspaceEdit: class WorkspaceEdit {
    constructor() { this.edits = []; }
    replace(uri, range, newText) { this.edits.push({ uri, range, newText }); }
  },
  SemanticTokensLegend: class SemanticTokensLegend {
    constructor(tokenTypes, tokenModifiers = []) { Object.assign(this, { tokenTypes, tokenModifiers }); }
  },
  /** Keeps the tokens pushed, absolute, rather than VS Code's relative encoding. */
  SemanticTokensBuilder: class SemanticTokensBuilder {
    constructor(legend) { Object.assign(this, { legend, tokens: [] }); }
    push(line, character, length, type, modifiers = 0) { this.tokens.push([line, character, length, type, modifiers]); }
    build() { return { tokens: this.tokens }; }
  },
};

/** A text document as VS Code gives one to providers: offsets and positions in UTF-16 units. */
function textDocument(uri, text, languageId = 'ipu-asm') {
  const doc = {
    uri, languageId, version: 1, isClosed: false, text,
    getText(range) {
      return range ? this.text.slice(this.offsetAt(range.start), this.offsetAt(range.end)) : this.text;
    },
    get lineCount() { return this.text.split('\n').length; },
    lineAt(at) {
      const line = typeof at === 'number' ? at : at.line;
      const body = this.text.split('\n')[line].replace(/\r$/, '');
      return { text: body, range: new Range(line, 0, line, body.length) };
    },
    offsetAt(p) {
      const lines = this.text.split('\n');
      let at = 0;
      for (let n = 0; n < p.line && n < lines.length; n++) at += lines[n].length + 1;
      return Math.min(at + p.character, this.text.length);
    },
    positionAt(offset) {
      const before = this.text.slice(0, offset);
      const line = before.split('\n').length - 1;
      return new Position(line, offset - (before.lastIndexOf('\n') + 1));
    },
  };
  return doc;
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
  const log = { statuses: [], warnings: [], errors: [], terminals: [], watchers: [], commands: new Map(), providers: {}, diagnostics: {} };
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
    ...records,
    EventEmitter,
    Uri,
    RelativePattern: class RelativePattern {
      constructor(base, pattern) {
        Object.assign(this, { baseUri: base, pattern });
      }
    },
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
      /** Each watcher is noted with its pattern; `fire('Change', uri)` reports a file event. */
      createFileSystemWatcher: (pattern) => {
        const emitters = { Create: new EventEmitter(), Change: new EventEmitter(), Delete: new EventEmitter() };
        const watcher = {
          pattern,
          onDidCreate: emitters.Create.event, onDidChange: emitters.Change.event, onDidDelete: emitters.Delete.event,
          fire: (event, uri) => emitters[event].fire(uri),
          dispose() {},
        };
        log.watchers.push(watcher);
        return watcher;
      },
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
      /** Each collection by name, in `log.diagnostics`. */
      createDiagnosticCollection: (name) => {
        const map = new Map();
        log.diagnostics[name] = map;
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
  // `register…Provider(selector, provider, ...rest)`: each kind's providers, in `log.providers`.
  vscode.languages = new Proxy(vscode.languages, {
    get: (t, key) => {
      if (key in t) return t[key];
      const kind = /^register(\w+)Provider$/.exec(String(key));
      if (!kind) return disposable;
      return (_selector, provider, ...rest) => {
        (log.providers[kind[1]] = log.providers[kind[1]] || []).push({ provider, rest });
        return disposable();
      };
    },
  });
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

  // Loaded now, so activating it reads no module from disk.
  const extension = require(path.join(EXTENSION, 'extension.js'));
  const context = { subscriptions: [], workspaceState: { get() {}, async update() {} } };
  return {
    vscode,
    log,
    statusItem,
    context,
    events,
    /** Open a document holding `text` at `uri` in the active editor, as VS Code reports it. */
    open(uri, text, languageId = 'ipu-asm') {
      const doc = textDocument(uri, text, languageId);
      documents.push(doc);
      events.onDidOpenTextDocument.fire(doc);
      this.show(doc);
      return doc;
    },
    /** Replace `doc`'s text, as an edit does: a new version, and the change reported. */
    edit(doc, text) {
      doc.text = text;
      doc.version++;
      events.onDidChangeTextDocument.fire({ document: doc, contentChanges: [{ text }] });
    },
    /** The first provider of `kind` (`Hover`, `CompletionItem`, …) the extension registered;
     *  with `n`, the n-th. */
    provider(kind, n = 0) {
      const all = log.providers[kind] || [];
      if (!all[n]) throw new Error(`no ${kind} provider #${n} registered`);
      return all[n].provider;
    },
    /** Make `doc` the active editor's document. */
    show(doc) {
      vscode.window.activeTextEditor = { document: doc };
      events.onDidChangeActiveTextEditor.fire(vscode.window.activeTextEditor);
    },
    /** Trust the workspace, as the user granting it does. */
    trust() {
      trusted = true;
      events.onDidGrantWorkspaceTrust.fire();
    },
    activate() {
      return extension.activate(context);
    },
    /** Dispose what activation registered, as VS Code does on deactivation. */
    deactivate() {
      for (const d of context.subscriptions) if (d && d.dispose) d.dispose();
    },
  };
}

module.exports = { install, Uri, Position, Range };
