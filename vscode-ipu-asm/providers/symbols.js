'use strict';

// Labels and Jinja names: definition, references, inlay values, unused names faded; and the
// outline and folding.

const vscode = require('vscode');

const { LANGUAGE_ID, onSwitch, provide, rangeOf, switched } = require('./shared');
const { featureOn } = require('../settings');

function register(context, { langOf, semantics, onAnswer, onLanguage = () => ({ dispose() {} }) }) {
  const symbolAt = async (document, position) => (await semantics(document)).symbolAt(document.offsetAt(position));
  const locations = (document, spans) => spans.map((s) => new vscode.Location(document.uri, rangeOf(document, s)));

  provide(context, 'Definition', switched('navigation', {
    async provideDefinition(document, position) {
      const sym = await symbolAt(document, position);
      return sym ? locations(document, sym.defs) : undefined;
    },
  }));

  provide(context, 'Reference', switched('navigation', {
    async provideReferences(document, position, { includeDeclaration }) {
      const sym = await symbolAt(document, position);
      if (!sym) return undefined;
      return locations(document, sym.refs.filter((r) => includeDeclaration || !sym.defs.some((d) => d.start === r.start)));
    },
  }));

  // A `{{ name }}` with a literal value shows it after the tag: `{{ lr_row }}` lr1.
  const hintsChanged = new vscode.EventEmitter();
  provide(context, 'InlayHints', switched('jinjaValueHints', {
    onDidChangeInlayHints: hintsChanged.event,
    async provideInlayHints(document, range) {
      const [from, to] = [document.offsetAt(range.start), document.offsetAt(range.end)];
      return (await semantics(document)).jinjaHints().filter((h) => from <= h.at && h.at <= to)
        .map((h) => Object.assign(new vscode.InlayHint(document.positionAt(h.at), h.value), { paddingLeft: true }));
    },
  }));

  // Labels nothing branches to and `set` names nothing reads, faded (a hint: not in Problems).
  const unused = vscode.languages.createDiagnosticCollection('ipu-asm-unused');
  const markUnused = async (document) => {
    if (document.languageId !== LANGUAGE_ID) return;
    if (!featureOn(document.uri, 'unusedNames')) return unused.delete(document.uri);
    const { version } = document;
    const found = (await semantics(document)).unused();
    // A newer version has its own marking on the way; a closed document has none, and one
    // switched off meanwhile none either.
    if (document.isClosed || document.version !== version || !featureOn(document.uri, 'unusedNames')) return;
    unused.set(document.uri, found.map((u) => Object.assign(
      new vscode.Diagnostic(rangeOf(document, u), u.kind === 'label' ? `${u.name} is never branched to` : `${u.name} is set but never used`,
        vscode.DiagnosticSeverity.Hint),
      { source: 'ipu-asm', tags: [vscode.DiagnosticTag.Unnecessary] })));
  };
  const unusedTimers = new Map();
  context.subscriptions.push(
    unused,
    vscode.workspace.onDidOpenTextDocument(markUnused),
    vscode.workspace.onDidCloseTextDocument((d) => unused.delete(d.uri)),
    vscode.workspace.onDidChangeTextDocument(({ document, contentChanges }) => {
      if (document.languageId !== LANGUAGE_ID || !contentChanges.length) return;
      clearTimeout(unusedTimers.get(document));
      unusedTimers.set(document, setTimeout(() => {
        unusedTimers.delete(document);
        markUnused(document);
      }, 300));
    }),
    { dispose: () => unusedTimers.forEach((timer) => clearTimeout(timer)) },
    // An analysis that arrives replaces what was first painted from the editor's own rules.
    hintsChanged,
    onAnswer((key, answer) => {
      if (!answer) return;
      const document = vscode.workspace.textDocuments.find((d) => d.uri.toString() === key);
      if (document) markUnused(document);
      hintsChanged.fire();
    }),
    // Another ISA (the checkout's, once generated): the editor's own rules may read otherwise.
    onLanguage(() => {
      vscode.workspace.textDocuments.forEach(markUnused);
      hintsChanged.fire();
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('ipuAsm.unusedNames')) vscode.workspace.textDocuments.forEach(markUnused);
      if (e.affectsConfiguration('ipuAsm.jinjaValueHints')) hintsChanged.fire();
    })
  );
  vscode.workspace.textDocuments.forEach(markUnused);

  const symbolKinds = { label: vscode.SymbolKind.Function, set: vscode.SymbolKind.Variable, macro: vscode.SymbolKind.Method };
  provide(context, 'DocumentSymbol', switched('outline', {
    provideDocumentSymbols: async (document) => (await semantics(document)).symbols().map((s) => {
      const range = rangeOf(document, s);
      return new vscode.DocumentSymbol(s.name, s.detail || s.kind, symbolKinds[s.kind], range, range);
    }),
  }));

  provide(context, 'FoldingRange', switched('folding', {
    onDidChangeFoldingRanges: onSwitch(context, 'folding'),
    provideFoldingRanges: (document) => langOf(document).folding(document.getText()).map((f) => {
      const start = document.positionAt(f.start).line;
      const end = document.positionAt(f.end).line;
      // Keep a block's closing tag visible; a comment folds whole.
      const comment = f.kind === 'comment';
      return new vscode.FoldingRange(start, !comment && end > start ? end - 1 : end,
        comment ? vscode.FoldingRangeKind.Comment : vscode.FoldingRangeKind.Region);
    }).filter((r) => r.end > r.start),
  }));
}

module.exports = { register };
