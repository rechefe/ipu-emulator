'use strict';

// Labels and Jinja names: definition, references, inlay values, unused names faded; and the
// outline and folding.

const vscode = require('vscode');

const { LANGUAGE_ID, provide, rangeOf } = require('./shared');

function register(context, { lang }) {
  const symbolAt = (document, position) => lang.symbolAt(document.getText(), document.offsetAt(position));
  const locations = (document, spans) => spans.map((s) => new vscode.Location(document.uri, rangeOf(document, s)));

  provide(context, 'Definition', {
    provideDefinition(document, position) {
      const sym = symbolAt(document, position);
      return sym ? locations(document, sym.defs) : undefined;
    },
  });

  provide(context, 'Reference', {
    provideReferences(document, position, { includeDeclaration }) {
      const sym = symbolAt(document, position);
      if (!sym) return undefined;
      return locations(document, sym.refs.filter((r) => includeDeclaration || !sym.defs.some((d) => d.start === r.start)));
    },
  });

  // A `{{ name }}` with a literal value shows it after the tag: `{{ lr_row }}` lr1.
  provide(context, 'InlayHints', {
    provideInlayHints(document, range) {
      const [from, to] = [document.offsetAt(range.start), document.offsetAt(range.end)];
      return lang.jinjaHints(document.getText()).filter((h) => from <= h.at && h.at <= to)
        .map((h) => Object.assign(new vscode.InlayHint(document.positionAt(h.at), h.value), { paddingLeft: true }));
    },
  });

  // Labels nothing branches to and `set` names nothing reads, faded (a hint: not in Problems).
  const unused = vscode.languages.createDiagnosticCollection('ipu-asm-unused');
  const markUnused = (document) => {
    if (document.languageId !== LANGUAGE_ID) return;
    unused.set(document.uri, lang.unused(document.getText()).map((u) => Object.assign(
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
    { dispose: () => unusedTimers.forEach((timer) => clearTimeout(timer)) }
  );
  vscode.workspace.textDocuments.forEach(markUnused);

  const symbolKinds = { label: vscode.SymbolKind.Function, set: vscode.SymbolKind.Variable, macro: vscode.SymbolKind.Method };
  provide(context, 'DocumentSymbol', {
    provideDocumentSymbols: (document) => lang.symbols(document.getText()).map((s) => {
      const range = rangeOf(document, s);
      return new vscode.DocumentSymbol(s.name, s.detail || s.kind, symbolKinds[s.kind], range, range);
    }),
  });

  provide(context, 'FoldingRange', {
    provideFoldingRanges: (document) => lang.folding(document.getText()).map((f) => {
      const start = document.positionAt(f.start).line;
      const end = document.positionAt(f.end).line;
      // Keep a block's closing tag visible; a comment folds whole.
      const comment = f.kind === 'comment';
      return new vscode.FoldingRange(start, !comment && end > start ? end - 1 : end,
        comment ? vscode.FoldingRangeKind.Comment : vscode.FoldingRangeKind.Region);
    }).filter((r) => r.end > r.start),
  });
}

module.exports = { register };
