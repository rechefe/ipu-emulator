'use strict';

// Formatting (whole document, range, and as you type), and how `;;` words are set apart on
// screen (ipuAsm.wordSeparation).

const vscode = require('vscode');

const { LANGUAGE_ID, provide } = require('./shared');

/** How words are set apart (ipuAsm.wordSeparation): space on screen, an
 *  empty line in the file, a line drawn on screen, or nothing. */
const separation = (document) =>
  vscode.workspace.getConfiguration('ipuAsm', document.uri).get('wordSeparation', 'space');

function registerFormatting(context, lang) {
  /** Replace lines [from, to] with their formatted text, if it differs: indentation from the
   *  editor, the empty line between words from the setting. */
  const formatLines = (document, options, from, to) => {
    const indent = options.insertSpaces ? ' '.repeat(options.tabSize) : '\t';
    const blankLine = separation(document) === 'emptyLine';
    const formatted = lang.format(document.getText(), { indent, blankLine, fromLine: from, toLine: to });
    const range = new vscode.Range(from, 0, to, document.lineAt(to).text.length);
    return formatted === document.getText(range) ? [] : [vscode.TextEdit.replace(range, formatted)];
  };

  const formatting = {
    provideDocumentFormattingEdits: (document, options) => formatLines(document, options, 0, document.lineCount - 1),
    provideDocumentRangeFormattingEdits: (document, range, options) =>
      formatLines(document, options, range.start.line, range.end.line),
  };
  provide(context, 'DocumentFormattingEdit', formatting);
  provide(context, 'DocumentRangeFormattingEdit', formatting);

  // As you type: `;` and `:` tidy the current line (its indent, a label to column 0, a second
  // word after `;;` onto its own line); Enter after a `;;` leaves the empty line between words.
  provide(context, 'OnTypeFormattingEdit', {
    provideOnTypeFormattingEdits(document, position, ch, options) {
      if (ch !== '\n') return formatLines(document, options, position.line, position.line);
      const line = position.line;
      if (line === 0) return [];
      const edits = formatLines(document, options, line - 1, line - 1);
      if (separation(document) === 'emptyLine' && lang.lineEndsWord(document.getText(), line - 1)) {
        // At the end of the line above: an edit at the cursor's line would drop the
        // indentation Enter just gave it.
        edits.push(vscode.TextEdit.insert(document.lineAt(line - 1).range.end, '\n'));
      }
      return edits;
    },
  }, '\n', ';', ':');
}

/** wordSeparation "line" and "space", drawn on screen; for tests, the lines each puts them at. */
function registerSeparation(context, lang) {
  // "line": a thin line drawn (not written) under each `;;` word.
  const separator = vscode.window.createTextEditorDecorationType({
    isWholeLine: true,
    borderStyle: 'solid',
    borderWidth: '0 0 1px 0',
    borderColor: new vscode.ThemeColor('ipuAsm.wordSeparator'),
  });
  const separatorLines = (document) => (separation(document) === 'line' ? lang.wordEndLines(document.getText()) : []);
  const paint = (editor) => {
    const lines = editor.document.languageId === LANGUAGE_ID ? separatorLines(editor.document) : [];
    editor.setDecorations(separator, lines.map((n) => new vscode.Range(n, 0, n, 0)));
  };
  const paintAll = () => vscode.window.visibleTextEditors.forEach(paint);

  // With wordSeparation "space": an empty CodeLens above each word after the first. It is
  // the one way an extension can add vertical space that is not a line of the file.
  const gapsChanged = new vscode.EventEmitter();
  const gapLines = (document) => (separation(document) === 'space' ? lang.wordGapLines(document.getText()) : []);
  provide(context, 'CodeLens', {
    onDidChangeCodeLenses: gapsChanged.event,
    provideCodeLenses: (document) =>
      gapLines(document).map((n) => new vscode.CodeLens(new vscode.Range(n, 0, n, 0), { title: '​', command: '' })),
  });
  const repaints = new Map();
  context.subscriptions.push(
    gapsChanged,
    separator,
    vscode.window.onDidChangeVisibleTextEditors(paintAll),
    // A language change reopens the document.
    vscode.workspace.onDidOpenTextDocument(paintAll),
    { dispose: () => repaints.forEach((timer) => clearTimeout(timer)) },
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('ipuAsm.wordSeparation')) {
        paintAll();
        gapsChanged.fire();
      }
    }),
    vscode.workspace.onDidChangeTextDocument(({ document, contentChanges }) => {
      // Only a drawn line needs repainting after an edit, once per burst of keystrokes.
      if (document.languageId !== LANGUAGE_ID || !contentChanges.length || separation(document) !== 'line') return;
      clearTimeout(repaints.get(document));
      repaints.set(document, setTimeout(() => {
        repaints.delete(document);
        vscode.window.visibleTextEditors.filter((e) => e.document === document).forEach(paint);
      }, 100));
    })
  );
  paintAll();
  return { separatorLines, gapLines };
}

function register(context, { lang }) {
  registerFormatting(context, lang);
  return registerSeparation(context, lang);
}

module.exports = { register };
