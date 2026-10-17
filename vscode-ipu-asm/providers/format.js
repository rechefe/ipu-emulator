'use strict';

// Formatting (whole document, range, and as you type), and how `;;` words are set apart on
// screen (ipuAsm.wordSeparation). A document or range is formatted by the checkout's assembler
// (`ipu-as format`, formatting.js) when it has `format`, else by the editor's own rules
// (language.js `format`), which are the same rules. Typing formats one line at a time, with the
// editor's rules: a process per keystroke would be too slow.

const vscode = require('vscode');

const { LANGUAGE_ID, provide } = require('./shared');

/** How words are set apart (ipuAsm.wordSeparation): space on screen, an
 *  empty line in the file, a line drawn on screen, or nothing. */
const separation = (document) =>
  vscode.workspace.getConfiguration('ipuAsm', document.uri).get('wordSeparation', 'space');

/** The language.js `format` options for lines [from, to]: indentation from the editor, the
 *  empty line between words from the setting. */
const formatOptions = (document, options, from, to) => ({
  indent: options.insertSpaces ? ' '.repeat(options.tabSize) : '\t',
  blankLine: separation(document) === 'emptyLine',
  fromLine: from,
  toLine: to,
});

/** The edit replacing lines [from, to] with `formatted`, if it differs. */
const editFor = (document, from, to, formatted) => {
  const range = new vscode.Range(from, 0, to, document.lineAt(to).text.length);
  return formatted === document.getText(range) ? [] : [vscode.TextEdit.replace(range, formatted)];
};

/** `format(document, options, token)`: the assembler's { lines }, { refused }, or null (see
 *  formatting.js CheckoutFormatter.format); absent, the editor's rules format everything. */
function registerFormatting(context, langOf, format = async () => null) {
  /** Lines [from, to] formatted by the editor's rules. */
  const formatLines = (document, options, from, to) =>
    editFor(document, from, to, langOf(document).format(document.getText(), formatOptions(document, options, from, to)));

  /** Lines [from, to] formatted by the assembler if it can, else by the editor's rules. A null
   *  `to` is the document's last line, whatever it is when the edits are made. */
  const formatWithAssembler = async (document, options, from, to, token) => {
    const last = () => (to === null ? document.lineCount - 1 : Math.min(to, document.lineCount - 1));
    const version = document.version;
    const asked = last();
    const result = await format(document, formatOptions(document, options, from, asked), token);
    if (token && token.isCancellationRequested) return [];
    // Edited while the assembler ran: its answer was for the old text, and the lines may be others.
    if (!result || document.version !== version) {
      return from <= last() ? formatLines(document, options, from, last()) : [];
    }
    if (result.refused) {
      vscode.window.showWarningMessage(`IPU Assembly: left as it is, since ${result.refused}`);
      return [];
    }
    return editFor(document, from, asked, result.lines);
  };

  const formatting = {
    provideDocumentFormattingEdits: (document, options, token) => formatWithAssembler(document, options, 0, null, token),
    provideDocumentRangeFormattingEdits: (document, range, options, token) =>
      formatWithAssembler(document, options, range.start.line, range.end.line, token),
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
      if (separation(document) === 'emptyLine' && langOf(document).lineEndsWord(document.getText(), line - 1)) {
        // At the end of the line above: an edit at the cursor's line would drop the
        // indentation Enter just gave it.
        edits.push(vscode.TextEdit.insert(document.lineAt(line - 1).range.end, '\n'));
      }
      return edits;
    },
  }, '\n', ';', ':');
}

/** wordSeparation "line" and "space", drawn on screen; for tests, the lines each puts them at. */
function registerSeparation(context, langOf, onLanguage) {
  // "line": a thin line drawn (not written) under each `;;` word.
  const separator = vscode.window.createTextEditorDecorationType({
    isWholeLine: true,
    borderStyle: 'solid',
    borderWidth: '0 0 1px 0',
    borderColor: new vscode.ThemeColor('ipuAsm.wordSeparator'),
  });
  const separatorLines = (document) => (separation(document) === 'line' ? langOf(document).wordEndLines(document.getText()) : []);
  const paint = (editor) => {
    const lines = editor.document.languageId === LANGUAGE_ID ? separatorLines(editor.document) : [];
    editor.setDecorations(separator, lines.map((n) => new vscode.Range(n, 0, n, 0)));
  };
  const paintAll = () => vscode.window.visibleTextEditors.forEach(paint);

  // With wordSeparation "space": an empty CodeLens above each word after the first. It is
  // the one way an extension can add vertical space that is not a line of the file.
  const gapsChanged = new vscode.EventEmitter();
  const gapLines = (document) => (separation(document) === 'space' ? langOf(document).wordGapLines(document.getText()) : []);
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
    // Another ISA (the checkout's, once generated) may split words elsewhere.
    onLanguage(() => {
      paintAll();
      gapsChanged.fire();
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

function register(context, { langOf, format, onLanguage = () => ({ dispose() {} }) }) {
  registerFormatting(context, langOf, format);
  return registerSeparation(context, langOf, onLanguage);
}

module.exports = { register };
