'use strict';

// Rename for labels and Jinja names, refused on a clash, or on a change the assembler reports
// as a new problem. The assembler is consulted only here, where a guess would be unsafe.

const vscode = require('vscode');

const { provide, rangeOf } = require('./shared');

function register(context, { lang, checkText }) {
  const symbolAt = (document, position) => lang.symbolAt(document.getText(), document.offsetAt(position));
  provide(context, 'Rename', {
    prepareRename(document, position) {
      const sym = symbolAt(document, position);
      if (!sym) throw new Error('Only labels and Jinja names can be renamed.');
      return { range: rangeOf(document, sym.range), placeholder: sym.name };
    },

    async provideRenameEdits(document, position, newName, token) {
      const sym = symbolAt(document, position);
      if (!sym) return undefined;
      const label = sym.kind === 'label';
      if (!lang.validName(sym.kind, newName)) throw new Error(`"${newName}" is not a valid ${label ? 'label' : 'Jinja name'}.`);
      const text = document.getText();
      if (label ? lang.labels(text).defs.some((d) => lang.sameLabel(d.name, newName)) : lang.jinjaClash(text, sym, newName)) {
        throw new Error(label
          ? `"${newName}" is already defined in this file.`
          : `"${newName}" is already defined or used where "${sym.name}" is.`);
      }
      // Check original and renamed text at once: refuse a problem only the renamed file has
      // (one that merely mentions the old name is the same problem). Without a working
      // checker the lexical check above is all there is, and the rename says so.
      const [original, result] = await Promise.all(
        [text, lang.rename(text, sym, newName)].map((t) => checkText(document.uri, t, token)));
      if (token && token.isCancellationRequested) return undefined;
      if (original && original.found && result && result.found) {
        const key = (d) => `${d.stage}|${d.message.split(sym.name).join(newName)}`;
        const known = new Set(original.found.map(key));
        const added = result.found.find((d) => !known.has(key(d)));
        if (added) throw new Error(`Renaming to "${newName}" would not assemble: ${added.message}`);
      } else if (!(original && original.outside)) {
        // (Outside an IPU checkout no checker runs, and none is missed.)
        const why = [original, result].map((r) => r && r.error).find(Boolean) || 'the checker did not run';
        vscode.window.showWarningMessage(`IPU: renamed without the assembler's check (${why}).`);
      }
      const edit = new vscode.WorkspaceEdit();
      for (const s of sym.refs) edit.replace(document.uri, rangeOf(document, s), newName);
      return edit;
    },
  });
}

module.exports = { register };
