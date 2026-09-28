'use strict';

// Editing help that needs no toolchain, all on language.js (the ISA from data/isa.json).
// The assembler is consulted only where a guess would be unsafe: before a rename.

const vscode = require('vscode');

const LANGUAGE_ID = 'ipu-asm';

/** One line summarising what an operand type accepts. */
function describeDomain(domain) {
  if (!domain) return '';
  const range = 'min' in domain ? `${domain.min} … ${domain.max}` : '';
  const values = (domain.values || []).join(', ');
  const kinds = {
    enum: `One of: ${values}`,
    number: `Integer, ${range}`,
    mixed: `Integer ${range}, or one of: ${values}`,
    label: `A label defined in this file, or +N (a relative offset up to +${domain.relative_max})`,
  };
  return Object.hasOwn(kinds, domain.kind) ? kinds[domain.kind] : '';
}

function register(context, env) {
  const { lang, renderInstruction, renderRegister, checkText } = env;
  // Instruction docs never change: rendered once per mnemonic, not per request.
  const docs = new Map();
  const docFor = (mnemonic, forms) => {
    if (!docs.has(mnemonic)) docs.set(mnemonic, new vscode.MarkdownString(renderInstruction(mnemonic, forms)));
    return docs.get(mnemonic);
  };
  const rangeOf = (document, span) => new vscode.Range(document.positionAt(span.start), document.positionAt(span.end));
  const provide = (kind, provider, ...rest) =>
    context.subscriptions.push(vscode.languages[`register${kind}Provider`](LANGUAGE_ID, provider, ...rest));
  const item = (label, kind, props) => Object.assign(new vscode.CompletionItem(label, vscode.CompletionItemKind[kind]), props);

  // --- Completion ---
  provide('CompletionItem', {
    provideCompletionItems(document, position, _token, completionContext) {
      const text = document.getText();
      const offset = document.offsetAt(position);
      const ctx = lang.context(text, offset);
      // Space triggers so operand values appear as the operand starts; at an instruction's
      // start it (or the newline after `;;`) must not open the mnemonic list, or Enter would
      // accept an item instead of ending the line. Typing a letter opens it anyway.
      const typedTrigger = completionContext
        && completionContext.triggerKind === vscode.CompletionTriggerKind.TriggerCharacter;
      if (typedTrigger && ctx.kind !== 'operand') return undefined;
      if (ctx.kind === 'jinja') {
        if (ctx.region.kind === 'comment') return undefined;
        return lang.jinjaVisible(text, offset)
          .filter((d) => d.start < offset)
          .map((d) => item(d.name, d.kind === 'macro' ? 'Function' : 'Variable', {
            detail: d.kind === 'set' && d.expression ? `= ${d.expression}` : d.kind,
          }));
      }
      if (ctx.kind === 'comment' || ctx.kind === 'label') return undefined;
      if (ctx.kind === 'mnemonic') {
        // Operands already follow the cursor (a mnemonic being replaced): the name alone. A `;`
        // or a comment after it is no operand.
        const bare = /^\s*(?:[^\s;{]|\{\{)/.test(document.lineAt(position).text.slice(position.character).replace(/^[^\s;]*/, ''));
        return lang.mnemonicsFor(ctx.others).map(({ mnemonic, forms }) => {
          const form = forms[0];
          const operands = bare ? [] : (forms.find((f) => f.operands.length) || form).operands;
          return item(mnemonic, 'Keyword', {
            detail: forms.length > 1 ? 'any slot' : form.pseudo ? `${form.slot} · pseudo → ${form.expandsTo}` : form.slot,
            documentation: docFor(mnemonic, forms),
            // Its operands as tab stops, named as the instruction spec names them.
            ...(operands.length && {
              insertText: new vscode.SnippetString(`${mnemonic} ${operands.map((op, i) => `\${${i + 1}:${op.name}}`).join(' ')}`),
              command: { command: 'editor.action.triggerParameterHints', title: '' },
            }),
          });
        });
      }
      const typed = lang.operandType(ctx.mnemonic, ctx.index);
      if (!typed) return undefined;
      const { domain } = typed;
      // In the assembler's order (lr0 … lr15, not lr0, lr1, lr10, …), after the file's Jinja names.
      const items = (domain.values || []).map((value, i) => {
        const register = lang.registers.has(value.toLowerCase());
        return item(value, register ? 'Variable' : 'EnumMember', {
          sortText: `1${String(i).padStart(4, '0')}`,
          ...(register && { documentation: new vscode.MarkdownString(renderRegister(value.toLowerCase())) }),
        });
      });
      if (domain.kind === 'label') {
        const names = new Set(lang.labels(text).defs.map((d) => d.name));
        for (const name of names) items.push(item(name, 'Reference', { sortText: `2${name}` }));
      }
      // Template names whose value in force at the cursor this operand accepts: `{{ lr_row }}`.
      // Braces already typed are replaced, not doubled.
      const braces = ctx.prefix.startsWith('{');
      for (const d of lang.jinjaVisible(text, offset)) {
        if (d.kind !== 'set' || !lang.accepts(domain, d.value)) continue;
        items.push(item(`{{ ${d.name} }}`, 'Variable', {
          filterText: braces ? ctx.prefix + d.name : d.name,
          ...(braces && { range: new vscode.Range(document.positionAt(offset - ctx.prefix.length), position) }),
          detail: `= ${d.expression}`,
          sortText: `0${d.name}`, // the file's own names first
        }));
      }
      return items;
    },
  }, ' ');

  // --- Operand hints (signature help) ---
  provide('SignatureHelp', {
    provideSignatureHelp(document, position) {
      const ctx = lang.context(document.getText(), document.offsetAt(position));
      if (ctx.kind !== 'operand') return undefined;
      const entry = lang.lookup(ctx.mnemonic);
      if (!entry) return undefined;
      const help = new vscode.SignatureHelp();
      for (const form of entry.forms) {
        if (!form.operands.length) continue;
        let label = entry.mnemonic;
        const parameters = form.operands.map((op, i) => {
          const start = label.length + 1;
          label += ` ${op.name}`;
          const domain = lang.isa.operandTypes[op.type];
          const docLine = ((form.doc && form.doc.operands) || [])[i] || '';
          const md = [docLine, `\`${op.type}\` — ${describeDomain(domain)}`, domain && domain.description].filter(Boolean);
          return new vscode.ParameterInformation([start, start + op.name.length], new vscode.MarkdownString(md.join('\n\n')));
        });
        help.signatures.push(Object.assign(new vscode.SignatureInformation(label, (form.doc && form.doc.summary) || ''), { parameters }));
      }
      if (!help.signatures.length) return undefined;
      help.activeSignature = 0;
      help.activeParameter = Math.min(ctx.index, help.signatures[0].parameters.length - 1);
      return help;
    },
  }, { triggerCharacters: [' '], retriggerCharacters: [' '] });

  // --- Labels and Jinja names ---
  const symbolAt = (document, position) => lang.symbolAt(document.getText(), document.offsetAt(position));
  const locations = (document, spans) => spans.map((s) => new vscode.Location(document.uri, rangeOf(document, s)));

  provide('Definition', {
    provideDefinition(document, position) {
      const sym = symbolAt(document, position);
      return sym ? locations(document, sym.defs) : undefined;
    },
  });

  provide('Reference', {
    provideReferences(document, position, { includeDeclaration }) {
      const sym = symbolAt(document, position);
      if (!sym) return undefined;
      return locations(document, sym.refs.filter((r) => includeDeclaration || !sym.defs.some((d) => d.start === r.start)));
    },
  });

  provide('Rename', {
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

  // Hover on a Jinja name: its value where the cursor is, then the reference for that value if
  // it is a register or an instruction.
  const jinjaKinds = { macro: 'macro', param: 'macro parameter', for: 'loop variable' };
  provide('Hover', {
    provideHover(document, position) {
      const sym = symbolAt(document, position);
      if (!sym || sym.kind !== 'jinja') return undefined;
      const def = sym.binding;
      const lines = [def.kind !== 'set' ? `\`${sym.name}\` — Jinja ${jinjaKinds[def.kind]}`
        : def.expression === null ? `\`${sym.name}\` — block \`set\`` : `\`${sym.name}\` = \`${def.expression}\``];
      const value = def.value && def.value.toLowerCase();
      if (value && lang.registers.has(value)) lines.push('', renderRegister(value));
      const instruction = value && lang.lookup(value);
      if (instruction) lines.push('', renderInstruction(instruction.mnemonic, instruction.forms));
      return new vscode.Hover(new vscode.MarkdownString(lines.join('\n')), rangeOf(document, sym.range));
    },
  });

  // A `{{ name }}` with a literal value shows it after the tag: `{{ lr_row }}` lr1.
  provide('InlayHints', {
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
  provide('DocumentSymbol', {
    provideDocumentSymbols: (document) => lang.symbols(document.getText()).map((s) => {
      const range = rangeOf(document, s);
      return new vscode.DocumentSymbol(s.name, s.detail || s.kind, symbolKinds[s.kind], range, range);
    }),
  });

  provide('FoldingRange', {
    provideFoldingRanges: (document) => lang.folding(document.getText()).map((f) => {
      const start = document.positionAt(f.start).line;
      const end = document.positionAt(f.end).line;
      // Keep a block's closing tag visible; a comment folds whole.
      const comment = f.kind === 'comment';
      return new vscode.FoldingRange(start, !comment && end > start ? end - 1 : end,
        comment ? vscode.FoldingRangeKind.Comment : vscode.FoldingRangeKind.Region);
    }).filter((r) => r.end > r.start),
  });

  // --- Formatting ---
  /** How words are set apart (ipuAsm.wordSeparation): space on screen, an
   *  empty line in the file, a line drawn on screen, or nothing. */
  const separation = (document) =>
    vscode.workspace.getConfiguration('ipuAsm', document.uri).get('wordSeparation', 'space');

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
  provide('DocumentFormattingEdit', formatting);
  provide('DocumentRangeFormattingEdit', formatting);

  // As you type: `;` and `:` tidy the current line (its indent, a label to column 0, a second
  // word after `;;` onto its own line); Enter after a `;;` leaves the empty line between words.
  provide('OnTypeFormattingEdit', {
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

  // --- wordSeparation "line": a thin line drawn (not written) under each `;;` word ---
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
  provide('CodeLens', {
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

module.exports = { register };
