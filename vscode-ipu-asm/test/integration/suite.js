'use strict';

// Runs inside VS Code's extension host (see run.js), through VS Code's own command API. Language
// logic is asserted by ../language.js; these checks prove the wiring of each provider, command and setting.

const assert = require('assert');
const path = require('path');
const vscode = require('vscode');

const runs = require('../../runs');
const isa = require('../../data/isa.json');
const { test, sleep, until, runAll } = require('../harness');

const workspace = process.env.IPU_TEST_WORKSPACE;
const kernelFile = path.join(workspace, 'kernels/demo/demo_kernel/demo_kernel.asm');
const scratchFile = path.join(workspace, 'kernels/demo/scratch.asm');

let doc;
/** Position in the kernel of the first `needle`, plus `delta` characters. */
function pos(needle, delta = 0) {
  const at = doc.getText().indexOf(needle);
  assert(at >= 0, `fixture has no ${JSON.stringify(needle)}`);
  return doc.positionAt(at + delta);
}

/** An unsaved document holding `text`. */
const withText = (text) => vscode.workspace.openTextDocument({ language: 'ipu-asm', content: text });
const exec = (...args) => vscode.commands.executeCommand(...args);
const extension = () => vscode.extensions.getExtension('yardencarmi.ipu-asm');
const api = () => extension().exports.kernels;
const editingApi = () => extension().exports.editing;

const name = (i) => (typeof i.label === 'string' ? i.label : i.label.label);
const labels = (items) => items.map(name);
const completion = async (uri, at, trigger) => (await exec('vscode.executeCompletionItemProvider', uri, at, trigger)).items;
/** Completion items in a document holding `text`, at line 0 character `ch`. */
const completeIn = async (text, ch, trigger) => completion((await withText(text)).uri, new vscode.Position(0, ch), trigger);
const rename = (uri, at, to) => exec('vscode.executeDocumentRenameProvider', uri, at, to);
const hoverText = async (uri, at) =>
  (await exec('vscode.executeHoverProvider', uri, at)).map((h) => h.contents.map((c) => c.value || c).join('\n')).join('\n');
/** Lines ending a `;;` word, but the last. */
const wordEnds = () => doc.getText().split('\n').map((l, n) => (l.trimEnd().endsWith(';;') ? n : -1)).filter((n) => n >= 0).slice(0, -1);
const loopText = '{% set r = "lr0" %}\n{% for i in range(2) %}\nSET {{ r }} cr0;;\n{% endfor %}\nBKPT;;\n';

/** Run `fn` with the `ipuAsm` workspace settings in `settings`, then restore them. */
async function withSettings(settings, fn) {
  const config = () => vscode.workspace.getConfiguration('ipuAsm');
  const set = async (entries) => { for (const [k, v] of entries) await config().update(k, v, vscode.ConfigurationTarget.Workspace); };
  const saved = Object.keys(settings).map((k) => [k, config().inspect(k).workspaceValue]);
  await set(Object.entries(settings));
  return Promise.resolve().then(fn).finally(() => set(saved));
}

// --- completion, signature help ------------------------------------------------

test('mnemonics that fit the word are offered at the start of an instruction, not on a typed space', async () => {
  const names = labels(await completeIn('    \n', 4));
  for (const m of ['ACC.ADD.FIRST', 'MULT.RC.VE', 'BKPT', 'SET', 'B']) assert(names.includes(m), m);
  const inWord = labels(await completeIn('BKPT ; \n', 7));
  assert(!inWord.includes('BEQ') && !inWord.includes('B') && inWord.includes('SET'), 'cond is full');
  // A typed space or newline there opens nothing: Enter would accept a suggestion instead of ending
  // the line. Plain words (kind Text) aside, no mnemonic (Keyword).
  const mnemonics = (await completeIn('SET lr0 cr0 ;; \n', 15, ' ')).filter((i) => i.kind === vscode.CompletionItemKind.Keyword);
  assert.strictEqual(mnemonics.length, 0, labels(mnemonics).join(' '));
});

test('a typed space at an operand opens the values of its type, in the assembler\'s order', async () => {
  const names = labels(await completeIn('ACTIVATE.QUANTIZE \n', 18, ' '));
  assert(names.includes('window') && names.includes('relu') && !names.includes('lr0'));
  const regs = (await completeIn('ADD \n', 4)).filter((i) => /^lr\d+$/.test(name(i))).sort((a, b) => (a.sortText < b.sortText ? -1 : 1));
  assert.deepStrictEqual(labels(regs.slice(0, 3)), ['lr0', 'lr1', 'lr2']);
});

test('Jinja names and labels are offered; braces already typed are replaced', async () => {
  assert(labels(await completion(doc.uri, pos('ADD {{lr_row}}', 'ADD '.length))).includes('{{ lr_row }}'));
  assert(labels(await completion(doc.uri, pos('row_loop ;;'))).includes('row_loop'));
  const d = await withText('{% set r = "lr1" %}\nSET {');
  const item = (await completion(d.uri, new vscode.Position(1, 5))).find((i) => name(i) === '{{ r }}');
  assert(item, 'offered');
  assert.strictEqual((item.range.inserting || item.range).start.character, 4, 'the replacement starts at the typed brace');
});

test('a mnemonic brings its operands as tab stops; a {{ name }} shows its value; unused names are faded', async () => {
  const items = await completeIn('    \n', 4);
  const set = items.find((i) => name(i) === 'SET');
  assert.match(set.insertText.value, /^SET \$\{1:\w+\} \$\{2:\w+\}$/);
  const template = (item) => item.insertText instanceof vscode.SnippetString;
  assert(!template(items.find((i) => name(i) === 'BKPT')));
  // Operands already there: the name alone.
  assert(!template((await completeIn('S lr0 cr0;;\n', 1)).find((i) => name(i) === 'SET')));
  const d = await vscode.window.showTextDocument(await withText('{% set r = "lr0" %}\nSET {{ r }} cr0;;\nend:\n    BKPT;;\n')).then((e) => e.document);
  const hints = await exec('vscode.executeInlayHintProvider', d.uri, new vscode.Range(0, 0, 4, 0));
  assert.deepStrictEqual(hints.map((h) => [h.label, h.position.line]), [['lr0', 1]]);
  const fadedNow = () => vscode.languages.getDiagnostics(d.uri).filter((x) => x.tags && x.tags.includes(vscode.DiagnosticTag.Unnecessary));
  const faded = await until(() => fadedNow().length && fadedNow(), 'the unused label');
  assert.deepStrictEqual(faded.map((x) => [x.message, x.severity]), [['end is never branched to', vscode.DiagnosticSeverity.Hint]]);
});

test('signature help highlights the operand being typed', async () => {
  const d = await withText('MULT.RC.VE lr0 \n');
  const help = await exec('vscode.executeSignatureHelpProvider', d.uri, new vscode.Position(0, 15), ' ');
  assert(help && help.signatures.length && help.signatures[0].label.startsWith('MULT.RC.VE '));
  assert.strictEqual(help.activeParameter, 1);
  const docs = help.signatures[0].parameters[1].documentation;
  assert((docs.value || docs).length > 0);
});

// --- navigation, hover, outline, folding ------------------------------------------------

test('definition of a label and a Jinja name, references, rename and its refusals', async () => {
  for (const [from, delta, to] of [['row_loop ;;', 2, 'row_loop:'], ['{{lr_row}} cr3', 3, 'set lr_row']]) {
    const locs = await exec('vscode.executeDefinitionProvider', doc.uri, pos(from, delta));
    assert.deepStrictEqual(locs.map((l) => l.range.start.line), [pos(to).line], from);
  }
  const refs = await exec('vscode.executeReferenceProvider', doc.uri, pos('{{lr_row}} cr3', 3));
  assert.strictEqual(refs.length, 1 + doc.getText().match(/\{\{lr_row\}\}/g).length);
  // Rename a label changes its definition and references; refused for a Jinja constant, a name in use, a non-label.
  const changes = (await rename(doc.uri, pos('row_loop:', 1), 'next_row')).get(doc.uri);
  assert.deepStrictEqual(changes.map((c) => c.newText), ['next_row', 'next_row']);
  await assert.rejects(rename(doc.uri, pos('set lr_zero', 5), 'None'));
  await assert.rejects(rename((await withText(loopText)).uri, new vscode.Position(0, 8), 'range'), /already defined or used/);
  await assert.rejects(rename(doc.uri, pos('row_loop:', 1), '+3'));
});

test('hover on an instruction, a Jinja name (its value and register) and a macro parameter', async () => {
  const instruction = await hoverText(doc.uri, pos('ACC.ADD.FIRST', 1));
  assert(instruction.includes(isa.instructions['ACC.ADD.FIRST'][0].doc.title), instruction);
  const jinja = await hoverText(doc.uri, pos('{{lr_row}} cr3', 3));
  assert(jinja.includes('"lr1"') && jinja.includes(isa.registerDocs.lr1.about), jinja);
  // A macro parameter is named as one, and not taken for the top-level name it shares.
  const d = await withText('{% set r = "lr0" %}\n{% macro m(r) %}\nSET {{ r }} cr0;;\n{% endmacro %}\n');
  const param = await hoverText(d.uri, d.positionAt(d.getText().indexOf('{{ r }}') + 3));
  assert(param.includes('macro parameter') && !param.includes('lr0'), param);
});

test('outline lists labels and Jinja names; the header comment folds', async () => {
  const names = (await exec('vscode.executeDocumentSymbolProvider', doc.uri)).map((s) => s.name);
  for (const n of ['lr_row', 'lr_zero', 'row_loop']) assert(names.includes(n), n);
  const ranges = await exec('vscode.executeFoldingRangeProvider', doc.uri);
  assert(ranges.some((r) => r.start === 0 && r.end >= 2), JSON.stringify(ranges));
});

// --- formatting, word separation, typing ------------------------------------------------------------

const messy = 'top: SET lr0 cr0 ;; BKPT;;\n  ACC.ADD.FIRST;\nBKPT;;\n';
const tidy = 'top:\n    SET lr0 cr0 ;;\n    BKPT;;\n    ACC.ADD.FIRST;\n    BKPT;;\n';
/** `text` after Format Document, or Format Selection on `range`. */
async function formatted(text, range) {
  const d = await withText(text);
  const options = { tabSize: 4, insertSpaces: true };
  const we = new vscode.WorkspaceEdit();
  we.set(d.uri, (await (range ? exec('vscode.executeFormatRangeProvider', d.uri, range, options)
    : exec('vscode.executeFormatDocumentProvider', d.uri, options))) || []);
  await vscode.workspace.applyEdit(we);
  return d.getText();
}

test('Format Selection formats the selected lines only', async () =>
  assert.strictEqual(await formatted('  a:\nBKPT;;\nBKPT;; BKPT;;\n', new vscode.Range(1, 0, 1, 6)), '  a:\n    BKPT;;\nBKPT;; BKPT;;\n'));

const none = () => [];
const spaceGaps = () => { const t = doc.getText().split('\n'); return wordEnds().map((n) => n + 1).filter((n) => t[n].trim()); };
// [wordSeparation (undefined: the default, "space"), Format Document's result, on-screen gaps (CodeLens), drawn lines]
for (const [mode, result, gaps, drawn] of [
  [undefined, tidy, spaceGaps, none],
  ['emptyLine', 'top:\n    SET lr0 cr0 ;;\n\n    BKPT;;\n\n    ACC.ADD.FIRST;\n    BKPT;;\n', none, none],
  ['line', tidy, none, wordEnds],
  ['none', tidy, none, none],
]) {
  test(`wordSeparation ${mode || 'by default'}: Format Document, on-screen space and drawn lines`, () => withSettings({ wordSeparation: mode }, async () => {
    assert.strictEqual(await formatted(messy), result);
    const lenses = await exec('vscode.executeCodeLensProvider', doc.uri);
    assert.deepStrictEqual(lenses.map((l) => l.range.start.line), gaps());
    assert(lenses.every((l) => l.command.title.trim() === '' || l.command.title === '\u200b'), 'no text');
    assert.deepStrictEqual(editingApi().gapLines(doc), gaps());
    assert.deepStrictEqual(editingApi().separatorLines(doc), drawn());
  }));
}

/** Type `text` in a fresh editor on `start` (cursor at `at`, else the end) until `done(lines)`. */
async function typeIn(start, text, done, at) {
  const editor = await vscode.window.showTextDocument(await withText(start));
  const cursor = at || editor.document.lineAt(0).range.end;
  editor.selection = new vscode.Selection(cursor, cursor);
  const lines = () => editor.document.getText().split('\n');
  await exec('type', { text });
  await until(() => done(lines()), `typing ${JSON.stringify(text)}`);
  return { editor, lines };
}

test('Enter after ;; or a label indents the next line; after ;; it leaves an empty line with "emptyLine"', async () => {
  const { lines } = await typeIn('    SET lr0 cr0;;', '\n', (l) => l.length === 2);
  await sleep(500); // and no late empty line
  assert.deepStrictEqual(lines(), ['    SET lr0 cr0;;', '    ']);
  await withSettings({ wordSeparation: 'emptyLine' }, async () => {
    const { editor, lines } = await typeIn('    SET lr0 cr0;;', '\n', (l) => l.length === 3);
    assert.deepStrictEqual(lines(), ['    SET lr0 cr0;;', '', '    ']);
    assert.strictEqual(editor.selection.active.line, 2);
    // After a label it indents; within a word it adds no empty line, even with "emptyLine".
    const label = (await typeIn('row_loop:', '\n', (l) => l.length === 2)).lines;
    assert.strictEqual(label()[1], '    ');
    await exec('type', { text: 'SET lr0 cr0;\n' });
    await until(() => label().length === 3, 'the next line');
    await sleep(500); // and no late empty line
    assert.deepStrictEqual(label(), ['row_loop:', '    SET lr0 cr0;', '    ']);
  });
});

test('typing ;; before more code moves that code to its own word; a label\'s colon outdents it', async () => {
  const { lines } = await typeIn('SET lr0 cr0 BKPT', ';;', (l) => l.length > 1, new vscode.Position(0, 11));
  assert.deepStrictEqual(lines(), ['    SET lr0 cr0;;', '    BKPT']);
  await typeIn('    top', ':', (l) => l[0] === 'top:');
});

// --- kernel buttons and terminals -----------------------------------------------------------------------

const kernelTerminals = () => vscode.window.terminals.filter((t) => t.name === 'IPU: demo_kernel');
async function closeKernelTerminals() {
  for (const t of kernelTerminals()) t.dispose();
  await until(() => !kernelTerminals().length, 'the kernel terminals to close');
}

test('the title bar buttons appear on kernel files only, and act on the file clicked on', async () => {
  const kernels = api();
  await until(() => kernels.kernelOfFile(doc.uri), 'the manifest to load');
  assert.strictEqual(kernels.kernelOfFile(doc.uri).kernel, 'demo_kernel');
  assert.strictEqual(kernels.kernelOfFile(vscode.Uri.file(scratchFile)), undefined);
  // Each title bar tests its own file (`resourcePath`) against these lists, so they hold exactly the kernels.
  assert.deepStrictEqual(kernels.editorContext.kernelFiles, [doc.uri.fsPath]);
  assert.deepStrictEqual(kernels.editorContext.benchmarkFiles, [doc.uri.fsPath]);
  for (const m of extension().packageJSON.contributes.menus['editor/title']) {
    const list = m.command === 'ipuAsm.benchmarkKernel' ? 'ipuAsm.benchmarkFiles' : 'ipuAsm.kernelFiles';
    assert(m.when.includes(`resourcePath in ${list}`), `${m.command}: ${m.when}`);
  }
  // They act on the file clicked on (VS Code passes its Uri): only a kernel opens a terminal.
  await closeKernelTerminals();
  await exec('ipuAsm.benchmarkKernel', vscode.Uri.file(scratchFile));
  await sleep(1000);
  assert.strictEqual(kernelTerminals().length, 0, 'a file that is no kernel runs nothing');
  await exec('ipuAsm.benchmarkKernel', doc.uri);
  await until(() => kernelTerminals().length === 1, 'the kernel terminal');
});

/** The kernel terminal once its shell is back at its prompt; null if `ps` cannot tell (shell integration decides then). */
async function idleTerminal() {
  const [first] = await until(() => kernelTerminals().length && kernelTerminals(), 'the kernel terminal');
  const pid = await first.processId;
  if ((await runs.atPrompt(pid)) === null) return null;
  await sleep(2000); // past the moment a just-sent command may not have started
  await until(async () => (await runs.atPrompt(pid)) === true, 'the command to end', 120000);
  return first;
}
/** The kernel terminals a moment after running `commands` on the kernel together. */
async function after(...commands) {
  await Promise.all(commands.map((c) => exec(c, 'demo_kernel')));
  await sleep(1000);
  return kernelTerminals();
}

test('Run and Debug start the default case; the case form runs, adds, edits and resets cases in the settings', async () => {
  const config = () => vscode.workspace.getConfiguration('ipuAsm');
  const tail = (command) => command.slice(command.indexOf(' -- '));
  assert.strictEqual(tail(await exec('ipuAsm.runKernel', doc.uri)), ' -- --case default --rows 4');
  assert.strictEqual(tail(await exec('ipuAsm.debugKernel', 'demo_kernel')), ' -- --case default --rows 4');
  const form = await exec('ipuAsm.addCase', 'demo_kernel');
  const send = (command, name, values, original) => form.receive({ command, name, base: 'default', values, original });
  // Run what the form holds, unsaved; values are checked against the option's type.
  assert.strictEqual(tail((await send('run', '', { rows: 9 })).command), ' -- --case default --rows 9');
  assert.match((await send('run', '', { rows: 2.5 })).error, /whole number/);
  assert.match((await send('save', 'wide', { rows: 1 })).error, /already has a case named wide/);
  await send('save', 'tall', { rows: 64 });
  assert.deepStrictEqual(config().get('cases'), { demo_kernel: { tall: { base: 'default', options: { rows: 64 } } } });
  const [family] = await api().tree.getChildren();
  const [kernel] = await api().tree.getChildren(family);
  assert.deepStrictEqual((await api().tree.getChildren(kernel)).map((c) => c.item.label), ['default', 'tall', 'wide']);
  assert.strictEqual(tail(await exec('ipuAsm.runKernel', { kernel: 'demo_kernel', caseName: 'tall' })), ' -- --case default --rows 64');
  // A registry case edited, then reset; a saved case renamed, then deleted.
  await send('save', 'default', { rows: 8 }, 'default');
  assert.strictEqual(tail(await exec('ipuAsm.runKernel', 'demo_kernel')), ' -- --case default --rows 8');
  await send('delete', undefined, {}, 'default');
  assert.strictEqual(tail(await exec('ipuAsm.runKernel', 'demo_kernel')), ' -- --case default --rows 4');
  await send('save', 'taller', { rows: 128 }, 'tall');
  assert.deepStrictEqual(Object.keys(config().get('cases').demo_kernel), ['taller']);
  await send('delete', undefined, {}, 'taller');
  assert.deepStrictEqual(config().get('cases'), {});
  await closeKernelTerminals();
});

test('Test opens a kernel terminal, reused once back at its prompt, even with a background job', async () => {
  await closeKernelTerminals();
  await exec('ipuAsm.testKernel', 'demo_kernel');
  const first = await idleTerminal();
  if (!first) return;
  assert.deepStrictEqual(await after('ipuAsm.testKernel'), [first]);
  await idleTerminal();
  first.sendText('sleep 60 &');
  await idleTerminal();
  assert.deepStrictEqual(await after('ipuAsm.testKernel'), [first]);
});

test('a kernel terminal still running something is never typed into; two runs started together get one each', async () => {
  const first = await idleTerminal();
  if (!first) return;
  first.sendText('sleep 60'); // typed by the user, so shell integration aside
  const pid = await first.processId;
  await until(async () => (await runs.atPrompt(pid)) === false, 'the sleep to start');
  await exec('ipuAsm.testKernel', 'demo_kernel');
  await until(() => kernelTerminals().length === 2, 'a second terminal');
  await closeKernelTerminals();
  await exec('ipuAsm.testKernel', 'demo_kernel');
  await idleTerminal();
  assert.strictEqual((await after('ipuAsm.testKernel', 'ipuAsm.benchmarkKernel')).length, 2, 'the second typed into the first one\'s terminal');
});

// --- sidebar and query ------------------------------------------------------------------

test('the open sidebar follows the editor to the kernel', async () => {
  await exec('workbench.view.extension.ipu');
  await vscode.window.showTextDocument(await withText('BKPT;;\n'));
  await vscode.window.showTextDocument(doc);
  await until(() => api().view.selection.length && api().view.selection[0].kernel === 'demo_kernel', 'the kernel row selected');
});

test('the sidebar shows family, kernel and cases from the manifest; a query returns the registry verdict', async () => {
  const { tree } = api();
  const [family] = await tree.getChildren();
  assert.deepStrictEqual([family.item.label, family.item.contextValue], ['demo', 'ipuFamily']);
  const [kernel] = await tree.getChildren(family);
  // Rows with children can be expanded (the tree reads this, not getChildren).
  for (const row of [family, kernel]) assert.strictEqual(row.item.collapsibleState, vscode.TreeItemCollapsibleState.Collapsed, row.item.label);
  assert.deepStrictEqual([kernel.item.label, kernel.item.resourceUri.fsPath], ['demo_kernel', kernelFile]);
  assert.match(kernel.item.contextValue, /ipuKernel/);
  assert.match(kernel.item.contextValue, /ipuBenchmark/);
  assert.deepStrictEqual((await tree.getChildren(kernel)).map((c) => c.item.label), ['default', 'wide']);
  const yes = await api().runQuery('identity', 'shape=4,128');
  assert.deepStrictEqual([yes.supported, yes.app], [true, 'demo_kernel']);
  assert.strictEqual((await api().runQuery('identity', 'shape=5,5')).supported, false);
});

// --- the checker, when there is one -----------------------------------------------------

if (process.env.IPU_TEST_HAS_CHECKER) {
  test('diagnostics come from the assembler; a rename goes through only when it accepts the result', async () => {
    const d = await vscode.window.showTextDocument(await withText('BEQ lr0, cr0, +1;;\n')).then((e) => e.document);
    const diags = await until(() => vscode.languages.getDiagnostics(d.uri).length && vscode.languages.getDiagnostics(d.uri), 'a diagnostic for the comma');
    assert(diags[0].source.startsWith('ipu-as'));
    // `loop` is unused, but inside the loop it is Jinja's loop object: only the assembler knows.
    const l = await withText(loopText);
    await assert.rejects(rename(l.uri, new vscode.Position(0, 8), 'loop'), /would not assemble/);
    assert.strictEqual((await rename(l.uri, new vscode.Position(0, 8), 'row')).get(l.uri).length, 2);
  });
}

// Last, since it swaps the checker out.
test('a checker that hangs is stopped after ipuAsm.timeoutSeconds, and rename goes on', () =>
  withSettings({ checkCommand: ['sh', '-c', 'sleep 60; :', 'hang'], timeoutSeconds: 1 }, async () => {
    await vscode.window.showTextDocument(doc);
    let started = Date.now();
    await exec('ipuAsm.check');
    await until(() => /timed out after 1s/.test(extension().exports.checkerError() || ''), 'the check to time out', 15000);
    assert(Date.now() - started < 10000, `stopped after ${Date.now() - started}ms`);
    started = Date.now();
    const edit = await rename(doc.uri, pos('row_loop:', 1), 'row_loop2');
    assert(edit.get(doc.uri).length >= 2, 'renamed on the lexical check');
    assert(Date.now() - started < 10000, `rename waited ${Date.now() - started}ms`);
  }));

async function run() {
  doc = await vscode.workspace.openTextDocument(kernelFile);
  await vscode.window.showTextDocument(doc);
  assert.strictEqual(doc.languageId, 'ipu-asm');
  await runAll('integration');
}

module.exports = { run };
