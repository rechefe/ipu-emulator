'use strict';

// The case form: a kernel's cases in a webview, one field per option (typed as the runner
// parses it), to run or debug with, save, or add to. Saved cases go to ipuAsm.cases in the
// folder's settings, as { kernel: { case: { base, options } } }: the registry's cases.py is
// never written, and a saved case with a registry case's name is that case edited.

const crypto = require('crypto');
const vscode = require('vscode');
const runs = require('./runs');

/** The sidebar view the form lives in; the kernel it shows ({ index, kernelName, run }); the
 *  case to show when it is (re)filled. */
const VIEW = 'ipuAsm.caseForm';
let view;
let shown = null;
let select;

const savedCases = (index) => vscode.workspace.getConfiguration('ipuAsm', index.folder.uri).get('cases') || {};

/** What the form shows: each case with its options and values, and which can be a base. */
function state(index, kernelName) {
  const kernel = index.manifest && index.manifest.kernels[kernelName];
  if (!kernel) return null;
  const cases = runs.kernelCases(kernel, savedCases(index)[kernelName]);
  const view = (c) => ({
    base: c.base, saved: Boolean(c.saved), builtIn: c.builtIn,
    options: Object.entries(c.entry.options).map(([name, o]) => ({
      name, flag: o.flag, type: runs.optionType(o), default: o.default, value: Object.hasOwn(c.values, name) ? c.values[name] : o.default,
    })),
  });
  return {
    cases: Object.fromEntries(Object.entries(cases).map(([name, c]) => [name, view(c)])),
    bases: Object.fromEntries(Object.keys(kernel.cases).map((name) => [name, view(cases[name].saved ? { ...cases[name], values: {} } : cases[name])])),
  };
}

/** Show the form for `kernelName` in the sidebar, on `caseName`, or on a new case when
 *  `adding`. `run(action, kernelName, choice)` runs a case the form describes. */
async function open(index, kernelName, { caseName = 'default', adding = false, run }) {
  shown = { index, kernelName, run };
  select = adding ? '' : caseName;
  await vscode.commands.executeCommand('setContext', 'ipuAsm.caseFormOpen', true);
  await vscode.commands.executeCommand(`${VIEW}.focus`);
  update(true);
  return { receive };
}

/** Send the form its state, and with `reselect` the case to show (else it keeps its own). */
function update(reselect) {
  if (!view || !shown) return;
  view.description = shown.kernelName;
  view.webview.postMessage({ state: state(shown.index, shown.kernelName), ...(reselect && { select }) });
}

/** The sidebar view, filled once its page has loaded (and kept while hidden). */
const provider = {
  resolveWebviewView(webviewView) {
    view = webviewView;
    view.webview.options = { enableScripts: true };
    view.webview.html = html(view.webview);
    view.webview.onDidReceiveMessage((m) => (m.ready ? update(true) : receive(m)));
    view.onDidDispose(() => (view = undefined));
  },
};

/** Act on a button: `{ command, name, base, values, original }` from the form. Resolves to
 *  what it did, for the tests: the command run, or the error shown. */
async function receive({ command, name, base, values, original }) {
  if (!shown) return undefined;
  const { index, kernelName, run } = shown;
  const fail = (error) => {
    if (view) view.webview.postMessage({ error });
    return { error };
  };
  const kernel = index.manifest && index.manifest.kernels[kernelName];
  if (!kernel) return fail(`${kernelName} is not in the manifest any more.`);
  if (command === 'delete') return save(index, kernelName, (cases) => delete cases[original], '');
  const entry = kernel.cases[base];
  if (!entry) return fail(`${kernelName} has no case ${base} to start from.`);
  const invalid = runs.invalidValues(entry, values);
  if (invalid) return fail(invalid);
  if (command === 'run' || command === 'debug') return { command: await run(command, kernelName, { base, entry, values }) };
  name = String(name || '').trim();
  if (!/^[\w.-]+$/.test(name)) return fail('A case name is letters, digits, _ . or -.');
  if (name !== original && runs.kernelCases(kernel, savedCases(index)[kernelName])[name]) return fail(`${kernelName} already has a case named ${name}.`);
  return save(index, kernelName, (cases) => {
    if (original && original !== name) delete cases[original];
    cases[name] = { base, options: values };
  }, name);
}

/** Change this kernel's saved cases with `edit`, write them back, and show `select`. */
async function save(index, kernelName, edit, selected) {
  const config = vscode.workspace.getConfiguration('ipuAsm', index.folder.uri);
  const all = JSON.parse(JSON.stringify(savedCases(index)));
  const cases = (all[kernelName] = all[kernelName] || {});
  edit(cases);
  if (!Object.keys(cases).length) delete all[kernelName];
  await config.update('cases', Object.keys(all).length ? all : undefined, vscode.ConfigurationTarget.WorkspaceFolder);
  select = selected || 'default';
  update(true);
  return { saved: all };
}

/** Register the sidebar view. */
const register = (context) => context.subscriptions.push(
  vscode.window.registerWebviewViewProvider(VIEW, provider, { webviewOptions: { retainContextWhenHidden: true } }));

function html(webview) {
  const nonce = crypto.randomBytes(16).toString('base64');
  return `<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
  body { font: var(--vscode-font-size) var(--vscode-font-family); color: var(--vscode-foreground); padding: 4px 12px; }
  label { display: flex; flex-direction: column; gap: 3px; margin: 8px 0; }
  input, select { font: inherit; color: var(--vscode-input-foreground); background: var(--vscode-input-background);
    border: 1px solid var(--vscode-input-border, var(--vscode-panel-border)); padding: 3px 5px; min-width: 0; }
  label.check { flex-direction: row; align-items: center; }
  small { color: var(--vscode-descriptionForeground); margin-left: 4px; }
  .buttons { position: sticky; bottom: 0; padding: 4px 0 6px; background: var(--vscode-sideBar-background); }
  fieldset { border: 1px solid var(--vscode-panel-border); margin: 12px 0; padding: 4px 10px; }
  button { font: inherit; border: none; padding: 5px 12px; margin: 4px 6px 0 0; cursor: pointer;
    color: var(--vscode-button-foreground); background: var(--vscode-button-background); }
  button.secondary { color: var(--vscode-button-secondaryForeground); background: var(--vscode-button-secondaryBackground);
    outline: 1px solid var(--vscode-button-border, var(--vscode-panel-border)); }
  [hidden] { display: none !important; }
  #error { color: var(--vscode-errorForeground); min-height: 1.4em; }
</style></head><body>
<form id="form" hidden>
  <label>Case <select id="case"></select></label>
  <label id="nameRow">Name <input id="name" autocomplete="off"></label>
  <label id="baseRow">Start from <select id="base"></select></label>
  <fieldset><legend>Options</legend><div id="options"></div></fieldset>
  <p id="error" role="alert"></p>
  <div class="buttons"><button data-command="run">Run</button><button data-command="debug">Debug</button>
  <button data-command="save" class="secondary">Save</button><button data-command="delete" class="secondary" id="delete" hidden></button></div>
</form>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
const $ = (id) => document.getElementById(id);
let data, selected;
const option = (text, value) => Object.assign(document.createElement('option'), { textContent: text, value });

function show(name) {
  selected = name;
  const c = data.cases[name];
  $('case').replaceChildren(...Object.keys(data.cases).map((n) => option(n + (data.cases[n].saved ? ' (saved)' : ''), n)), option('New case…', ''));
  $('case').value = name;
  $('name').value = c ? name : '';
  $('nameRow').hidden = c && c.builtIn;
  $('baseRow').hidden = Boolean(c);
  $('base').replaceChildren(...Object.keys(data.bases).map((n) => option(n, n)));
  $('base').value = c ? c.base : 'default' in data.bases ? 'default' : Object.keys(data.bases)[0];
  fields(c || data.bases[$('base').value]);
  $('delete').hidden = !(c && c.saved);
  $('delete').textContent = c && c.builtIn ? 'Reset to registry' : 'Delete';
  $('error').textContent = '';
}

function fields(c) {
  $('options').replaceChildren(...(c.options.length ? c.options.map((o) => {
    const input = document.createElement('input');
    input.name = o.name;
    input.dataset.type = o.type;
    if (o.type === 'bool') Object.assign(input, { type: 'checkbox', checked: o.value });
    else Object.assign(input, { type: o.type === 'str' ? 'text' : 'number', step: o.type === 'int' ? '1' : 'any', value: String(o.value) });
    const row = Object.assign(document.createElement('label'), { title: o.flag, className: o.type === 'bool' ? 'check' : '' });
    const name = document.createElement('span');
    name.append(o.name, Object.assign(document.createElement('small'), { textContent: o.type + ', default ' + o.default }));
    row.append(...(o.type === 'bool' ? [input, name] : [name, input]));
    return row;
  }) : [Object.assign(document.createElement('p'), { textContent: 'This case has no options.' })]));
}

function values() {
  return Object.fromEntries([...$('options').querySelectorAll('input')].map((i) =>
    [i.name, i.dataset.type === 'bool' ? i.checked : i.dataset.type === 'str' ? i.value : i.value.trim() === '' ? NaN : Number(i.value)]));
}

$('case').onchange = () => show($('case').value);
$('base').onchange = () => fields(data.bases[$('base').value]);
$('form').onsubmit = (e) => {
  e.preventDefault();
  const c = data.cases[selected];
  vscode.postMessage({ command: e.submitter.dataset.command, name: c && c.builtIn ? selected : $('name').value,
    base: c ? c.base : $('base').value, values: values(), original: c ? selected : undefined });
};
window.addEventListener('message', ({ data: m }) => {
  if (m.error !== undefined) { $('error').textContent = m.error; return; }
  if (m.state) data = m.state;
  if (!data) return;
  $('form').hidden = false;
  show(m.select !== undefined ? m.select : selected in data.cases ? selected : 'default');
});
// Only now can it be sent anything: a message posted before the page loaded is lost.
vscode.postMessage({ ready: true });
</script></body></html>`;
}

module.exports = { open, refresh: () => update(false), register, savedCases };
