'use strict';

// Run, Debug, Test and Benchmark buttons for kernel files, and the IPU sidebar. Both
// read the kernel manifest the workspace itself produces (Bazel's targets joined with
// the kernel registry's cases), so a new kernel shows up with nothing edited here.

const fs = require('fs').promises;
const path = require('path');
const vscode = require('vscode');

const caseform = require('./caseform');
const checker = require('./checker');
const project = require('./project');
const runs = require('./runs');
const trust = require('./trust');

/** Debounce file-change refreshes: a branch switch touches many files. */
const REFRESH_DEBOUNCE_MS = 1500;

/** Setting ipuAsm.`key` for workspace folder `folder`, or `fallback` when it is not of its type. */
function setting(folder, key, fallback) {
  const value = vscode.workspace.getConfiguration('ipuAsm', folder && folder.uri).get(key);
  return (Array.isArray(fallback) ? Array.isArray(value) : typeof value === typeof fallback) ? value : fallback;
}

const timeoutSeconds = (folder) => setting(folder, 'timeoutSeconds', 300);

/** The last few lines of a failure, for a message. */
const tail = (text) => text.trim().split('\n').slice(-4).join('\n').slice(0, 600);

/** One checkout's manifest, reloaded on demand and on change. `folder` is the workspace
 *  folder it was found from (for settings); `root` the checkout (in, around or below it). */
class KernelIndex {
  constructor(folder, root) {
    this.folder = folder;
    this.root = root;
    this.name = root === folder.uri.fsPath ? folder.name : path.basename(root);
    this.manifest = null;
    /** The Bazel workspace the manifest's paths are relative to. */
    this.workspace = root;
    this.error = null;
    /** The kernels root (runs.kernelsRoot) of the last manifest that loaded: kept while a
     *  reload fails, so fixing what broke it still reloads it. */
    this.kernelsRoot = null;
    this.loading = false;
    this._again = false;
    this.disposed = false;
    this._changed = new vscode.EventEmitter();
    this.onDidChange = this._changed.event;
  }

  /** A refresh requested while one runs is run once after it, so the latest tree wins. */
  async refresh() {
    if (this.disposed) return;
    if (this.loading) {
      this._again = true;
      return;
    }
    if (!trust.mayRun()) {
      // Restricted Mode: the manifest command is Bazel in this folder, or the folder's own setting.
      this.manifest = null;
      this.error = `${this.name}'s kernels are not listed: ${trust.RESTRICTED}`;
      this._changed.fire();
      return;
    }
    this.loading = true;
    // The first load shows as loading; a reload keeps showing the last list.
    if (!this.manifest) this._changed.fire();
    const configured = setting(this.folder, 'manifestCommand', []);
    const [command, ...args] = configured.length ? configured : runs.manifestCommand();
    const result = await checker.run(command, args, this.root, timeoutSeconds(this.folder));
    const before = this.manifest && this._stdout;
    try {
      if (result.code !== 0) throw new Error(tail(result.stderr) || `exit ${result.code}`);
      const manifest = JSON.parse(result.stdout);
      const workspace = await this.workspaceOf(manifest.workspace);
      if (this.disposed) return; // the checkout went while it loaded
      this.workspace = workspace;
      this.manifest = manifest;
      this._stdout = result.stdout;
      this.kernelsRoot = runs.kernelsRoot(manifest);
      this.error = null;
    } catch (err) {
      this.manifest = null;
      // A checkout predating the manifest says so, not how Bazel failed to find it.
      this.error = runs.manifestMissing(result.stderr)
        ? `${this.name} has no kernel manifest (${runs.MANIFEST_TARGET}): this version of the repository predates it, so its kernels are not listed.`
        : `Could not load the kernel manifest (${[command, ...args].join(' ')}): ${err.message}`;
      console.warn(`[ipu-asm] ${this.error}`);
    }
    this.loading = false;
    // Nothing to redraw when a reload says what the last one did.
    if (!(before && before === this._stdout && this.manifest)) this._changed.fire();
    if (this._again) {
      this._again = false;
      this.refresh();
    }
  }

  /** The manifest's workspace under the checkout as opened: Bazel names it by its real
   *  path, VS Code by the path opened, and through a symlink the two differ. */
  async workspaceOf(workspace) {
    if (!workspace) return this.root;
    // A folder that is gone has only the manifest's own path.
    const real = await fs.realpath(this.root).catch(() => this.root);
    return path.resolve(this.root, path.relative(real, workspace));
  }

  /** True when `fsPath` could change the manifest: anything under the kernels, or any build
   *  file of the workspace. Before a manifest has said where the kernels are, build files only
   *  (and the Refresh button): every Python file of the workspace would rerun a failing load. */
  affects(fsPath, buildFile = false) {
    if (buildFile) {
      // Not Bazel's own output trees, which hold copies of everything.
      const rel = project.within(this.workspace, fsPath);
      return Boolean(rel) && !rel.split(path.sep)[0].startsWith('bazel-');
    }
    return this.kernelsRoot !== null && Boolean(project.within(path.join(this.workspace, this.kernelsRoot), fsPath));
  }

  /** Stop: a checkout no longer open (its folder removed, or no checkout any more). */
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.manifest = null;
    this._changed.dispose();
  }
}

// --- Running ---

const terminals = new Map();
/** Terminals running a command we sent, until shell integration reports its end. */
const busy = new Set();
/** When each terminal was last sent a command: until the shell starts it, it looks idle. */
const sentAt = new Map();
const STARTING_MS = 1500;

/** Whether `terminal` is at its prompt, so a command goes to the shell, not to a running
 *  program (the terminal debugger reads the keyboard). The shell must hold the terminal's
 *  foreground (runs.atPrompt); with shell integration our last command must also be reported
 *  finished (covering a shell still starting); where `ps` cannot tell, only the report counts. */
async function isIdle(terminal) {
  if (terminal.exitStatus !== undefined || Date.now() - (sentAt.get(terminal) || 0) < STARTING_MS) return false;
  // The shell's process id, or undefined if it does not say soon.
  const prompt = await runs.atPrompt(await Promise.race([terminal.processId, new Promise((r) => setTimeout(r, 2000))]));
  if (terminal.exitStatus !== undefined) return false;
  if (terminal.shellIntegration) return !busy.has(terminal) && prompt !== false;
  return prompt === null ? !busy.has(terminal) : prompt;
}

/** Runs are sent one at a time, so two started together (a double click, or
 *  Debug then Test) do not both find the same terminal idle. */
let sending = Promise.resolve();

/** Send `command` to the terminal for this folder and `name`, reused whenever it is idle.
 *  Resolves to whether it was sent: not in Restricted Mode. */
function sendToTerminal(index, name, command) {
  const send = async () => {
    if (!trust.mayRun(name)) return false;
    const key = `${index.root}\u0000${name}`;
    let terminal = terminals.get(key);
    if (!terminal || !(await isIdle(terminal))) {
      terminal = vscode.window.createTerminal({
        name: index.siblings.size > 1 ? `IPU: ${name} (${index.name})` : `IPU: ${name}`,
        cwd: index.workspace,
      });
      terminals.set(key, terminal);
    }
    terminal.show();
    busy.add(terminal);
    sentAt.set(terminal, Date.now());
    terminal.sendText(command);
    return true;
  };
  return (sending = sending.then(send, send));
}

/** `build()`'s command, or undefined after showing why a manifest value made it refuse. */
function commandOrReport(name, build) {
  try {
    return build();
  } catch (err) {
    vscode.window.showErrorMessage(`IPU: ${name}: ${err.message}`);
    return undefined;
  }
}

/** The checkout to run `kernelName` from: `folder` (an index key) if it has it, else the
 *  first that does. */
const indexOf = (indexes, kernelName, folder) =>
  [indexes.get(folder), ...indexes.values()].find((i) => i && i.manifest && i.manifest.kernels[kernelName]);

/** Run, test, debug or benchmark a kernel, with the case named `caseName` (a registry or a
 *  saved one) or the one the case form describes (`choice`: { base, entry, values }).
 *  Resolves to the command sent, if one was. */
async function runKernel(indexes, action, kernelName, caseName = 'default', folder, choice) {
  const index = indexOf(indexes, kernelName, folder);
  if (!index) {
    vscode.window.showWarningMessage(`IPU: no kernel named ${kernelName} in the manifest.`);
    return undefined;
  }
  const kernel = index.manifest.kernels[kernelName];
  const chosen = choice || runs.kernelCases(kernel, caseform.savedCases(index)[kernelName])[caseName];
  if (!chosen) {
    vscode.window.showWarningMessage(`IPU: ${kernelName} has no case ${caseName}.`);
    return undefined;
  }
  const command = commandOrReport(kernelName, () => runs.kernelCommand(kernel, action, {
    caseName: chosen.base, options: runs.optionArgs(chosen.entry, chosen.values), flags: setting(index.folder, 'bazelFlags', []),
  }));
  if (command === undefined) return undefined;
  if (!command) {
    vscode.window.showWarningMessage(`IPU: ${kernelName} has no ${action} target.`);
    return undefined;
  }
  return (await sendToTerminal(index, kernelName, command)) ? command : undefined;
}

// --- Sidebar ---

/** A tree node: `data.children` (a function) makes it collapsible, `props` sets the item's fields. */
function node(label, icon, props, data = {}) {
  const item = new vscode.TreeItem(label, data.children ? vscode.TreeItemCollapsibleState.Collapsed : vscode.TreeItemCollapsibleState.None);
  if (icon) item.iconPath = new vscode.ThemeIcon(icon);
  return { item: Object.assign(item, props), ...data };
}

const message = (text, icon, command) => node(text, icon, { tooltip: text, command: command && { command, title: 'Retry' } });

function roots(indexes) {
  if (!indexes.size) {
    return [message('No IPU checkout is open: open the repository (or a folder in or around it) to see its kernels.', 'info')];
  }
  if (indexes.size === 1) return folderNodes([...indexes.values()][0]);
  return [...indexes.values()].map((index) => {
    const root = node(index.name, null, { id: index.root, collapsibleState: vscode.TreeItemCollapsibleState.Expanded },
      { root: index.root, children: () => folderNodes(index, root) });
    return root;
  });
}

/** Ids let `reveal` find a row again: the tree is rebuilt on every change. */
function folderNodes(index, parent) {
  const { manifest, error } = index;
  if (index.loading && !manifest) return [message('Loading kernels…', 'loading~spin')];
  if (error && !manifest) return [message(error, 'error', 'ipuAsm.refreshKernels')];
  if (!manifest) return [];
  const nodes = runs.familyTree(manifest).map(({ family, entry, kernels }) => {
    const row = node(family, 'folder-library', {
      id: `${index.root}|${family}`,
      description: `${kernels.length}`,
      contextValue: 'ipuFamily',
      tooltip: entry && (entry.test ? `bazel test ${entry.suite} (includes ${entry.test})` : `bazel test ${entry.suite}`),
    }, { family, entry, folder: index.root, parent, kernels, children: () => kernels.map((k) => kernelNode(index, k, row)) });
    return row;
  });
  const { skipped } = manifest;
  if (skipped.length) {
    nodes.push(node(`${skipped.length} module(s) skipped during discovery`, 'warning', {}, {
      children: () => skipped.map((s) => node(s.module, null, { description: s.error, tooltip: s.error })),
    }));
  }
  if (error) nodes.unshift(message(error, 'warning', 'ipuAsm.refreshKernels'));
  return nodes;
}

/** A case's options as the sidebar shows them; a manifest flag that could not be run says so. */
function describeOptions(c) {
  try {
    return runs.optionArgs(c.entry, c.values);
  } catch (err) {
    return err.message;
  }
}

function kernelNode(index, name, parent) {
  const kernel = index.manifest.kernels[name];
  const resourceUri = vscode.Uri.file(path.join(index.workspace, kernel.asm));
  const about = [`**${name}**`, `operation \`${kernel.op}\``, kernel.tags.length && `tags: ${kernel.tags.join(', ')}`, `\`${kernel.asm}\``];
  const row = node(name, 'circuit-board', {
    id: `${parent.item.id}|${name}`,
    description: kernel.variant ? `${kernel.op} · ${kernel.variant}` : kernel.op,
    tooltip: new vscode.MarkdownString(about.filter(Boolean).join('\n\n')),
    contextValue: kernel.targets.benchmark ? 'ipuKernel ipuBenchmark' : 'ipuKernel',
    resourceUri,
    command: { command: 'vscode.open', title: 'Open', arguments: [resourceUri] },
  }, {
    kernel: name,
    folder: index.root,
    parent,
    children: () => Object.entries(runs.kernelCases(kernel, caseform.savedCases(index)[name])).map(([caseName, c]) => node(caseName, c.saved ? 'symbol-property' : 'symbol-event', {
      id: `${row.item.id}|${caseName}`,
      description: describeOptions(c),
      tooltip: [c.saved && `saved in ipuAsm.cases${c.base === caseName ? '' : `, from ${c.base}`}`, `max cycles ${c.entry.max_cycles}`].filter(Boolean).join('; '),
      contextValue: 'ipuCase',
    }, { kernel: name, caseName, folder: index.root, parent: row })),
  });
  return row;
}

// --- Query ---

/** Ask the registry which kernel handles `op` with `params`: the verdict (query --json),
 *  with the exchange logged to `output`. */
async function runQuery(index, op, params, output) {
  if (!trust.mayRun()) throw new Error(trust.RESTRICTED);
  const argv = runs.queryCommand(index.manifest, op, params, setting(index.folder, 'queryCommand', []));
  const [command, ...args] = argv;
  output.appendLine(`$ ${argv.map(runs.shellQuote).join(' ')}`);
  const result = await checker.run(command, args, index.root, timeoutSeconds(index.folder));
  try {
    // Exit 1 means "not supported", which is still an answer.
    const verdict = JSON.parse(result.stdout);
    output.appendLine(JSON.stringify(verdict, null, 2));
    return verdict;
  } catch {
    output.appendLine(result.stderr || result.stdout);
    throw new Error(tail(result.stderr) || `exit ${result.code}`);
  }
}

/** The index of the checkout holding `fsPath`: the innermost one. */
function indexFor(indexes, fsPath) {
  const holding = [...indexes.values()].filter((i) => i.root === fsPath || project.within(i.root, fsPath));
  return holding.sort((a, b) => b.root.length - a.root.length)[0] || null;
}

async function query(indexes, output, state) {
  // The active editor's checkout, else the first that has loaded.
  const editor = vscode.window.activeTextEditor;
  const own = editor && indexFor(indexes, editor.document.uri.fsPath);
  const index = own && own.manifest ? own : [...indexes.values()].find((i) => i.manifest);
  if (!index) {
    vscode.window.showWarningMessage('IPU: the kernel manifest has not loaded yet.');
    return;
  }
  const { operations } = index.manifest;
  const picked = await vscode.window.showQuickPick(
    Object.entries(operations).map(([label, { params }]) => ({ label, description: params.join(' ') })),
    { title: 'IPU query: which operation?', matchOnDescription: true }
  );
  if (!picked) return;
  const op = picked.label;
  const memoryKey = `ipuAsm.query.${op}`;
  const params = await vscode.window.showInputBox({
    title: `IPU query: ${op}`,
    prompt: 'Parameters as name=value, separated by spaces',
    value: state.get(memoryKey) ?? operations[op].params.map((p) => `${p}=`).join(' '),
  });
  if (params === undefined) return;
  await state.update(memoryKey, params);
  let verdict;
  try {
    verdict = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: `IPU: querying ${op}…` },
      () => runQuery(index, op, params, output)
    );
  } catch (err) {
    output.show(true);
    vscode.window.showErrorMessage(`IPU query failed: ${err.message}`);
    return;
  }
  const actions = verdict.app ? ['Open kernel', 'Run', 'Details'] : ['Details'];
  const show = verdict.supported ? vscode.window.showInformationMessage : vscode.window.showWarningMessage;
  const choice = await show(runs.describeVerdict(verdict), ...actions);
  if (choice === 'Details') output.show(true);
  if (choice === 'Run') runKernel(indexes, 'run', verdict.app, undefined, index.root);
  if (choice === 'Open kernel') {
    const kernel = index.manifest.kernels[verdict.app];
    if (kernel) vscode.window.showTextDocument(vscode.Uri.file(path.join(index.workspace, kernel.asm)));
  }
}

// --- Wiring ---

/** The IPU sidebar and the kernel commands, over one KernelIndex per checkout the open
 *  folders stand for. */
class Kernels {
  constructor(context) {
    this.context = context;
    this.indexes = new Map();
    /** Debounced manifest reloads, by index. */
    this.timers = new Map();
    this.changed = new vscode.EventEmitter();
    this.tree = {
      onDidChangeTreeData: this.changed.event,
      refresh: () => this.changed.fire(),
      getTreeItem: (n) => n.item,
      getChildren: (n) => (!n ? roots(this.indexes) : n.children ? n.children() : []),
      getParent: (n) => n.parent,
    };
    this.output = vscode.window.createOutputChannel('IPU Query');
    this.view = vscode.window.createTreeView('ipuKernels', { treeDataProvider: this.tree, showCollapseAll: true });
    // Title bar buttons show on kernel files (Benchmark where there is one). Each editor group
    // has its own title bar, so the keys list every such file rather than describe the active
    // editor. They hold paths, matched against `resourcePath`: `resource` is the Uri as the
    // window sees it, which in a remote window (WSL, SSH) is vscode-remote://, never file://.
    this.editorContext = { kernelFiles: [], benchmarkFiles: [] };
    /** Scans of the open folders, one at a time, so two never add the same checkout. */
    this.scanning = Promise.resolve();
    this.rescanTimer = undefined;
  }

  /** The sidebar follows the editor: a kernel's .asm selects its row, while the sidebar is open. */
  follow() {
    const editor = vscode.window.activeTextEditor;
    const found = this.view.visible && editor && this.kernelOfFile(editor.document.uri);
    if (!found) return;
    const index = this.indexes.get(found.folder);
    const top = this.indexes.size > 1 ? roots(this.indexes).find((r) => r.root === index.root).children() : folderNodes(index);
    const family = top.find((f) => f.kernels && f.kernels.includes(found.kernel));
    if (family) this.view.reveal(kernelNode(index, found.kernel, family), { select: true, focus: false }).then(undefined, () => {});
  }

  redraw() {
    this.tree.refresh();
    this.updateEditorContext();
    caseform.refresh();
    this.follow();
  }

  /** One index per checkout the open folders stand for; a folder with none (another project,
   *  or a version predating the Python assembler) gets none: nothing runs for it. */
  async addFolders() {
    for (const folder of vscode.workspace.workspaceFolders || []) {
      for (const root of await project.checkoutsFor(folder.uri.fsPath)) {
        if (this.indexes.has(root)) continue;
        const index = new KernelIndex(folder, root);
        index.siblings = this.indexes;
        this.indexes.set(root, index);
        this.context.subscriptions.push(index);
        index.onDidChange(() => this.redraw());
        // Launched after activation returns: starting a process blocks briefly.
        setImmediate(() => index.refresh());
      }
    }
  }

  /** Which checkouts the folders stand for can change: another version can bring in the
   *  assembler or take it away, and folders come and go. */
  rescan() {
    const scan = async () => {
      const before = [...this.indexes.keys()].join('\n');
      const open = new Set((vscode.workspace.workspaceFolders || []).map((f) => f.uri.toString()));
      for (const [root, index] of this.indexes) {
        if (open.has(index.folder.uri.toString()) && await project.isCheckout(root)) continue;
        this.indexes.delete(root);
        clearTimeout(this.timers.get(index));
        this.timers.delete(index);
        index.dispose();
        const { subscriptions } = this.context;
        if (subscriptions.includes(index)) subscriptions.splice(subscriptions.indexOf(index), 1);
      }
      await this.addFolders();
      if ([...this.indexes.keys()].join('\n') !== before) this.redraw();
    };
    return (this.scanning = this.scanning.then(scan, scan));
  }

  scheduleRescan() {
    clearTimeout(this.rescanTimer);
    this.rescanTimer = setTimeout(() => this.rescan(), REFRESH_DEBOUNCE_MS);
  }

  /** A handler reloading the manifest of the checkout a changed file can change, debounced. */
  onFileChange(buildFile) {
    return (uri) => {
      const index = indexFor(this.indexes, uri.fsPath);
      if (!index || !index.affects(uri.fsPath, buildFile)) return;
      clearTimeout(this.timers.get(index));
      this.timers.set(index, setTimeout(() => index.refresh(), REFRESH_DEBOUNCE_MS));
    };
  }

  watchFiles() {
    const watch = (glob, events, ...handlers) => {
      const watcher = vscode.workspace.createFileSystemWatcher(glob, ...['Create', 'Change', 'Delete'].map((e) => !events.includes(e)));
      for (const e of events) for (const handler of handlers) watcher[`onDid${e}`](handler);
      this.context.subscriptions.push(watcher);
    };
    const rescan = () => this.scheduleRescan();
    // What the manifest is made of: a kernel's .asm only by existing (its name is its folder's, so
    // an edit does not reload); the Python under the kernels (the registry's modules) by content;
    // build files' targets.
    watch('**/*.asm', ['Create', 'Delete'], this.onFileChange(false));
    watch('**/*.py', ['Create', 'Change', 'Delete'], this.onFileChange(false));
    watch('**/{BUILD,BUILD.bazel,*.bzl,MODULE.bazel,WORKSPACE,WORKSPACE.bazel}', ['Create', 'Change', 'Delete'], this.onFileChange(true), rescan);
    // The assembler's package coming or going makes or unmakes a checkout. VS Code reports a
    // new folder tree by its topmost folder only, so any create or delete is a reason to look
    // again (a few stat calls), as is coming back to the window or to another editor.
    watch('**/*', ['Create', 'Delete'], rescan);
  }

  /** The kernel whose .asm `uri` is, with its checkout, or undefined. */
  kernelOfFile(uri) {
    for (const index of this.indexes.values()) {
      const found = runs.kernelForFile(index.manifest, index.workspace, uri.fsPath);
      if (found) return { kernel: found.name, folder: index.root, entry: found.kernel };
    }
    return undefined;
  }

  updateEditorContext() {
    const files = [...this.indexes.values()].flatMap((index) => Object.values(index.manifest ? index.manifest.kernels : {})
      .map((kernel) => ({ kernel, file: path.resolve(index.workspace, kernel.asm) })));
    const lists = { kernelFiles: files.map((f) => f.file), benchmarkFiles: files.filter((f) => f.kernel.targets.benchmark).map((f) => f.file) };
    // Sent to the window only when they change: a sidebar redraw mostly leaves them as they were.
    for (const [key, list] of Object.entries(lists)) {
      if (list.join('\n') === this.editorContext[key].join('\n')) continue;
      this.editorContext[key] = list;
      vscode.commands.executeCommand('setContext', `ipuAsm.${key}`, list);
    }
  }

  /** What a command acts on: a kernel name, a sidebar row ({ kernel, caseName, folder }), the
   *  file's Uri from the title bar, or nothing (from the palette): the kernel in the editor. */
  targetOf(arg) {
    const editor = vscode.window.activeTextEditor;
    const target = typeof arg === 'string'
      ? { kernel: arg }
      : arg instanceof vscode.Uri
        ? this.kernelOfFile(arg)
        : (arg && arg.kernel ? arg : editor && this.kernelOfFile(editor.document.uri));
    if (!target) vscode.window.showWarningMessage('IPU: open a kernel .asm, or pick a kernel in the IPU sidebar.');
    return target;
  }

  /** The command handlers, by name under `ipuAsm.`. */
  commands() {
    const { indexes } = this;
    const action = (name) => (arg) => {
      const target = this.targetOf(arg);
      return target && runKernel(indexes, name, target.kernel, target.caseName, target.folder);
    };
    /** The case form for the kernel `arg` names, on its case (or a new one when `adding`). */
    const cases = (adding) => (arg) => {
      const target = this.targetOf(arg);
      const index = target && indexOf(indexes, target.kernel, target.folder);
      return index && caseform.open(index, target.kernel, {
        caseName: target.caseName, adding,
        run: (name, kernelName, choice) => runKernel(indexes, name, kernelName, undefined, index.root, choice),
      });
    };
    return {
      runKernel: action('run'),
      testKernel: action('test'),
      debugKernel: action('debug'),
      benchmarkKernel: action('benchmark'),
      editCases: cases(false),
      addCase: cases(true),
      testFamily: (n) => {
        const index = n && n.entry ? indexes.get(n.folder) : undefined;
        if (!index) return undefined;
        const command = commandOrReport(n.family, () => runs.familyCommand(n.entry, { flags: setting(index.folder, 'bazelFlags', []) }));
        return command && sendToTerminal(index, n.family, command);
      },
      refreshKernels: () => indexes.forEach((index) => index.refresh()),
      query: () => query(indexes, this.output, this.context.workspaceState),
    };
  }

  /** Register everything; resolves once the open folders' checkouts are found. */
  start() {
    this.watchFiles();
    this.updateEditorContext();
    caseform.register(this.context);
    this.context.subscriptions.push(
      this.output,
      this.view,
      this.changed,
      { dispose: () => this.dispose() },
      ...Object.entries(this.commands()).map(([name, run]) => vscode.commands.registerCommand(`ipuAsm.${name}`, run)),
      ...this.listeners()
    );
    return this.rescan();
  }

  listeners() {
    const listeners = [
      vscode.window.onDidChangeWindowState((state) => state.focused && this.scheduleRescan()),
      vscode.window.onDidChangeActiveTextEditor(() => {
        this.scheduleRescan();
        this.follow();
      }),
      this.view.onDidChangeVisibility(() => this.follow()),
      vscode.workspace.onDidChangeWorkspaceFolders(() => this.rescan()),
      vscode.workspace.onDidChangeConfiguration((e) => e.affectsConfiguration('ipuAsm.cases') && this.redraw()),
      // Out of Restricted Mode: list the kernels it kept from loading.
      vscode.workspace.onDidGrantWorkspaceTrust(() => this.indexes.forEach((index) => index.refresh())),
      vscode.window.onDidCloseTerminal((t) => {
        busy.delete(t);
        sentAt.delete(t);
        for (const [key, terminal] of terminals) if (terminal === t) terminals.delete(key);
      }),
    ];
    if (vscode.window.onDidEndTerminalShellExecution) {
      listeners.push(vscode.window.onDidEndTerminalShellExecution((e) => busy.delete(e.terminal)));
    }
    return listeners;
  }

  dispose() {
    clearTimeout(this.rescanTimer);
    this.timers.forEach((timer) => clearTimeout(timer));
  }
}

/** Register the sidebar and the kernel commands. For tests: the live index and tree, the query
 *  without its prompts, and `ready`, settled once the open folders' checkouts are found. */
function register(context) {
  const kernels = new Kernels(context);
  const ready = kernels.start();
  const { indexes, tree, view, editorContext } = kernels;
  return {
    indexes, tree, view, editorContext, ready,
    kernelOfFile: (uri) => kernels.kernelOfFile(uri),
    runQuery: (op, params) => runQuery([...indexes.values()].find((i) => i.manifest), op, params, kernels.output),
  };
}

module.exports = { register, setting, timeoutSeconds };
