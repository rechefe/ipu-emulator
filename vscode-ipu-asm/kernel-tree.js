'use strict';

// The IPU sidebar's rows, built from each checkout's KernelIndex (kernels.js): a folder per
// checkout when there are several, then families, kernels and their cases. Rebuilt on every
// change; ids let `reveal` find a row again.

const path = require('path');
const vscode = require('vscode');

const caseform = require('./caseform');
const runs = require('./runs');

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

module.exports = { folderNodes, kernelNode, roots };
