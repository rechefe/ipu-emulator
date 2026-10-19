'use strict';

// The extension's ipuAsm.* settings, read per workspace folder (package.json declares them).

const vscode = require('vscode');

/** ipuAsm.timeoutSeconds's default: the same value as its `default` in package.json. */
const DEFAULT_TIMEOUT_SECONDS = 300;

/** Setting ipuAsm.`key` for workspace folder `folder`, or `fallback` when it is not of its type. */
function setting(folder, key, fallback) {
  const value = vscode.workspace.getConfiguration('ipuAsm', folder && folder.uri).get(key);
  return (Array.isArray(fallback) ? Array.isArray(value) : typeof value === typeof fallback) ? value : fallback;
}

/** A feature's on/off switch (ipuAsm.`key`), as set for the folder of the document at `uri`. */
function featureOn(uri, key) {
  const value = vscode.workspace.getConfiguration('ipuAsm', uri).get(key);
  return typeof value === 'boolean' ? value : true;
}

/** ipuAsm.timeoutSeconds for `folder`: the longest a checker, manifest or query run may take. */
const timeoutSeconds = (folder) => setting(folder, 'timeoutSeconds', DEFAULT_TIMEOUT_SECONDS);

module.exports = { setting, featureOn, timeoutSeconds, DEFAULT_TIMEOUT_SECONDS };
