'use strict';

// What every provider uses: the language id, spans (offsets) as VS Code ranges, and each
// feature's on/off switch (ipuAsm.<feature>).

const vscode = require('vscode');

const { featureOn } = require('../settings');

const LANGUAGE_ID = 'ipu-asm';

const rangeOf = (document, span) => new vscode.Range(document.positionAt(span.start), document.positionAt(span.end));

/** Register `provider` as the IPU assembly's `kind` provider (`CompletionItem`, `Hover`, …). */
const provide = (context, kind, provider, ...rest) =>
  context.subscriptions.push(vscode.languages[`register${kind}Provider`](LANGUAGE_ID, provider, ...rest));

/** `provider`, answering nothing while its switch (ipuAsm.`setting`) is off for the document. */
function switched(setting, provider) {
  const gated = { ...provider };
  for (const [name, fn] of Object.entries(provider)) {
    if (!/^(provide|prepare)/.test(name)) continue;
    gated[name] = (document, ...args) => {
      if (featureOn(document.uri, setting)) return fn.call(provider, document, ...args);
      // An empty prepareRename falls back to renaming the word: refuse instead.
      if (name.startsWith('prepare')) throw new Error(`Off (ipuAsm.${setting}).`);
      return undefined;
    };
  }
  return gated;
}

/** An event fired when ipuAsm.`setting` changes, for a provider VS Code asks again only when told
 *  (inlay hints, folding); the emitter goes with the extension. */
function onSwitch(context, setting) {
  const changed = new vscode.EventEmitter();
  context.subscriptions.push(changed,
    vscode.workspace.onDidChangeConfiguration((e) => e.affectsConfiguration(`ipuAsm.${setting}`) && changed.fire()));
  return changed.event;
}

module.exports = { LANGUAGE_ID, onSwitch, provide, rangeOf, switched };
