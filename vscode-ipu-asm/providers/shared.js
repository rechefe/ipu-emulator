'use strict';

// What every provider uses: the language id, and spans (offsets) as VS Code ranges.

const vscode = require('vscode');

const LANGUAGE_ID = 'ipu-asm';

const rangeOf = (document, span) => new vscode.Range(document.positionAt(span.start), document.positionAt(span.end));

/** Register `provider` as the IPU assembly's `kind` provider (`CompletionItem`, `Hover`, …). */
const provide = (context, kind, provider, ...rest) =>
  context.subscriptions.push(vscode.languages[`register${kind}Provider`](LANGUAGE_ID, provider, ...rest));

module.exports = { LANGUAGE_ID, provide, rangeOf };
