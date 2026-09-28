'use strict';

/** A layout-only rewrite for formatter tests: indentation stripped and
 *  consecutive words joined onto one line, only where no comment or Jinja on
 *  the line could swallow what follows. */
function mangle(lang, raw) {
  const codeLines = lang.mask(raw).masked.split('\n');
  const out = [];
  raw.split('\n').forEach((line, i) => {
    const simple = codeLines[i].trim() && codeLines[i] === line; // no comment, no Jinja
    const prev = out[out.length - 1];
    if (simple && prev && prev.simple && prev.text.trimEnd().endsWith(lang.wordEnd)) prev.text = `${prev.text.trimEnd()} ${line.trim()}`;
    else out.push({ text: simple ? line.trim() : line, simple });
  });
  return out.map((o) => o.text).join('\n');
}

module.exports = { mangle };
