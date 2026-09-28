'use strict';

// The runner every test shares: `test(name, fn)` adds a case (sync or async). Plain node: `main(summary)`
// runs them in order, exits 1 on any failure, else prints `summary(count)`. Inside VS Code
// (integration/): `runAll(label, prefix)` prints a line per check and throws on any failure.

const cases = [];
const test = (name, fn) => cases.push([name, fn]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** `fn()` once it is truthy, polled every 200ms for up to `ms`. */
async function until(fn, what, ms = 30000) {
  for (const end = Date.now() + ms; ; await sleep(200)) {
    const value = await fn();
    if (value) return value;
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
  }
}

/** Run every case in order, calling `report(name, error or null)`; resolves to the failure count. */
async function runCases(report) {
  let failed = 0;
  for (const [name, fn] of cases) {
    const err = await Promise.resolve().then(fn).then(() => null, (e) => e || new Error(String(e)));
    if (err) failed++;
    report(name, err);
  }
  return failed;
}

const main = (summary) => runCases((name, err) => err && console.error(`FAIL ${name}\n     ${err.message}`)).then((failed) => {
  if (failed) {
    console.error(`\n${failed} failure(s).`);
    process.exit(1);
  }
  console.log(summary(cases.length));
});

async function runAll(label, prefix = '') {
  const failed = await runCases((name, err) => console.log(err ? `  FAIL  ${prefix}${name}\n        ${err.message}` : `  ok    ${prefix}${name}`));
  console.log(`${cases.length - failed} of ${cases.length} ${label} checks pass.`);
  if (failed) throw new Error(`${failed} ${label} check(s) failed`);
}

module.exports = { test, main, sleep, until, runAll, count: () => cases.length };
