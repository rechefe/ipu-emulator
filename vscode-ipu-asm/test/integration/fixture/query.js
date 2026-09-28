// Stands in for `query --json` in the integration test: argv is
// `--json <op> name=value...`, answered the way the real query answers.
const [flag, op, ...params] = process.argv.slice(2);
if (flag !== '--json') process.exit(2);
const supported = op === 'identity' && params.includes('shape=4,128');
const reason = supported ? 'rows of 128 elements' : `no ${op} kernel covers ${params.join(' ')}`;
process.stdout.write(JSON.stringify({ supported, reason, app: supported ? 'demo_kernel' : null,
  use: null, kwargs: {}, shapes: null, notes: [], alternatives: [] }));
process.exit(supported ? 0 : 1);
