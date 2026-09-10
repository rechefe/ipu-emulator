# ETISS Integration — Issue Drafts

This directory holds ready-to-file GitHub issue drafts for re-implementing the
IPU emulator **on top of ETISS** (Extendable Translating Instruction Set
Simulator, <https://github.com/tum-ei-eda/etiss>). The accompanying design spec
lives at
[`docs/content/specs/etiss-integration.md`](../../docs/content/specs/etiss-integration.md).

These are markdown drafts; paste each file's body into a new issue, or file
them with `gh issue create`.

## Index

| # | File | Title | Labels |
|---|------|-------|--------|
| 0 | [`00-epic.md`](00-epic.md) | Epic: IPU as a native ETISS architecture, Python emulator as golden reference | enhancement |
| 1 | [`01-spike-etiss-ipu-arch.md`](01-spike-etiss-ipu-arch.md) | Spike: minimal `IPU` ETISS architecture executing a 224-bit NOP program | enhancement, spike |
| 2 | [`02-bazel-etiss-build.md`](02-bazel-etiss-build.md) | Build ETISS and the `IPU` plugin under Bazel; CI wiring | enhancement, build |
| 3 | [`03-spec-codegen.md`](03-spec-codegen.md) | `gen_etiss.py`: generate struct, bit layout, and decode callback from `instruction_spec.py` | enhancement |
| 4 | [`04-semantics-port.md`](04-semantics-port.md) | Port instruction semantics and numerics to `IPUFuncs.c` | enhancement |
| 5 | [`05-python-backend-differential-tests.md`](05-python-backend-differential-tests.md) | Python `etiss` backend for `run_test` / `IpuApp` and differential test suite | enhancement, testing |
| 6 | [`06-debug-stats-docs.md`](06-debug-stats-docs.md) | Debug (GDB), run stats, tracing, and user documentation | enhancement, docs |
| 7 | [`07-host-vp.md`](07-host-vp.md) | Follow-up: RISC-V host + IPU in one ETISS virtual platform | enhancement, follow-up |

## Suggested order

```
1 ──▶ 2 ──▶ 3 ──▶ 4 ──▶ 5 ──▶ 6
                              └──▶ 7 (follow-up)
```

Issue 1 is a go/no-go spike on the two things ETISS was not obviously built
for (a 224-bit instruction word and JIT-linked vector helpers). Nothing else
starts until it passes. Issues 3 and 4 can overlap once the generated
prototypes from 3 exist.
