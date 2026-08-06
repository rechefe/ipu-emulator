# GitHub issue drafts — softmax kernels + kernel registry

Target: `rechefe/ipu-emulator`, base branch `master`.
Create order: children A-D first, then umbrella, then edit umbrella to add the checklist.
Nothing below has been filed.

## The drafts

| file | becomes | branch |
|---|---|---|
| [00-epic.md](00-epic.md) | umbrella issue | — |
| [01-softmax-kernels.md](01-softmax-kernels.md) | child A | `pr1-softmax-kernels` |
| [02-benchmarks-and-layout.md](02-benchmarks-and-layout.md) | child B | `pr2-softmax-benchmarks` |
| [03-kernel-registry.md](03-kernel-registry.md) | child C | `pr3-kernel-registry` |
| [04-docs.md](04-docs.md) | child D | `pr4-registry-docs` |

## Branches

Already built and verified: each passes its own tests standing alone, and each
is a LAYER of the finished product rather than a step in how it was built — so
nothing is introduced and then corrected.

| branch | base | files | diff | tests |
|---|---|---|---|---|
| `pr1-softmax-kernels` | `origin/master` | 25 | +3329 | 86 pass |
| `pr2-softmax-benchmarks` | pr1 | 18 | +593 | 96 pass |
| `pr3-kernel-registry` | pr2 | 19 | +2251 −55 | 322 pass |
| `pr4-registry-docs` | pr3 | 3 | +393 | 322 pass |

`pr4`'s tree is identical to the `pr-softmax-apps` integration branch (the only
difference is `BUILD.bazel` target ordering — same target set).

## Note on PRs 1-2

The app modules import `kernel_registry`, which does not exist until PR 3, so
PRs 1-2 ship pre-registry variants of the five `__init__.py` files with explicit
bounds checks. PR 3's diff is then precisely the anti-drift change: explicit
guards out, `SPEC.guard()` delegation and the `SPEC` declaration in.
