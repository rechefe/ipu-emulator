# docs: application coverage, and a guide for adding applications

### Description

Two pages, added to the mkdocs nav.

**`app-coverage.md`** (Programming the IPU) — how the emulator knows which
application implements a given computation: how a kernel declares itself, how
discovery works at any nesting depth, how overlapping claims resolve by cost
rather than discovery order, why shapes travel as a role-keyed bundle, and what
keeps the whole thing honest.

It states the failure it exists to prevent, with the concrete instance from
this codebase: a rule written twice drifts, and the second copy is the one that
rots.

**`adding-applications.md`** (Contributing) — the contributor guide: what you
deliver, how to write a `KernelSpec`, and the two rules that are easy to get
wrong:

- `supports` states a kernel's TRUE domain, not the range where you would
  prefer it chosen; `cost` decides the winner.
- constructor guards delegate to `SPEC.guard()` rather than restating bounds.

Plus layer-adapter obligations (refuse config you do not model; refuse
look-alike layers), what the generic conformance suites check for you, and a
checklist ending in `bazel test //src/tools/ipu-apps:all`.

### Acceptance

- [ ] A contributor can add a kernel using only these two pages
- [ ] Routing tables shown are generated output, not prose
- [ ] Both pages in the mkdocs nav
- [ ] Docs build cleanly
