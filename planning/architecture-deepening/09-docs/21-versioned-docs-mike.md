# Versioned docs site with mike

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-21 · Size M · Depends on: E09-06 to E09-19 (the first published version is the restructured site), E01-07 (offline docs build) · Behaviour change: yes (the site URL gains a version segment, and Pages deploys from a branch)

## Goal

Readers can switch between documentation versions with the Material version selector: `dev` (master) and each release.

## Decisions (user, grilling Q16–Q19)

- Versions: `dev`, rebuilt on every master push, plus one version per release tag `vX.Y` (a new scheme; `assembler_0.1.0` is left alone).
- Alias `latest` points at the newest `vX.Y`. The site root redirects to `latest`, or to `dev` until the first `vX.Y` tag exists.
- The Bazel offline build (`bazel build --config=docs //docs:build_docs`) stays the only way the site is built. CI publishes its output into the `gh-pages` branch in mike's layout (`<version>/` directories plus `versions.json`). It does not run `mike deploy`, which would rebuild the site outside Bazel. The layout stays compatible with the `mike` CLI, so `mike list`, `mike alias` and `mike delete` work on the branch.
- Manual step for the repository owner, listed in the PR: Settings → Pages → Source changes from "GitHub Actions" to "Deploy from a branch: `gh-pages`".

## Implementation

1. Pin `mike` in `requirements_lock.txt` as a docs-only dependency (the update tool and local `mike` CLI use).
2. `docs/mkdocs.yml`: `extra.version.provider: mike`, plus `extra.version.default: latest`.
3. A publish tool (`//docs:publish_version`, a Python binary) takes the built `site.tar`, a version name and optional aliases. It checks out `gh-pages` into a temporary worktree, replaces `<version>/`, writes alias directories as redirects the way mike does, and updates `versions.json` (`[{"version", "title", "aliases"}]`, newest first). Then it commits and pushes. Running it twice with the same input changes nothing.
4. `.github/workflows/docs.yml`:
   - on master pushes, publish `dev`;
   - on `v*` tags, publish `vX.Y` and move `latest`;
   - drop the `upload-pages-artifact` / `deploy-pages` jobs;
   - keep PR builds (build only, no publish).
5. Root `index.html` on `gh-pages` redirects to the default version.
6. Update the docs that state the site URL, and the history-link hook (E09-04) if links gain the version segment.

## Tests

- A unit test for the publish tool against a temporary git repository covers:
  - a first publish;
  - a republish with no change;
  - an alias move;
  - the `versions.json` shape matching what the mike CLI reads.
- `//docs:test_doc_links` still passes.

## Acceptance criteria

- [ ] The version selector shows `dev` (and `vX.Y`/`latest` once a tag exists).
- [ ] A master push updates only `dev/` on `gh-pages`.
- [ ] A `v*` tag publishes that version and moves `latest`.
- [ ] The site build is still the offline Bazel build.
- [ ] The PR lists the Pages source switch as a manual step.
