# Docs build: network only under `--config=docs`, path-filtered docs workflow

Part of [Epic E01: Foundation](00-epic.md) · ID E01-07 · Size S · Depends on: E00-06, E00-07, E01-01, E01-04 · Behaviour change: none (same site); non-docs builds stop touching the network and stop depending on `GITHUB_TOKEN`

## Goal

Only the docs build reaches the network, and only under `--config=docs`. No other target's cache key depends on a token in the environment. The docs workflow runs when docs or their generators change, and authenticates its GitHub API calls.

## Background

Verified at 694a8a8:

- `.bazelrc:24-25` sets `build --action_env=GITHUB_TOKEN` for every build. Every action's key then includes the token value (or its absence), so a developer with the token set and CI without it never share cache entries, and a token rotation invalidates everything.
- `//docs:generate_release_log` (`docs/BUILD.bazel:75-81`) calls the GitHub API from a genrule tagged only `no-sandbox`. It is not tagged `docs`, so `--build_tag_filters=-docs` (ci.yml:35, session-start.sh:64) still builds it, and `bazel build //...` reaches the network. Its output depends on the time of the build and on GitHub state.
- `docs.yml` never sets `GITHUB_TOKEN` for the build step (docs.yml:43-54), so the release log is fetched unauthenticated (60 requests per hour per IP).
- `docs.yml:3-7` runs on every push to every branch and every pull request, whatever changed.
- `build_docs` packs the site with `tar -cf $@ -C $(@D)/site_output .` (docs/BUILD.bazel, end of the genrule): file order, mtimes and owners vary per build, so two builds of the same commit give different `site.tar` digests.
- E00-06 adds the `build:docs` config (giscus variables). E00-07 makes the release-log script survive network errors.

## Decisions applied

- Q71: CI time and cache reuse.
- Plan 08 question 4 (how the release log is produced) is not decided. This issue keeps generating it during the docs build and only scopes its network use.

## Implementation

1. `.bazelrc`: delete `build --action_env=GITHUB_TOKEN` and add `build:docs --action_env=GITHUB_TOKEN`.
2. `docs/BUILD.bazel` `generate_release_log`: `tags = ["docs", "no-sandbox", "requires-network", "no-remote", "no-cache"]`. The `no-cache` tag keeps a stale log from being served from a cache.
3. `docs.yml`:
   - build step: `env: GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}` (read-only `contents` permission is enough for the PR list).
   - triggers:
     ```yaml
     on:
       push:
         branches: [master]
       pull_request:
         paths:
           - "docs/**"            # includes docs/mkdocs.yml
           - "src/tools/ipu-as-py/src/**"
           - "src/tools/ipu-common/src/**"
           - "src/tools/ipu-apps/src/ipu_apps/kernels/**/*.md"
           - ".github/workflows/docs.yml"
           - "requirements_lock.txt"
       workflow_dispatch:
     ```
     Master pushes always rebuild (the release log changes on every merge). The paths cover the generators at HEAD; E03, E08 and E09 must update them when packages move.
   - toolchain via `./tools/bootstrap.sh` (E01-01).
4. Deterministic archive: `tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf $@ -C $(@D)/site_output .`. The `git-revision-date-localized` plugin still embeds commit dates, which are stable per commit.
5. `CLAUDE.md` and `docs/content/setup.md`: the local docs build command is `bazel build --config=docs //docs:build_docs`.

## Tests

- [ ] `bazel build --config=ci //...` succeeds with networking disabled (for example in a container started with `--network none` after the repository cache is warm).
- [ ] With and without `GITHUB_TOKEN` exported, `bazel build --config=ci //src/tools/...` reports the same action cache hits (no rebuild after toggling the variable).
- [ ] Two `bazel build --config=docs //docs:build_docs` runs on the same clean commit give the same `site.tar` sha256, except for `release-log.md` if a PR merged in between (state it in the PR).
- [ ] A PR touching only `src/tools/ipu-emu-py/` does not start the docs workflow; one touching `docs/content/` does.

## Acceptance Criteria

- [ ] `GITHUB_TOKEN` appears only under `build:docs` in `.bazelrc`.
- [ ] Every networked target is tagged `docs` and `requires-network`.
- [ ] The docs workflow runs on relevant PRs, master pushes and manual dispatch only, and calls the API authenticated.

Open question (plan 08 question 4): keep generating the release log during the docs build (this issue), or commit `release-log.md` and refresh it from a master-only workflow so the docs build is fully offline?
