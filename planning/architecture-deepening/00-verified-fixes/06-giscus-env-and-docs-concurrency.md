# Docs: forward giscus settings into the MkDocs build, and never cancel a master deploy

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-06 · Size S · Depends on: none · Behaviour change: yes (the deployed site shows the giscus comment widget; a feature-branch push no longer cancels a running master deploy)

## Goal

The giscus IDs that `docs.yml` reads from repository secrets reach the MkDocs hook, so the deployed pages render comments. A docs run on one branch never cancels the Pages deploy of another.

## Background

Verified at 694a8a8:

- `docs.yml:43-54` exports `GISCUS_REPO_ID`, `GISCUS_CATEGORY_ID`, `GISCUS_REPO`, `GISCUS_CATEGORY`, `GISCUS_MAPPING`, `GISCUS_THEME` and `GISCUS_LANG`, then runs `bazel build //docs:build_docs`.
- Bazel does not pass the client environment to actions unless a variable is named with `--action_env`. `.bazelrc:25` forwards only `GITHUB_TOKEN`.
- So `docs/hooks/giscus_env.py:35` (`os.environ.get(env_name)`) sees none of the `GISCUS_*` variables inside the `build_docs` genrule (`docs/BUILD.bazel`, tags `["docs", "no-sandbox"]`). `docs/mkdocs.yml:143-154` ships `repo_id: ""` and `category_id: ""`, and `docs/overrides/partials/giscus.html:2` renders the widget only when `repo`, `repo_id` and `category_id` are all set. The deployed site has no comments.
- `docs.yml:14-17`:
  ```yaml
  concurrency:
    group: "pages"
    cancel-in-progress: true
  ```
  The workflow runs on every push and every pull request (`docs.yml:3-7`), so any branch push cancels an in-progress master build-and-deploy.
- `getting-started-guide.yml` already uses a per-ref group (`getting-started-guide-${{ github.ref }}`), the pattern to copy.

## Implementation

1. `.bazelrc`: add a `docs` config that forwards exactly the variables `giscus_env._ENV_TO_YAML` reads (giscus_env.py:12-24):
   ```
   # Docs build: giscus settings for docs/hooks/giscus_env.py.
   build:docs --action_env=GISCUS_REPO_ID
   build:docs --action_env=GISCUS_CATEGORY_ID
   build:docs --action_env=GISCUS_REPO
   build:docs --action_env=GISCUS_CATEGORY
   build:docs --action_env=GISCUS_MAPPING
   build:docs --action_env=GISCUS_REACTIONS_ENABLED
   build:docs --action_env=GISCUS_EMIT_METADATA
   build:docs --action_env=GISCUS_INPUT_POSITION
   build:docs --action_env=GISCUS_THEME
   build:docs --action_env=GISCUS_LANG
   build:docs --action_env=GISCUS_STRICT
   ```
   Scoping them to `--config=docs` keeps them out of every other action's cache key.
2. `docs.yml`: build with `bazel build --config=docs //docs:build_docs`.
3. `docs.yml`: after the build, on `push` to `master` only (secrets are not available to fork PRs), check the output:
   ```bash
   tar -xOf bazel-bin/docs/site.tar ./index.html | grep -q 'data-repo-id="[^"]'
   ```
   Adjust the member path to the archive layout. The step fails the job if the widget is missing.
4. `docs.yml` concurrency:
   ```yaml
   concurrency:
     group: pages-${{ github.ref }}
     cancel-in-progress: ${{ github.ref != 'refs/heads/master' }}
   ```
5. Update the comments in `docs/mkdocs.yml:140-142` and the docstring of `giscus_env.py` to say "build with `--config=docs`".

## Tests

- [ ] Local: `GISCUS_REPO_ID=test-id GISCUS_CATEGORY_ID=test-cat bazel build --config=docs //docs:build_docs`; the built `index.html` contains `data-repo-id="test-id"`.
- [ ] Local: the same build without `--config=docs` has no widget (proves the config is what forwards the values).
- [ ] CI: the master run's new check step passes.
- [ ] CI: push to a feature branch while a master docs run is in progress; the master run completes.

## Acceptance Criteria

- [ ] The deployed site shows the giscus widget on content pages.
- [ ] Only `--config=docs` builds see `GISCUS_*`.
- [ ] Docs runs are grouped per ref, and master runs are never cancelled.
- [ ] The PR is labelled as a behaviour change (deployed site).

Open question: the giscus IDs are public once rendered into page HTML. Committing them to `docs/mkdocs.yml` would remove the secrets and this plumbing entirely. This issue keeps the secrets because that is the current setup; the user may choose to commit the IDs instead.

## Risks

- E01-07 moves `GITHUB_TOKEN` from `build` to `build:docs` and restricts docs triggers. That issue builds on the `docs` config created here.
