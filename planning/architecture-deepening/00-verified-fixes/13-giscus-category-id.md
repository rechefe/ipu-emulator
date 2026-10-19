# giscus: commit the Announcements `category_id` in `docs/mkdocs.yml`

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-13 · Size XS · Depends on: E00-06 · Behaviour change: yes (the deployed site shows the giscus widget) · Label: `ready-for-human`

## Goal

`docs/mkdocs.yml` `extra.giscus.category_id` holds the real id of the "Announcements" discussion category, so a plain `bazel build --config=docs //docs:build_docs` with no `GISCUS_*` overrides renders the giscus widget. This completes the decision "giscus IDs: committed in `mkdocs.yml`" (IMPLEMENTATION-DECISIONS.md, E00-06).

## Why a human

E00-06 committed `repo_id: "R_kgDOQZJnxQ"` (checked against `gh api repos/rechefe/ipu-emulator --jq .node_id`). The category id is not in the REST API. It comes only from giscus.app or from GraphQL `repository.discussionCategories`, and the agent sessions cannot reach either (both return 403 through the session proxy).

Until this lands:

- `docs/overrides/partials/giscus.html` omits the widget, because it needs `repo`, `repo_id` and `category_id`.
- The deployed site shows comments only if the `GISCUS_CATEGORY_ID` repository secret is set. `docs.yml` passes it to the build, and `--config=docs` forwards it.
- If that secret is not set, the master-only `check-giscus` job in `.github/workflows/docs.yml` fails every master docs run. The deploy job does not depend on it, so the site still updates.

## Steps

1. Get the id (`DIC_...`) of the "Announcements" category. Either:
   - open https://giscus.app, enter `rechefe/ipu-emulator`, pick category "Announcements", and copy `data-category-id`; or
   - run `gh api graphql -f query='{repository(owner:"rechefe",name:"ipu-emulator"){discussionCategories(first:25){nodes{id name}}}}'`.
2. Set `category_id: "<DIC_...>"` under `extra.giscus` in `docs/mkdocs.yml`, and delete the "still has to be filled in" comment above it.
3. Optional: delete the `GISCUS_REPO_ID` and `GISCUS_CATEGORY_ID` repository secrets, which only override the committed values.

## Tests

- [ ] `bazel build --config=docs //docs:build_docs` with no `GISCUS_*` set; `tar -xOf bazel-bin/docs/site.tar ./index.html | grep -c 'data-category-id="DIC_'` prints at least 1.
- [ ] CI: the `check-giscus` job passes on the next master docs run.

## Acceptance Criteria

- [ ] `extra.giscus.category_id` in `docs/mkdocs.yml` is non-empty and correct.
- [ ] The deployed site shows the giscus widget on content pages.
- [ ] Master docs runs are green.

## Interim

Before merging E00-06, check under Settings > Secrets and variables > Actions that the `GISCUS_CATEGORY_ID` secret exists. If it does, the deployed site shows the widget and master runs stay green until this issue lands. If it does not, land this issue together with E00-06.
