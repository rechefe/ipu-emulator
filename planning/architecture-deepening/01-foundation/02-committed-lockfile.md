# Committed Python lockfile: `requirements_lock.txt` replaces fetch-time resolution

Part of [Epic E01: Foundation](00-epic.md) · ID E01-02 · Size M · Depends on: E01-01 · Behaviour change: none (the first lock pins exactly the versions Bazel resolves today)

## Goal

Bazel installs Python wheels from a committed, hash-pinned lockfile. No repository rule runs `uv` or resolves against live PyPI. Two machines on the same commit test against the same numpy, lark and click. Refreshing the lock is one `bazel run` command that needs no host tools.

## Background

Verified at 694a8a8:

- `MODULE.bazel:19-31` declares `generate_requirements(name = "ipu_as_requirements", ...)` over all four `pyproject.toml` files. `tools/python_deps.bzl:8-20` runs `uv pip compile <pyprojects> --extra=docs --extra=dev --python-version=3.10 --generate-hashes` inside a `local = True` repository rule, and `pip.parse` (`MODULE.bazel:57-71`) reads its output.
- Consequences:
  - The resolution happens on every fetch against live PyPI, so `>=` pins (`click>=8.3.1`, `numpy>=1.24.0`, `ml-dtypes>=0.2.0`, ...) float. Two fetches a week apart can install different versions. `MODULE.bazel.lock` cannot capture this.
  - Every environment needs `uv` on the host, which is why it is installed in three workflows and the Dockerfile (E01-01). `vscode-extension.yml:63-67` documents the failure without it: `execvp(uv): No such file or directory`.
- `src/tools/ipu-as-py/uv.lock` and `src/tools/ipu-emu-py/uv.lock` are not read by Bazel and are stale (the ipu-as-py lock has no `pyslang`, which `pyproject.toml` requires). `ipu-apps` and `ipu-common` have none.
- The versions resolved at the time of writing (from the generated `requirements.txt` in the Bazel output base) include `click==8.5.0`, `jinja2==3.1.6`, `lark==1.3.1`, `ml-dtypes==0.6.0`, `numpy==2.2.6`, `pyslang==11.0.0`, `pytest==9.1.1`, `mkdocs==1.6.1`, `mkdocs-material==9.7.7` (43 packages).

## Decisions applied

- Q61(a), Q71: reproducible, fast CI. The lock is the decided mechanism: `uv pip compile` into a committed `requirements_lock.txt` (epic decision list, plan 08 question 3 option (a)).

## Implementation

1. Add `tools/requirements-tools.in` for repository tools that belong to no package: `uv` (used by the refresh target), and later `ruff` (E01-09) and `mypy` (E01-11).
2. Add `//tools:update_requirements_lock`, a `py_binary` depending on `requirement("uv")`, that runs from `BUILD_WORKSPACE_DIRECTORY`:
   ```
   uv pip compile \
     src/tools/ipu-as-py/pyproject.toml src/tools/ipu-emu-py/pyproject.toml \
     src/tools/ipu-apps/pyproject.toml src/tools/ipu-common/pyproject.toml \
     tools/requirements-tools.in \
     --extra dev --extra docs --python-version 3.10 --universal --generate-hashes \
     --no-emit-package ipu-as-py --no-emit-package ipu-emu-py \
     --no-emit-package ipu-apps --no-emit-package ipu-common \
     -o requirements_lock.txt
   ```
   and passes through extra arguments (for example `--upgrade-package numpy`). The bootstrap problem (the first lock must contain `uv`) is solved once by running the same command with a host `uv` to produce the initial file.
3. Generate the first `requirements_lock.txt` with `--upgrade` off and the current versions as constraints (a temporary constraints file from the generated `requirements.txt`), so this PR changes no installed version. Diff the version list against the generated file and paste it into the PR.
4. `MODULE.bazel`: `pip.parse(..., requirements_lock = "//:requirements_lock.txt")`. Delete the `generate_requirements` `use_repo_rule` and `tools/python_deps.bzl`. Keep `parse_pyproject_deps` for now; E01-03 replaces it.
5. Delete `src/tools/ipu-as-py/uv.lock` and `src/tools/ipu-emu-py/uv.lock`.
6. `tools/bootstrap.sh`: remove `--with-uv`; workflows stop passing it. `devcontainer.json`: drop `uv --version`.
7. Document the refresh command in `README.md` (Development section) and in `src/tools/ipu-common/README.md` in place of the sentence E00-10 left.

*As built:* the lock is compiled with `--python-platform x86_64-unknown-linux-gnu`, not `--universal`. rules_python 0.40 keeps only the last line of a name listed twice, which `--universal` produces for platform markers (`tools/update_requirements_lock.py`). A dependency needed only on another platform would be dropped; none is today.

## Tests

- [ ] On a machine without `uv` on `PATH`: `bazel clean --expunge && bazel build //... --build_tag_filters=-docs` succeeds.
- [ ] `bazel run //tools:update_requirements_lock` on an unchanged tree produces a byte-identical lock.
- [ ] `bazel test //...` pass and fail counts are unchanged.
- [ ] Check the lock resolves on the platforms contributors use (Linux x86-64 in CI; ask in the PR whether macOS arm64 is needed and, if so, run `bazel build //src/tools/...` there once).

## Acceptance Criteria

- [ ] No repository rule executes `uv` or any resolver; the only network access at fetch time is the hash-pinned wheel download `pip.parse` already does.
- [ ] `requirements_lock.txt` is the only lock in the repository.
- [ ] `grep -rn "uv pip compile\|astral.sh/uv" .github Dockerfile .claude tools/bootstrap.sh` returns nothing.
- [ ] The refresh command is documented.

## Risks

- `--universal` locks carry environment markers; rules_python 0.40 `pip.parse` handles them, but a wheel missing for a platform fails only on that platform. The platform check in Tests covers this.
- E01-03 changes which targets see which packages. This issue changes only where the versions come from.
