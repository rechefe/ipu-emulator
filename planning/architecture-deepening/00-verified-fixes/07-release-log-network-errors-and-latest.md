# Release log script: survive network errors, and make `--latest` merge instead of overwrite

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-07 · Size S · Depends on: none · Behaviour change: yes (`--latest` keeps existing rows; an offline docs build writes a minimal log instead of failing)

## Goal

`docs/update_release_log.py` never fails the docs build because the network is down, and `--latest` adds the newest merged PR to the existing log instead of replacing the whole log with one row.

## Background

Verified at 694a8a8 (`docs/update_release_log.py`):

- It imports only `HTTPError` (update_release_log.py:20) and catches only `HTTPError` around the API call (update_release_log.py:63-71). `_get` uses `urlopen(req, timeout=30)` (update_release_log.py:48). With no network, DNS failure or a proxy refusal, `urlopen` raises `URLError`; a read timeout raises `TimeoutError` (`socket.timeout`). Both escape, and the `//docs:generate_release_log` genrule (`docs/BUILD.bazel:75-81`) fails, which fails `//docs:build_docs`.
- `--latest` (update_release_log.py:110-114) fetches one PR and then `build_content(prs)` writes the header plus that single row (update_release_log.py:101-105, 133-135). The file docstring says it is for "add only the most recently merged PR"; it actually deletes every other row.
- With `--latest`, the first page is sorted by `updated` (update_release_log.py:61), and the loop returns the first merged PR it sees (update_release_log.py:76-80). That is the most recently *updated* merged PR, not the most recently merged one.
- The script has no test.

## Implementation

1. Catch `URLError` (which includes `HTTPError`), `TimeoutError` and `OSError` around `_get`, with the same warning and fallback as today.
2. `--latest`:
   - Read the existing `--output` file if it exists and parse its table rows (`| [#N](url) | title | merged |`) into a dict keyed by PR number.
   - Fetch the first page, take the merged PR with the greatest `merged_at`, insert or replace it by number, sort all rows by merged time (newest first), and write the file.
   - If the output does not exist, behave like today (one row).
   - On a fetch error, leave an existing output file unchanged.
3. Split `main()` so the merge logic is a pure function `merge_rows(existing_text: str, prs: list[dict]) -> str`, testable without the network.
4. Add `docs/update_release_log_test.py` as a `py_pytest_test` in `docs/BUILD.bazel` (not tagged `docs`, so it runs in the default tier).

## Tests

- [ ] `_get` monkeypatched to raise `URLError`, `TimeoutError` and `HTTPError` in turn: `main()` writes the minimal log and exits 0.
- [ ] `merge_rows`: an existing table of three rows plus one new PR gives four rows, newest first; the same PR number twice gives one row with the newer title.
- [ ] `--latest` picks the greatest `merged_at` from a page whose order is by `updated`.
- [ ] `--latest` with a fetch error leaves an existing file byte-identical.
- [ ] `bazel build //docs:build_docs` with networking disabled (for example `unshare -rn bazel build ...` or a CI job without network) succeeds.

## Acceptance Criteria

- [ ] No network failure mode fails the docs build.
- [ ] `--latest` never removes rows.
- [ ] The script has a test in the default tier.
- [ ] The PR is labelled as a behaviour change.

## Risks

- E01-07 decides how the release log is produced in the hermetic docs build (genrule tags, token scope). This issue only fixes the script and is useful either way.
