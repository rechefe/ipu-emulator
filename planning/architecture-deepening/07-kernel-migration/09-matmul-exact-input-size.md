# Matmul: reject oversize inputs instead of truncating (L15)

Part of [Epic: Kernel migration](00-epic.md) · ID E07-09 · Size S · Depends on: E07-08 · Behaviour change: yes (an input file or array larger than the declared shape is now an error instead of being silently truncated)

## Goal

Make the input-size policy the same for every kernel: the size must be exact
(L15). After E07-08 the framework already rejects wrong shapes for arrays it
generates. This issue removes the remaining truncating paths, in chain fixtures
and callers, and states the change.

## Background

- 4 matmul harnesses truncate with `raw[:expected]`:
  `matmul_{144x144,144x288,288x144,432x144}_x128/app.py:73-90`.
- `kernels/projections/app.py:189-195` already requires an exact size.
- E06-05 packs with an exact shape check, so the new kernels cannot truncate.
  Callers that relied on truncation, such as the chain fixtures that pass larger
  buffers, now fail.

## Implementation

- Find the callers that pass oversize data: grep `test/mobilevit/` and
  `fixture_*.py` for these kernels. Fix each caller to slice explicitly, with
  a comment saying why.
- Note in the PR which callers changed, and that no kernel output changes for
  exact-size inputs (the E02 output digests are unchanged).

## Tests

- [ ] An oversize `--input` for `matmul_144x288_x128` fails with the tensor name and both shapes.
- [ ] Every MobileViT test that uses these kernels passes after the caller fixes. Run the manual ones by name.

## Acceptance Criteria

- [ ] No truncation of input data anywhere in `kernels/` or the chain tests.
- [ ] `bazel test //...` is green.
