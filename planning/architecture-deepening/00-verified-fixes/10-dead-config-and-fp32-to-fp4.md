# Remove dead build config and the unused `tools/fp32_to_fp4.py`

Part of [Epic E00: Verified fixes](00-epic.md) · ID E00-10 · Size S · Depends on: none · Behaviour change: none

## Goal

Every flag, `load` and script in the build configuration does something. Readers stop following instructions to targets that do not exist.

## Background

Verified at 694a8a8:

- `.bazelrc:3-10` sets `--cxxopt=-std=c++17`, `--conlyopt=-std=c11` and `--copt=-Wall/-Wextra/-Wpedantic`. The repository has no C or C++ source (`find . -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h'` outside `bazel-*` and `node_modules` is empty).
- `.bazelrc:17` sets `coverage --experimental_generate_llvm_lcov`, an LLVM/C++ coverage option with no effect on Python tests.
- `src/tools/ipu-emu-py/BUILD.bazel:5` loads `assemble_asm` from `//:asm_rules.bzl`; nothing in that file calls it.
- `src/tools/ipu-as-py/BUILD.bazel:8-9` and `src/tools/ipu-common/README.md:37-40` tell readers to run `bazel run //src/tools/ipu-as-py:update_requirements` / `//src/tools/ipu-common:update_requirements`. Neither target exists. Requirements are compiled by the `generate_requirements` repository rule (`MODULE.bazel:19-31`, `tools/python_deps.bzl`).
- `tools/fp32_to_fp4.py` is not referenced by any `BUILD` file, script or doc (`grep -rn fp32_to_fp4` outside `bazel-*` finds only the file). The architecture review found it wrong as well: 4.0 encodes as 6.0, 0.5 encodes as 0, and `inf` raises (review register item 50).

## Decisions applied

- L41 (accepted via Q29): delete `tools/fp32_to_fp4.py`.

## Implementation

1. `.bazelrc`: delete lines 3-10 (C/C++ standards and warnings) and line 17 (`--experimental_generate_llvm_lcov`). Leave `coverage --combined_report=lcov` and `--instrument_test_targets`; E01-06 decides coverage.
2. `src/tools/ipu-emu-py/BUILD.bazel`: delete the `assemble_asm` load.
3. `src/tools/ipu-as-py/BUILD.bazel:8-9`: delete the comment. `src/tools/ipu-common/README.md`: replace the "Updating" section with one sentence: "Python dependencies are declared in `pyproject.toml`; see the repository README for how the lockfile is refreshed." E01-02 adds the real refresh command and updates this sentence.
4. Delete `tools/fp32_to_fp4.py`.

## Tests

- [ ] `bazel query 'kind("cc_.* rule", //...)'` is empty (confirms the C/C++ flags had no consumer).
- [ ] `bazel build //... --build_tag_filters=-docs` and `bazel test //...` give the same pass and fail counts as before.
- [ ] `grep -rn "update_requirements\|fp32_to_fp4\|llvm_lcov" --exclude-dir=bazel-* --exclude-dir=planning .` returns nothing.

## Acceptance Criteria

- [ ] No C/C++ flags, no LLVM coverage flag, no unused `load`, no reference to a nonexistent target, and no `fp32_to_fp4.py`.

## Risks

- An out-of-tree script may call `tools/fp32_to_fp4.py`. L41 accepted deletion; mention it in the PR description so anyone affected sees it.
- The Dockerfile installs `build-essential`, `gcc` and `g++` (Dockerfile:7-21) for the same nonexistent C/C++ code. E01-01 rewrites the Dockerfile and drops them there.
