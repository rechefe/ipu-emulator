# Add `.github/CODEOWNERS` (last issue of the program)

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-20 · Size S · Depends on: every other issue in E01 to E09 (the package split must be final) · Behaviour change: none

## Goal

Every pull request automatically requests review from the people who own the
area it touches. This lands last, so the paths in the file are the final
package layout and do not need a second pass.

## Background

- There is no `.github/CODEOWNERS` today (`.github/` holds only `workflows/`).
- Decision (Round 10): CODEOWNERS is deferred to the end of the program, after
  the package split (Q69: `ipu_isa`, `ipu_as`, `ipu_emu`, `ipu_debug`,
  `ipu_gen`, `ipu_harness`, `ipu_kernels`, one directory each under
  `src/tools/`).
- Repository collaborators (GitHub API, checked during planning): `rechefe`
  (admin); `YuvalHarary`, `DavidShenzi`, `yarden-carmi`, `ZevProg`, `alonab01`,
  `ranfeb02-sys`, `davidlevy1206-alt` (write). All are valid CODEOWNERS
  handles. Note that `YuvalHarary` commits as `yuval93813` (same GitHub user
  id, 134804184), and `DavidShenzi` commits as `david_sh`.
- Commit history at `694a8a8` (`git shortlog -sn -- <path>`), top authors:

  | Path today | Top human authors |
  |---|---|
  | `src/tools/ipu-common` | yuval93813 (17), david_sh (7), Yarden Carmi (5) |
  | `src/tools/ipu-as-py` | yuval93813 (20), Yarden Carmi (6) |
  | `src/tools/ipu-emu-py` | david_sh (17), yuval93813 (17), Yarden Carmi (15) |
  | `src/tools/ipu-apps` | Yarden Carmi (14), david_sh (6) |
  | `vscode-ipu-asm` | Yarden Carmi (4) |
  | `docs` | yuval93813 (26), Yarden Carmi (22) |
  | `.github` | Yarden Carmi (5), Eyal Rechef (1) |

- Candidate owners from the decisions: `rechefe` (admin), `YuvalHarary`
  (faults owner, requirements OQ-1 and OQ-2), `DavidShenzi`, `yarden-carmi`
  (VS Code extension, kernels). `ZevProg`, `alonab01`, `ranfeb02-sys` and
  `davidlevy1206-alt` have no recorded area.
- `CLAUDE.md`: a downstream RTL repo vendors the generated SystemVerilog
  package; `*_WIDTH` changes are wire-format changes.

## Implementation

1. Before writing the file, ask the user:
   - Which areas, if any, `ZevProg`, `alonab01`, `ranfeb02-sys` and
     `davidlevy1206-alt` own.
   - Whether an RTL engineer should co-own `ipu_isa` and the SystemVerilog
     template, so wire-format changes get RTL review. If yes, which handle
     (must have write access to this repo).
   - Whether to require code-owner review in branch protection (a repository
     setting the user changes, not this PR).
2. Write `.github/CODEOWNERS` with the final paths. Proposed shape, owners to be
   confirmed by the user:

   ```text
   # Default
   *                                   @rechefe

   # ISA definition and generated wire format
   /src/tools/ipu_isa/                 @YuvalHarary @rechefe   # + RTL owner?
   /src/tools/ipu_gen/                 @YuvalHarary            # SV template, docs generators

   # Toolchain
   /src/tools/ipu_as/                  @YuvalHarary
   /src/tools/ipu_emu/                 @DavidShenzi @YuvalHarary
   /src/tools/ipu_debug/               @DavidShenzi @yarden-carmi
   /src/tools/ipu_harness/             @yarden-carmi @DavidShenzi
   /src/tools/ipu_kernels/             @yarden-carmi

   # Editor, docs, CI
   /vscode-ipu-asm/                    @yarden-carmi
   /docs/content/requirements/         @YuvalHarary
   /docs/                              @YuvalHarary @yarden-carmi
   /.github/                           @rechefe @yarden-carmi
   ```

   The mapping above follows commit history and the decision notes. It is a
   proposal, not a decision.
3. Order rules from general to specific; the last match wins in CODEOWNERS.
4. Add a one-line pointer to the file in the E01 review-guideline page.

## Tests

- [ ] GitHub reports no errors for the file:
      `gh api repos/rechefe/ipu-emulator/codeowners/errors` returns an empty
      `errors` list.
- [ ] Open a draft PR that touches one file in each owned path; the expected
      reviewers are requested.

## Acceptance Criteria

- [ ] `.github/CODEOWNERS` exists, every path in it exists, and every handle
      has write access.
- [ ] Every owner assignment was confirmed by the user in the issue thread.
- [ ] The file lands after the last package-moving PR of the program.
