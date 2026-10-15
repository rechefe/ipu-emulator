# Reference: reference-model API and trace schema

Part of [Epic E09: Documentation restructure](00-epic.md) · ID E09-16 · Size S · Depends on: E09-06, E09-13, E04 (stable reference-model API, commit record, JSON schema in `ipu_isa`, conformance suite) · Behaviour change: none

## Goal

The stable reference-model API and its JSON commit-record and trace schema have
a reference page generated from the code and the schema, with the version and
the compatibility promise stated.

## Background

- Q11: the stable API is a small versioned module; everything else is internal.
  Breaking changes bump the major version.
- Q12: `step()` returns a commit record.
- Q60: the commit record and trace are a versioned, language-neutral JSON
  schema in `ipu_isa`; the conformance suite is data.
- Q65: strict typing on `ipu_isa` and the reference API first, and Q66 Google
  docstrings, so signatures and docstrings are usable as the source.
- Q52, Q59: the API reports faults F-1 to F-9 by ID; F-8 is not modelled.

## Implementation

1. `reference/reference-model-api.md`:
   - Version and compatibility promise (what counts as breaking).
   - Each public function and type, generated from signatures and Google
     docstrings (`mkdocstrings` if E01's lockfile can add it; open question,
     same choice as E09-15).
   - The commit record and trace schema, rendered from the JSON schema file
     (field, type, meaning), and a link to the raw schema file for download.
   - The fault IDs the API can report, linked to `requirements/faults.md`, with
     F-8 marked "not modelled" (OQ-28).
   - The conformance suite: where it is, how to run it against another
     implementation.
2. Add the page to the `generated_pages` map (E09-04).
3. Mark everything outside the stable module as internal in the Explanation
   architecture page (E09-17); do not document internals here.

## Tests

- [ ] Changing a field in the JSON schema changes the page with no doc edit.
- [ ] `bazel build //docs:build_docs` shows no new warnings.

## Acceptance Criteria

- [ ] The page states the API version and what a breaking change is.
- [ ] Every public symbol and every schema field is on the page, generated.
- [ ] The cocotb how-to (E09-13) links into this page for every call it uses.
