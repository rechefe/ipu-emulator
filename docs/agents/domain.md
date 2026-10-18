# Domain docs

This repository has one context: one glossary and one ADR folder, both at the root.

```
/
├── GLOSSARY.md      the project's terms, and the synonyms to avoid
└── docs/adr/        the decisions (0001-isa-package-is-the-single-source-of-truth.md, ...)
```

## Before exploring

- Read `GLOSSARY.md`.
- Read the ADRs in `docs/adr/` that touch the area you are about to work in.

## Use the glossary's vocabulary

When your output names a domain concept (an issue title, a refactor proposal, a test name), use the term `GLOSSARY.md` defines, not a synonym it lists as one to avoid. If the concept is not in the glossary, either you are inventing language the project does not use, or there is a gap: note it for `/domain-modeling`.

## Flag ADR conflicts

If your output contradicts an ADR, say so instead of silently overriding it:

> _Contradicts ADR 0002 (timing contract), but worth reopening because…_
