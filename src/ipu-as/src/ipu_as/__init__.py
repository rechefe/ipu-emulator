"""The assembler: assembly source, optionally a Jinja2 template, to instruction words.

``ipu_as.session.AssemblySession`` is the pipeline (render, parse, resolve
labels, encode); the diagnostics, formatter and language service behind the
``ipu-as`` CLI and the VS Code extension sit beside it. The instruction tables
and the word codec come from ``ipu_isa``.
"""
