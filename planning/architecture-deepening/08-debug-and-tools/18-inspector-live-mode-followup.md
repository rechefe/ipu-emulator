# Follow-up: Cycle Inspector live mode (Pyodide), out of program scope

Part of [Epic: Debugger, generators and editor](00-epic.md) · ID E08-18 · Size — (not estimated) · Depends on: E08-17, E04-30 (stable reference-model API) · Behaviour change: n/a

> **Out of program scope.** This is the follow-up issue Q55 asks for (phase 2). File it so it is tracked, but do not schedule it inside the architecture-deepening program.

## Goal

A reader pastes or uploads a small kernel on the Cycle Inspector page and runs it in the browser. The page assembles it and records a trace live, then shows it in the same viewer as phase 1.

## Background

- Q37 asks to let a reader upload or write a small kernel and run it. Q55 splits that into phase 1 (recorded traces, E08-16 and E08-17) and phase 2 (live mode, this issue).
- Q63: the emulator is pure Python + numpy, which makes Pyodide feasible.
- D5 and Q60: the reference model's stable API and trace schema are the natural interface for a browser host.

## Sketch (to refine when scheduled)

- Ship wheels for `ipu_isa`, `ipu_as` and `ipu_emu` (pure Python) built by Bazel. Load them in Pyodide with numpy.
- The page runs `AssemblySession(source).assemble()`, then the reference model with the trace writer, then E08-16's adapter, then the existing viewer.
- Limits: cycle cap, source size cap, and a worker thread so the page stays responsive.
- Inputs: XMEM preload from a small form or a `.npy` upload. This reuses the E06 `kernel.toml` input description if available.

## Open questions

- Hosting size: Pyodide plus numpy is about 10 MB. Is that acceptable on the docs site?
- Which kernel inputs can a reader provide without the harness?

## Acceptance Criteria (when scheduled)

- [ ] A pasted 20-line kernel assembles, runs and replays in the browser with no server.
