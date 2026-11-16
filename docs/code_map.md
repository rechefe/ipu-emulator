# Code map of `src/ipu-memsim`

One page: what each file is for, and how one sweep travels from the menu to
the result table. Everything is Python; the only dependencies are numpy and
tomli (plus Streamlit for the optional browser app).

## The modules (`src/ipu-memsim/src/ipu_memsim/`)

| Module | Its job | Depends on |
|---|---|---|
| `trace.py` | The trace: five columns (cycle, op, region, row, bytes) and a header; save and load as gzip CSV. | numpy |
| `record.py` | Run a kernel once on the emulator and return its trace. Uses only the emulator's public reference-model API. The only module that imports the kernel framework. | `trace`, `ipu_harness`, `ipu_isa` |
| `replay.py` | Walk a trace through a model: `total = ideal + sum of stalls`. The loop every result comes from. | `trace`, `models` |
| `models.py` | The model interface (`start`, `access`, `finish`, `counters`) and the toy models: Ideal, AllMiss, PageFirstTouch, Banked, Cache. | `trace` |
| `cache_unit.py` | The model that follows the cache-unit specification: banks, tables, `jump_back`, background DMA, residency. | `models`, `dram` |
| `dram.py` | How long a DMA transfer takes: `BankFillDram` (the swept fill time), `FixedDram`, `TableDram`. | - |
| `dram_trace.py` | Run DRAMsim3 as an external program to measure fill times. | `cache_unit` |
| `params.py` | **The parameter registry**: every tunable parameter with type, unit, default, range, description, status label. Builds and validates configs. | - |
| `config.py` | Turn a config (dict or TOML file) into a model object, after validating it against the registry. | `params`, models |
| `experiment.py` | **The core API**: `run_sweep(spec)` over one or two parameters; break-even; charts; saves results with their spec. | `params`, `config`, `replay`, `network`, `svgplot` |
| `sensitivity.py` | Vary each assumed parameter alone, rank by spread. Built on `run_sweep`. | `experiment` |
| `knee.py` | From a `num_banks` sweep, the smallest XMEM at which more banks stop helping. | `experiment` |
| `formulas.py` | Closed-form working-set formulas per kernel, and their validation against the replay. | `cache_unit`, `replay` |
| `workset.py` | Working-set numbers measured from a trace (footprint, peak live rows, look-back). | `trace` |
| `network.py` | Load a network file's layers, record missing traces, the Stage 4 per-layer-type table. | `record`, `replay`, `config` |
| `netstats.py` | Network-only tables (weights, activations, operations) from layer shapes; no kernel needed. | `formulas` |
| `fpga.py` | Compare the model with measured data; fit the fill time; synthetic test data. | `replay`, `config` |
| `sweep.py` | The fixed Stage 3 sweeps on one trace (kept for `memsim sweep TRACE`). | models, `svgplot` |
| `svgplot.py` | Line chart and heatmap written as SVG text. | - |
| `toml_out.py` | Write a dict as TOML (the standard library only reads it). | - |
| `cli.py` | The `memsim` command line: one small function per command. | everything above |
| `ui/app.py` | The browser menu (Streamlit). No modelling code. | `params`, `experiment` |

Data files: `configs/` (one model config each), `experiments/` (sweep and
sensitivity specs), `networks/` (layer lists), `scripts/export_layers.py`
(layer shapes from a PyTorch model). Tests are in `test/`.

## The call path of one sweep, from the menu to the table

Example: in the browser app, MobileViT-S, model `cache_unit`, sweep
`fill_cycles_per_bank` over 250 to 8,000, press Run.

1. **`ui/app.py: spec_from_state`** reads the widgets and builds a *spec*, a
   plain dict: workload, model kind, the fixed values, the swept parameter
   and its values. The widgets themselves were generated from
   `params.REGISTRY` (`parameter_widget`, `sweep_widgets`).
2. **`experiment.load_workload(spec, root)`** opens the network file and, for
   each layer, calls **`network.trace_for`**: load the saved trace from
   `memsim-out/traces/`, or call **`record.record`** once to make it
   (`ipu_harness.runner.stage` prepares the kernel, then
   `ReferenceModel.step()` runs it word by word while the recorder notes
   every read and write).
3. **`experiment.run_sweep(spec, workload=...)`** checks the spec against the
   registry (`sweep_values`, `params.build_config`), lists every combination
   of swept values, and for each layer type, each combination and each
   layer:
   - **`params.build_config(kind, values)`** turns names and values into the
     `[model]` and `[dram]` tables (deriving `num_banks` from `xmem_mb`);
   - **`config.make_model(config)`** validates them and constructs the model
     (`CacheUnit(...)` with a `BankFillDram`);
   - **`replay.replay(trace, model)`** calls `model.start(trace)` and then
     `model.access(t, op, row, region)` for every access, adding up the
     stalls;
   - the layer's stall, times its count, is added to the layer type's sum.
4. `run_sweep` returns a **`SweepTable`**: one row per layer type and value
   (ideal cycles, stall cycles, stall percent), plus the list of placeholder
   parameters the numbers rest on (`params.placeholders_in_use`).
5. **`ui/app.py`** shows the table, draws the chart, calls
   **`experiment.break_even(table)`**, and on "Save" calls
   **`experiment.write_results`**, which writes `results.csv`, the charts and
   `spec.toml` (the exact spec, so the run can be repeated).

The command line does the same from step 2: `memsim sweep --spec FILE.toml`
(`cli._experiment`) loads the spec from a file instead of from widgets.

## Where to look for a given question

| Question | Look in |
|---|---|
| Why is runtime = ideal + stalls? | `replay.py`, module docstring |
| What happens on one access? | `cache_unit.py: CacheUnit.access` |
| How long does a bank fill take? | `dram.py: BankFillDram.transfer_cycles` |
| How are banks shared between tables? | `cache_unit.py: CacheUnit._allocate` |
| Where does a parameter's default come from? | `params.py: REGISTRY` (its `source`) |
| How is a trace recorded without changing the emulator? | `record.py: record_staged` |
| Where do the formulas come from and how are they checked? | `formulas.py`, module docstring and `validate` |
