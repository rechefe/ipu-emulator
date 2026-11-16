"""The menu-driven front end: pick a workload, a memory model and what to
sweep; run; look at the table and the chart; download or save the results.

Launch from ``src/ipu-memsim``::

    uv run --extra ui streamlit run src/ipu_memsim/ui/app.py

How the page is built:

- The widgets on the left only assemble an experiment *spec* (the same dict
  a ``memsim sweep --spec`` file holds). :func:`spec_from_state` does that.
- "Run" passes the spec to ``ipu_memsim.experiment.run_sweep`` and keeps the
  result in the session, so changing a widget does not rerun the experiment.
- Every parameter widget (its label, default, limits and tooltip) comes from
  the registry ``ipu_memsim.params.REGISTRY``. A parameter added there shows
  up here with no change to this file.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path
from typing import Any

import altair as alt
import streamlit as st

from ipu_memsim import experiment
from ipu_memsim.dram import HBM2_DRAMSIM3_FILL_CYCLES
from ipu_memsim.params import (
    CONFIRMED,
    MODEL_KINDS,
    Param,
    build_config,
    params_for,
    placeholders_in_use,
)
from ipu_memsim.toml_out import to_toml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

#: The checkout's top folder: where networks, traces and results live.
ROOT = Path(os.environ.get("MEMSIM_ROOT", Path(__file__).resolve().parents[5]))
NETWORKS = ROOT / "src" / "ipu-memsim" / "networks"
TRACES = ROOT / "memsim-out" / "traces"
NONE = "(none)"
#: Seconds one replay takes on a trace of a few hundred thousand accesses
#: (measured: 1110 replays in 51 s), for the time estimate before a run.
SECONDS_PER_REPLAY = 0.05


def help_text(p: Param) -> str:
    """A parameter's tooltip: what it is, how far to trust the default, why."""
    unit = f" Unit: {p.unit}." if p.unit else ""
    return f"{p.description}{unit} [{p.status}] {p.source}"


def parse_values(text: str) -> list[Any]:
    """``"250, 500, auto"`` as a list: numbers where they parse, true/false, else text."""
    values: list[Any] = []
    for part in text.replace(";", ",").split(","):
        word = part.strip()
        if not word:
            continue
        if word.lower() in ("true", "false"):
            values.append(word.lower() == "true")
            continue
        try:
            number = float(word)
            values.append(int(number) if number == int(number) and "." not in word else number)
        except ValueError:
            values.append(word)
    return values


def default_values(p: Param) -> str:
    """A starting list for sweeping ``p``, shown in the values box."""
    presets = {
        "fill_cycles_per_bank": f"250, 500, 1000, {HBM2_DRAMSIM3_FILL_CYCLES}, 4000, 8000",
        "xmem_mb": "0.5, 1, 2, 4, 8",
        "num_banks": "4, 8, 16, 32, 64",
        "bank_rows": "256, 512, 1024, 2048, 4096",
        "prefetch_distance": "0, 1, 2, 4, 16",
    }
    if p.name in presets:
        return presets[p.name]
    if p.type is bool:
        return "false, true"
    if p.choices:
        return ", ".join(p.choices)
    return str(p.default)


def parameter_widget(p: Param, loaded: dict[str, Any]) -> Any:
    """One editable parameter, by its type. Returns the value to put in the
    spec's ``[fixed]`` table, or ``None`` to leave the registry default."""
    label = f"{p.name}  ·  {p.status}"
    current = loaded.get(p.name, p.default)
    key = f"param_{p.name}"
    if p.type is bool:
        return st.checkbox(label, value=bool(current), help=help_text(p), key=key)
    if p.type is str and p.choices and not isinstance(current, dict):
        index = p.choices.index(current) if current in p.choices else 0
        return st.selectbox(label, p.choices, index=index, help=help_text(p), key=key)
    if p.type is dict or isinstance(current, dict):
        st.caption(f"{p.name}: a table, set it in the TOML file. {help_text(p)}")
        return current
    # Numbers are typed as text so that special values ("auto", "inf") fit too.
    text = st.text_input(
        label, value="" if current is None else str(current), help=help_text(p), key=key
    )
    return text.strip() or None


def sweep_widgets(
    index: int, kind: str, optional: bool, loaded: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The widgets of one swept parameter; ``None`` when it is switched off."""
    names = [p.name for p in params_for(kind) if p.sweepable]
    if not names:
        st.info("This model has no parameter to sweep.")
        return None
    before = loaded[index] if index < len(loaded) else {}
    options = ([NONE] if optional else []) + names
    preferred = before.get("param", NONE if optional else names[0])
    if not optional and "fill_cycles_per_bank" in names and "param" not in before:
        preferred = "fill_cycles_per_bank"
    name = st.selectbox(
        "Second parameter (for a heatmap)" if optional else "Parameter to sweep",
        options, index=options.index(preferred) if preferred in options else 0, key=f"sweep_param_{index}",
    )  # fmt: skip
    if name == NONE:
        return None
    p = next(q for q in params_for(kind) if q.name == name)
    st.caption(help_text(p))
    numeric = p.type in (int, float)
    how = st.radio(
        "Values",
        ["list", "range"] if numeric else ["list"],
        horizontal=True,
        key=f"sweep_how_{index}",
    )
    if how == "list":
        start = (
            ", ".join(str(v) for v in before["values"])
            if before.get("param") == name and "values" in before
            else default_values(p)
        )
        text = st.text_input(
            "Values, comma separated", value=start, key=f"sweep_values_{index}_{name}"
        )
        return {"param": name, "values": parse_values(text)}
    low_col, high_col, steps_col, scale_col = st.columns(4)
    low = low_col.number_input(
        "min", value=float(before.get("min", 250.0)), key=f"sweep_min_{index}_{name}"
    )
    high = high_col.number_input(
        "max", value=float(before.get("max", 8000.0)), key=f"sweep_max_{index}_{name}"
    )
    steps = steps_col.number_input(
        "steps",
        value=int(before.get("steps", 6)),
        min_value=2,
        max_value=200,
        key=f"sweep_steps_{index}_{name}",
    )
    scale = scale_col.selectbox("scale", ["log", "linear"], key=f"sweep_scale_{index}_{name}")
    return {"param": name, "min": low, "max": high, "steps": int(steps), "scale": scale}


def workload_widgets(loaded: dict[str, Any]) -> dict[str, Any]:
    """The workload part of the spec: a network, one kernel, or a saved trace."""
    kinds = ["network", "kernel", "trace"]
    kind = st.radio(
        "Workload", kinds, index=kinds.index(loaded.get("kind", "network")), horizontal=True
    )
    if kind == "network":
        files = sorted(NETWORKS.glob("*.toml"))
        chosen = st.selectbox("Network", files, format_func=lambda path: path.stem)
        with open(chosen, "rb") as handle:
            layers = tomllib.load(handle).get("layer", [])
        types = list(dict.fromkeys(layer["type"] for layer in layers if int(layer.get("count", 1))))
        picked = st.multiselect("Layer types", types, default=loaded.get("layer_types", types))
        workload: dict[str, Any] = {"kind": "network", "network": str(chosen.relative_to(ROOT))}
        if picked and picked != types:
            workload["layer_types"] = picked
        return workload
    if kind == "kernel":
        name = st.text_input("Kernel (a registered name)", value=loaded.get("kernel", "residual_add"),
                             help="For example residual_add, pointwise_conv_unified, matmul_144x144_x128.")  # fmt: skip
        case = st.text_input("Case", value=loaded.get("case", "default"))
        text = st.text_input("Parameter values (key=value, comma separated)", value="",
                             help="For example height=32, width=32, in_channels=96, out_channels=144.")  # fmt: skip
        workload = {"kind": "kernel", "kernel": name, "case": case}
        values = dict(part.split("=", 1) for part in text.split(",") if "=" in part)
        if values:
            workload["values"] = {k.strip(): parse_values(v)[0] for k, v in values.items()}
        return workload
    files = sorted(TRACES.glob("*.csv*")) if TRACES.exists() else []
    if not files:
        st.warning(
            f"No saved traces in {TRACES}. Pick a network or a kernel: it is recorded on first use."
        )
        return {"kind": "trace", "trace": ""}
    chosen = st.selectbox("Saved trace", files, format_func=lambda path: path.name)
    return {"kind": "trace", "trace": str(chosen.relative_to(ROOT))}


def spec_from_state(loaded: dict[str, Any]) -> dict[str, Any]:
    """Draw the sidebar and return the experiment it describes."""
    with st.sidebar:
        st.header("Experiment")
        name = st.text_input("Name", value=loaded.get("name", "experiment"),
                             help="Results are saved in memsim-out/<name>.")  # fmt: skip
        st.subheader("1. Workload")
        workload = workload_widgets(loaded.get("workload", {}))
        st.subheader("2. Memory model")
        loaded_kind = loaded.get("model", {}).get("kind", MODEL_KINDS[0])
        kind = st.selectbox("Model", MODEL_KINDS, index=MODEL_KINDS.index(loaded_kind),
                            help="cache_unit follows the cache-unit specification; the others are toy models.")  # fmt: skip
        st.subheader("3. Sweep")
        sweeps = []
        first = sweep_widgets(0, kind, optional=False, loaded=loaded.get("sweep", []))
        if first:
            sweeps.append(first)
            second = sweep_widgets(1, kind, optional=True, loaded=loaded.get("sweep", []))
            if second and second["param"] != first["param"]:
                sweeps.append(second)
        st.subheader("4. Everything else")
        swept = {s["param"] for s in sweeps}
        fixed: dict[str, Any] = {}
        before = loaded.get("fixed", {})
        # The DRAM kind decides which DRAM parameters apply, so it comes first.
        dram_kind = None
        for p in params_for(kind):
            if p.name == "dram_kind":
                dram_kind = parameter_widget(p, before)
                fixed[p.name] = dram_kind
        for p in params_for(kind):
            if p.name in swept or p.name == "dram_kind":
                continue
            if p.section == "dram" and dram_kind not in p.dram_kinds:
                continue
            value = parameter_widget(p, before)
            if value is not None and str(value) != str(p.default):
                fixed[p.name] = value
        st.subheader("5. Report")
        threshold = st.number_input(
            "Break-even: stall above (% of ideal cycles)",
            value=float(loaded.get("report", {}).get("threshold_percent", 10.0)),
            min_value=0.0,
        )
        limit = st.number_input(
            "Ask before more replays than",
            value=int(loaded.get("report", {}).get("max_replays", 2000)),
            min_value=1,
        )
    if fixed.get("dram_kind") == "bank_fill":
        fixed.pop("dram_kind")  # the default: keep saved specs short
    return {
        "name": name,
        "workload": workload,
        "model": {"kind": kind},
        "fixed": fixed,
        "sweep": sweeps,
        "report": {"threshold_percent": threshold, "max_replays": int(limit)},
    }


def results_csv(table: experiment.SweepTable) -> str:
    """The result table as CSV text, for the download button."""
    names = [p.name for p in table.params]
    columns = [
        "group",
        *names,
        "ideal_cycles",
        "stall_cycles",
        "total_cycles",
        "stall_percent",
        "note",
    ]
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(columns)
    for row in table.rows:
        writer.writerow(["" if row[c] is None else row[c] for c in columns])
    return out.getvalue()


def chart(table: experiment.SweepTable) -> alt.Chart | None:
    """An interactive chart of the result: lines for one swept parameter, a
    heatmap per group for two. Its "..." menu saves it as PNG or SVG."""
    rows = [row for row in table.rows if row["stall_percent"] is not None]
    if not rows:
        return None
    data = alt.Data(values=rows)
    first = table.params[0]
    tips = ["group:N", *(f"{p.name}:N" for p in table.params), "ideal_cycles:Q", "stall_cycles:Q",
            alt.Tooltip("stall_percent:Q", format=".2f")]  # fmt: skip
    if len(table.params) == 1:
        if first.type not in (int, float):
            return None
        xs = [row[first.name] for row in rows if not isinstance(row[first.name], str)]
        log = bool(xs) and min(xs) > 0 and max(xs) / min(xs) > 20
        x = alt.X(f"{first.name}:Q", title=f"{first.name} ({first.unit})",
                  scale=alt.Scale(type="log" if log else "linear"))  # fmt: skip
        return (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x=x,
                y=alt.Y("stall_percent:Q", title="stall (% of ideal cycles)"),
                color=alt.Color("group:N", title="group"),
                tooltip=tips,
            )  # fmt: skip
            .properties(height=380)
        )
    second = table.params[1]
    cells = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(f"{first.name}:O", title=f"{first.name} ({first.unit})"),
            y=alt.Y(f"{second.name}:O", title=f"{second.name} ({second.unit})", sort="descending"),
            color=alt.Color(
                "stall_percent:Q", title="stall %", scale=alt.Scale(type="symlog", scheme="blues")
            ),
            tooltip=tips,
        )  # fmt: skip
        .properties(width=360, height=180)
    )
    return cells.facet(facet=alt.Facet("group:N", title=None), columns=2)


def main() -> None:  # noqa: PLR0915 (the page, drawn top to bottom)
    st.set_page_config(page_title="IPU memory what-if", layout="wide")
    st.title("IPU memory what-if")
    st.caption("Record a kernel once, replay it under many memory configurations. "
               "Every parameter and its tooltip come from the registry (ipu_memsim/params.py).")  # fmt: skip

    upload = st.sidebar.file_uploader("Load an experiment (TOML)", type="toml")
    loaded = tomllib.loads(upload.getvalue().decode("utf-8")) if upload else {}
    spec = spec_from_state(loaded)
    kind = spec["model"]["kind"]

    try:
        swept = {s["param"] for s in spec["sweep"]}
        placeholders = [
            p for p in placeholders_in_use(build_config(kind, spec["fixed"])) if p.name not in swept
        ]
        grid = experiment.grid_size(spec) if spec["sweep"] else 0
        problem = "" if spec["sweep"] else "Pick a parameter to sweep."
    except ValueError as error:
        placeholders, grid, problem = [], 0, str(error)

    if placeholders:
        st.warning("**Toy model: placeholder values.** These parameters are not confirmed by any document: "
                   + ", ".join(f"`{p.name}` ({p.status})" for p in placeholders if p.status != CONFIRMED)
                   + ". Results show trends, not chip numbers.")  # fmt: skip
    if problem:
        st.error(problem)

    st.download_button(
        "Save this experiment (TOML)", to_toml(spec), file_name=f"{spec['name']}.toml"
    )
    run = st.button("Run", type="primary", disabled=bool(problem))
    big = grid > 0 and st.session_state.get("last_items", 1) * grid > spec["report"]["max_replays"]
    st.caption(f"{grid} combination(s) of swept values per trace.")
    confirmed = st.checkbox("This is a large grid; run it anyway", value=False) if big else True

    if run and confirmed:
        with st.spinner(
            "Loading traces (a missing trace is recorded once, which can take a minute)..."
        ):
            workload = experiment.load_workload(spec, ROOT)
        usable = sum(1 for item in workload if item.trace is not None)
        st.session_state["last_items"] = max(1, usable)
        total = usable * grid
        if total > spec["report"]["max_replays"] and not big:
            st.warning(f"This grid needs {total} replays (about {total * SECONDS_PER_REPLAY:.0f} s), more than the "
                       f"limit of {spec['report']['max_replays']}. Tick the box that has appeared and press Run again.")  # fmt: skip
        else:
            bar = st.progress(0.0, text=f"0 / {total} replays")
            table = experiment.run_sweep(
                spec, ROOT, workload=workload,
                progress=lambda done, all_: bar.progress(done / max(all_, 1), text=f"{done} / {all_} replays"),
            )  # fmt: skip
            bar.empty()
            st.session_state["table"] = table
    elif run:
        st.warning("Tick the box to confirm the large grid, then press Run again.")

    table = st.session_state.get("table")
    if table is None:
        st.info("Set up the experiment on the left and press Run.")
        return
    st.subheader(f"Result: {table.spec['name']}")
    st.caption(f"{table.replays} replays in {table.seconds:.1f} s.")
    picture = chart(table)
    if picture is not None:
        st.altair_chart(picture, width="stretch" if len(table.params) == 1 else "content")
        st.caption("PNG: use the ... menu at the top right of the chart (Save as PNG).")
    st.dataframe(table.rows, width="stretch")
    edges = experiment.break_even(table)
    st.subheader(
        f"Break-even: where stall first exceeds {table.spec['report']['threshold_percent']:g}%"
    )
    st.caption(
        f"Along {table.params[0].name}. 'crossing' is a straight-line estimate between two swept values."
    )
    st.dataframe(edges, width="stretch")
    left, middle, right = st.columns(3)
    left.download_button("Download table (CSV)", results_csv(table), file_name="results.csv")
    middle.download_button(
        "Download the spec used (TOML)", to_toml(table.spec), file_name="spec.toml"
    )
    if right.button("Save everything to memsim-out/"):
        written = experiment.write_results(table, ROOT / "memsim-out" / table.spec["name"])
        st.success("Saved: " + ", ".join(path.name for path in written))


main()
