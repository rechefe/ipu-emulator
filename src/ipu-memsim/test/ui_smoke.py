"""Start the browser app without a browser and run one sweep end to end.

Not a Bazel test: Streamlit is an optional extra and is not in the repo's
dependency lock. Run it from ``src/ipu-memsim``::

    uv run --extra ui python test/ui_smoke.py

It uses the traces in ``memsim-out/traces`` and records any that are missing.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "ipu_memsim" / "ui" / "app.py"


def main() -> None:
    app = AppTest.from_file(str(APP), default_timeout=600)
    app.run()
    assert not app.exception, app.exception
    # The banner names the placeholder parameters before anything runs.
    assert any("Toy model" in warning.value for warning in app.warning)
    # Two small layer types, the default sweep (bank fill time), then Run.
    app.multiselect[0].set_value(["elementwise", "normalization"]).run()
    next(button for button in app.button if button.label == "Run").click().run()
    assert not app.exception, app.exception
    table = app.session_state["table"]
    assert [p.name for p in table.params] == ["fill_cycles_per_bank"]
    assert sorted(table.groups) == ["elementwise", "normalization"]
    assert len(table.rows) == 2 * 6 and all(row["stall_cycles"] is not None for row in table.rows)
    assert any("Break-even" in text.value for text in app.subheader)
    print(f"ok: {len(table.rows)} rows, {table.replays} replays in {table.seconds:.1f} s")
    for row in table.rows:
        print(
            f"  {row['group']:<14}{row['fill_cycles_per_bank']:>8g}{row['stall_percent']:>10.2f}%"
        )


if __name__ == "__main__":
    main()
