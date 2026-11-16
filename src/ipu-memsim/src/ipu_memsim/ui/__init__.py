"""The browser front end of the memory what-if tool (optional).

``app.py`` is a Streamlit page. It holds no modelling code: every menu entry
is generated from the parameter registry (``ipu_memsim.params``), and a run
is one call to ``ipu_memsim.experiment.run_sweep``. Nothing else in the
package imports this folder, so the core, the command line and the tests
work without Streamlit installed.

Launch (from ``src/ipu-memsim``)::

    uv run --extra ui streamlit run src/ipu_memsim/ui/app.py
"""
