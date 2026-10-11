"""Run registry cases under table tracing, and check ``SPEC.tables`` against them.

The analysis itself is in ``ipu_emu.table_trace``; this only connects it to
kernels, cases and their declared parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ipu_emu.table_trace import BANK_ROWS, Finding, Table, TraceSession, check_tables, trace_tables

from ipu_apps.kernel_registry.cases import KernelCase, run_case
from ipu_apps.kernel_registry.registry import kernel_spec


@dataclass
class TracedRun:
    """One case run under tracing."""

    kernel: str
    params: Mapping[str, Any]
    session: TraceSession
    state: Any
    cycles: int

    @property
    def declared(self) -> dict[str, Table] | None:
        """``SPEC.tables`` evaluated at this run's parameters, if declared."""
        return declared_tables(self.kernel, self.params)


def declared_tables(kernel: str, params: Mapping[str, Any]) -> dict[str, Table] | None:
    spec = kernel_spec(kernel)
    if spec.tables is None:
        return None
    tables = dict(spec.tables(**params))
    for name, table in tables.items():
        if not isinstance(table, Table):
            raise TypeError(f"{kernel}: SPEC.tables[{name!r}] must be a Table; got {table!r}")
    return tables


class TracedRunFailed(Exception):
    """The case failed (did not halt, or its output check failed), but what
    was traced up to then is kept in ``run`` -- a failing kernel is exactly
    when its tables are worth looking at."""

    def __init__(self, run: TracedRun | None, cause: BaseException) -> None:
        super().__init__(str(cause) or type(cause).__name__)
        self.run = run
        self.cause = cause


def run_traced(kernel: str, case: KernelCase, **run_kwargs) -> TracedRun:
    """``run_case`` with every program it starts recorded.

    Raises :class:`TracedRunFailed` if the case fails, carrying the trace.
    """
    seen: dict[str, Mapping[str, Any]] = {}
    with trace_tables() as session:
        try:
            state, cycles = run_case(kernel, case,
                                     on_params=lambda p: seen.setdefault("params", p),
                                     **run_kwargs)
        except Exception as exc:
            failed = exc
        else:
            failed = None
    if failed is not None:
        partial = None
        if "params" in seen and session.programs:
            partial = TracedRun(kernel, seen["params"], session, None,
                                sum(p.cycles for p in session.programs))
        raise TracedRunFailed(partial, failed) from failed
    return TracedRun(kernel, seen["params"], session, state, cycles)


def check_declared(run: TracedRun, bank_rows: int = BANK_ROWS) -> list[Finding]:
    """Findings for a run of a kernel that declares its tables; ``[]`` otherwise."""
    tables = run.declared
    return [] if tables is None else check_tables(run.session, tables, bank_rows)
