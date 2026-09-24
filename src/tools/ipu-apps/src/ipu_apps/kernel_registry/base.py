"""Base class for IPU application test harnesses.

Subclass :class:`IpuApp`, implement :meth:`setup` (and optionally
:meth:`teardown`), then call :meth:`run`::

    class MyApp(IpuApp):
        def setup(self, state):
            load_binary_to_xmem(state, self.data_path, 0x0000, 128)

        def teardown(self, state):
            if self.output_path:
                dump_xmem_to_binary(state, self.output_path, 0x1000, 128, 1)

    app = MyApp(inst_path="program.bin", data_path="data.bin")
    state, cycles = app.run()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import DebugCallback, run_test

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState


class IpuApp:
    """Base class for IPU application test harnesses.

    Subclasses define how to prepare the IPU state (:meth:`setup`) and
    how to collect results (:meth:`teardown`).  All other fields are
    passed as ``**kwargs`` and stored as attributes automatically.

    Args:
        inst_path:   Path to the assembled instruction binary.
        output_path: Optional path to write output data.
        **kwargs:    Any extra fields are stored as attributes (for example
            ``elu_alpha``, ``window_a`` or ``window_b`` for :meth:`run`).
    """

    def __init__(
        self,
        *,
        inst_path: str | Path,
        output_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.inst_path = Path(inst_path)
        self.output_path = Path(output_path) if output_path else None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def make_state(self) -> "IpuState":
        """Create fresh state using this harness's registry execution config."""
        from ipu_apps.kernel_registry.registry import create_state

        return create_state(self)

    def setup(self, state: "IpuState") -> None:
        """Prepare the IPU state before execution. Override this."""

    def teardown(self, state: "IpuState") -> None:
        """Collect results after execution. Override this."""

    def run(
        self,
        *,
        max_cycles: int = 1_000_000,
        debug_callback: DebugCallback | None = None,
        state: "IpuState | None" = None,
        elu_alpha: float | None = None,
        window_a: float | None = None,
        window_b: float | None = None,
        alias_profile=None,
    ) -> tuple["IpuState", int]:
        """Run the app end-to-end. Returns ``(state, cycles)``.

        Optional ``elu_alpha``, ``window_a`` and ``window_b`` match
        :func:`ipu_emu.emulator.run_test`: they configure emulator-only activation
        parameters (not CR) — α for ``elu`` and the ``[a, b)`` bounds of ``window``.
        An explicit argument wins; otherwise an attribute of the same name stored on
        the app from ``__init__(**kwargs)`` (for example ``MyApp(..., elu_alpha=0.5)``)
        is used when present.
        """
        ea = elu_alpha if elu_alpha is not None else getattr(self, "elu_alpha", None)
        wa = window_a if window_a is not None else getattr(self, "window_a", None)
        wb = window_b if window_b is not None else getattr(self, "window_b", None)
        debug_launch = os.environ.get("IPU_DEBUG_TUI") == "1"
        if debug_launch:
            from ipu_emu.debug_launch import make_tui_debug_callback

            from ipu_apps.kernel_registry.registry import _harness_spec

            spec = _harness_spec(self)
            debug_callback = make_tui_debug_callback(
                spec.name if spec is not None else type(self).__name__
            )
        try:
            state = state if state is not None else self.make_state()
            if alias_profile is not None:
                state.alias_profile = alias_profile
                state.stats.alias_profile = alias_profile
            return run_test(
                inst_path=self.inst_path,
                setup=self.setup,
                teardown=self.teardown,
                max_cycles=max_cycles,
                debug_callback=debug_callback,
                break_on_entry=debug_launch,
                state=state,
                elu_alpha=ea,
                window_a=wa,
                window_b=wb,
            )
        except KeyboardInterrupt:
            if debug_launch:
                # F5 restores the terminal before resuming the emulator, so
                # interrupts between stops must cancel outside curses too.
                # Unwind case workspaces without checking incomplete output.
                raise SystemExit(0) from None
            raise
