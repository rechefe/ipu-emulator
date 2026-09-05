"""Debug launch belongs to the shared runner, never individual harnesses."""

from io import StringIO
import sys

import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.base import IpuApp
from ipu_emu.debug_control import get_debug_control
from ipu_emu.emulator import DebugAction, run_test
from ipu_emu.ipu_state import IpuState
import ipu_emu.debug_tui as tui
import ipu_emu.debug_launch as launch


class Terminal(StringIO):
    def isatty(self):
        return True


@pytest.fixture
def program(tmp_path):
    path = tmp_path / "program.bin"
    assemble_to_bin_file("INC lr0 1;;\nINC lr0 1;;\nBKPT;;", str(path))
    return path


class App(IpuApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events = []

    def setup(self, state):
        self.events.append("setup")
        state.regfile.set_lr(0, 41)
        state.xmem.write_address(0, b"prepared")

    def teardown(self, state):
        self.events.append("teardown")
        self.output_path.write_bytes(b"completed")


@pytest.fixture
def app(program, tmp_path):
    return App(inst_path=program, output_path=tmp_path / "output.bin")


@pytest.fixture
def debug_terminal(monkeypatch):
    monkeypatch.setenv("IPU_DEBUG_TUI", "1")
    monkeypatch.setattr(sys, "stdin", Terminal())
    monkeypatch.setattr(sys, "stdout", Terminal())


def test_entry_is_after_setup_and_before_side_effects(app, debug_terminal, monkeypatch):
    stops = []

    def forbidden(*args, **kwargs):
        pytest.fail("direct TUI launch must not construct the legacy CLI")

    monkeypatch.setattr("ipu_emu.debug_cli.DebugCLI", forbidden)

    def view(state, cycle):
        assert app.events == ["setup"]
        assert state.inst_mem[0] is not None
        assert state.xmem.read_address(0, 8) == b"prepared"
        assert not get_debug_control(state).breakpoints
        stops.append((state.program_counter, cycle, state.regfile.get_lr(0)))
        assert get_debug_control(state).stop_reason == "entry"
        assert get_debug_control(state).kernel_name == type(app).__name__
        return DebugAction.CONTINUE

    monkeypatch.setattr(tui, "run_debug_tui", view)
    state, cycles = app.run()
    assert stops == [(0, 0, 41)]
    assert cycles == 3
    assert state.regfile.get_lr(0) == 43
    assert app.events == ["setup", "teardown"]
    assert app.output_path.read_bytes() == b"completed"


def test_tui_steps_and_breakpoints_reopen_without_the_cli(app, debug_terminal, monkeypatch):
    stops = []

    def forbidden(*args, **kwargs):
        pytest.fail("TUI stops must not construct the legacy CLI")

    def view(state, cycle):
        stops.append((state.program_counter, cycle))
        session = tui.get_debug_view_session(state)
        if len(stops) == 1:
            session.focus = "registers"
            session.maximized = True
            get_debug_control(state).add_breakpoint(2)
            return DebugAction.STEP
        assert session.focus == "registers"
        assert session.maximized
        return DebugAction.CONTINUE

    monkeypatch.setattr("ipu_emu.debug_cli.DebugCLI", forbidden)
    monkeypatch.setattr(tui, "run_debug_tui", view)
    _, cycles = app.run()
    assert stops == [(0, 0), (1, 1), (2, 2)]
    assert cycles == 3


def test_quit_unwinds_without_teardown_or_returning_partial_state(app, debug_terminal, monkeypatch):
    monkeypatch.setattr(tui, "run_debug_tui", lambda state, cycle: DebugAction.QUIT)
    with pytest.raises(SystemExit) as result:
        app.run()
    assert result.value.code == 0
    assert app.events == ["setup"]
    assert not app.output_path.exists()


def test_curses_failure_does_not_fall_back_or_execute(app, debug_terminal, monkeypatch):
    state = IpuState()

    def failure(state, cycle):
        raise RuntimeError("curses unavailable")

    monkeypatch.setattr(tui, "run_debug_tui", failure)
    with pytest.raises(RuntimeError, match="curses unavailable"):
        app.run(state=state)
    assert state.program_counter == 0
    assert state.regfile.get_lr(0) == 41
    assert app.events == ["setup"]


def test_terminal_interrupt_cancels_without_entering_cli(app, debug_terminal, monkeypatch):
    def interrupted(run):
        raise KeyboardInterrupt

    def forbidden(*args, **kwargs):
        pytest.fail("interrupts must not enter the deprecated CLI")

    monkeypatch.setattr(tui.curses, "wrapper", interrupted)
    monkeypatch.setattr("ipu_emu.debug_cli.DebugCLI", forbidden)
    with pytest.raises(SystemExit) as result:
        app.run()
    assert result.value.code == 0
    assert app.events == ["setup"]
    assert not app.output_path.exists()


def test_normal_run_never_selects_debugger(app, monkeypatch):
    monkeypatch.delenv("IPU_DEBUG_TUI", raising=False)

    def forbidden():
        pytest.fail("normal app.run() must not select a debugger")

    monkeypatch.setattr(launch, "make_tui_debug_callback", forbidden)
    state, cycles = app.run()
    assert cycles == 3
    assert state.regfile.get_lr(0) == 43
    assert app.events == ["setup", "teardown"]


def test_interrupt_after_continue_cancels_without_teardown(app, debug_terminal, monkeypatch):
    state = IpuState()
    stops = []

    def view(state, cycle):
        stops.append((state.program_counter, cycle))
        return DebugAction.CONTINUE

    def interrupt(engine):
        # Entry resumed and completed one instruction before the interrupt.
        assert engine.state.program_counter == 1
        assert engine.state.regfile.get_lr(0) == 42
        raise KeyboardInterrupt

    monkeypatch.setattr(tui, "run_debug_tui", view)
    monkeypatch.setattr("ipu_emu.ipu.Ipu.execute_vliw_cycle", interrupt)
    with pytest.raises(SystemExit) as result:
        app.run(state=state)
    assert result.value.code == 0
    assert stops == [(0, 0)]
    assert state.stats.total_cycles == 1
    assert app.events == ["setup"]
    assert not app.output_path.exists()


@pytest.mark.parametrize("with_callback", [False, True])
def test_normal_run_interrupt_still_propagates(app, monkeypatch, with_callback):
    monkeypatch.delenv("IPU_DEBUG_TUI", raising=False)

    def interrupt(engine):
        raise KeyboardInterrupt

    monkeypatch.setattr("ipu_emu.ipu.Ipu.execute_vliw_cycle", interrupt)
    callback = (lambda state, cycle: DebugAction.CONTINUE) if with_callback else None
    with pytest.raises(KeyboardInterrupt):
        app.run(debug_callback=callback)
    assert app.events == ["setup"]
    assert not app.output_path.exists()


def test_explicit_callback_keeps_normal_behavior(program, tmp_path, monkeypatch):
    monkeypatch.delenv("IPU_DEBUG_TUI", raising=False)
    assemble_to_bin_file("BREAK;;\nBKPT;;", str(program))
    app = App(inst_path=program, output_path=tmp_path / "out.bin")
    stops = []

    def callback(state, cycle):
        stops.append((state.program_counter, cycle))
        return DebugAction.CONTINUE

    _, cycles = app.run(debug_callback=callback)
    assert cycles == 2
    assert stops == [(0, 0)]


def test_noninteractive_debug_is_rejected_before_setup(app, monkeypatch):
    monkeypatch.setenv("IPU_DEBUG_TUI", "1")
    monkeypatch.setattr(sys, "stdin", StringIO())
    with pytest.raises(RuntimeError, match="interactive"):
        app.run()
    assert not app.events


def test_entry_uses_prepared_pc_and_combines_break_causes(program):
    assemble_to_bin_file("INC lr0 1;;\nBREAK;;\nBKPT;;", str(program))
    state = IpuState()
    get_debug_control(state).add_breakpoint(1)
    stops = []

    def setup(state):
        state.program_counter = 1

    def callback(state, cycle):
        stops.append((state.program_counter, cycle, get_debug_control(state).stop_reason))
        return DebugAction.CONTINUE

    _, cycles = run_test(
        inst_path=program, state=state, setup=setup,
        debug_callback=callback, break_on_entry=True,
    )
    assert stops == [(1, 0, "entry, breakpoint, BREAK instruction")]
    assert cycles == 2
    assert state.regfile.get_lr(0) == 0
    assert get_debug_control(state).breakpoints == {1}


@pytest.mark.parametrize("cancel", ["quit", "interrupt_after_continue"])
def test_registry_case_cancel_skips_check_and_cleans_workspace(debug_terminal, monkeypatch, cancel):
    from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, load_cases, run_case

    original = load_cases("identity")["default"]
    workspaces = []
    checks = []

    def prepare(workspace):
        workspaces.append(workspace)
        prepared = original.prepare(workspace, rows=1)
        return PreparedCase(prepared.params, prepared.bindings, lambda: checks.append(True))

    def view(state, cycle):
        assert cycle == 0 and state.program_counter == 0
        assert state.regfile.get_cr(4) == 1
        return DebugAction.QUIT if cancel == "quit" else DebugAction.CONTINUE

    def interrupt(engine):
        raise KeyboardInterrupt

    monkeypatch.setattr(tui, "run_debug_tui", view)
    if cancel == "interrupt_after_continue":
        monkeypatch.setattr("ipu_emu.ipu.Ipu.execute_vliw_cycle", interrupt)
    with pytest.raises(SystemExit) as exc:
        run_case("identity", KernelCase(prepare))
    assert exc.value.code == 0
    assert not checks
    assert not workspaces[0].exists()


def test_registry_case_completion_checks_and_exports(debug_terminal, monkeypatch, tmp_path):
    from ipu_apps.kernel_registry.cases import load_cases, run_case

    stops = []
    def view(state, cycle):
        stops.append((state.program_counter, cycle))
        return DebugAction.CONTINUE

    monkeypatch.setattr(tui, "run_debug_tui", view)
    output = tmp_path / "completed.bin"
    state, cycles = run_case("identity", load_cases("identity")["single_row"], output_path=output)
    assert state.is_halted and cycles > 0
    assert stops == [(0, 0)]
    assert len(output.read_bytes()) == 512
