"""The application main window: layout, widget construction, and wiring.

This module assembles the full GUI for the SNN vector visualization prototype.
:class:`MainWindow` owns the single Qt-free
:class:`~snn_vector_sim.core.engine.SimulationEngine`, constructs the four live
plot widgets and the full control panel, lays them out, and wires the controls
to the engine's neurons and fields.

Layout
------
A horizontal :class:`QtWidgets.QSplitter` divides the window:

- **Left**: a 2x2 grid of the four plots -- the network diagram, the membrane
  potential plot, the input current plot, and the Vector_N plot (Req 8-11).
- **Right**: a :class:`QtWidgets.QScrollArea` holding the stacked control
  panels -- three :class:`~snn_vector_sim.gui.controls.NeuronControlPanel`
  instances (one per neuron, Req 4.1), two
  :class:`~snn_vector_sim.gui.controls.InputControlPanel` instances (one per
  input generator, Req 5.1), one
  :class:`~snn_vector_sim.gui.controls.WeightControls` (Req 6.1), and one
  :class:`~snn_vector_sim.gui.controls.SimControls`.

Responsibilities split with task 8.2
-------------------------------------
This module (task 8.1) builds the engine, all widgets, the layout, and the
wiring, and exposes :meth:`refresh_plots` so a tick loop can push engine state
into the plots. It also constructs the :class:`TickScheduler` and a
:class:`QtCore.QTimer` and connects it to a minimal :meth:`on_tick` **stub**.

The actual per-tick stepping logic -- reading the speed/paused state from
:class:`SimControls`, asking the scheduler for ``steps_for_tick`` and running
that many ``engine.step()`` calls through the fractional accumulator -- is the
responsibility of **task 8.2**, which fills in :meth:`on_tick`. The stub here
only refreshes the plots so the window renders without stepping; task 8.2
replaces the marked body.

Qt binding isolation
--------------------
The Qt binding import lives in one place (``from PySide6 import ...``), matching
``plots.py`` and ``controls.py``, so the binding can be swapped in a single
spot and the Qt-free ``core`` package stays headless-testable.
"""

from PySide6 import QtCore, QtWidgets

from snn_vector_sim.core.engine import SimulationEngine
from snn_vector_sim.core.timing import TickScheduler
from snn_vector_sim.gui.controls import (
    InputControlPanel,
    NeuronControlPanel,
    SaveLoadControls,
    SimControls,
    WeightControls,
)
from snn_vector_sim.gui.plots import (
    CurrentPlot,
    NetworkDiagram,
    PotentialPlot,
    VectorPlot,
)


class MainWindow(QtWidgets.QMainWindow):
    """Top-level window hosting the plots, the controls, and the engine.

    Constructs exactly one :class:`SimulationEngine` and binds every control
    panel to that engine's neurons and fields, so edits take effect live on the
    next step (Req 4/5/6). The four plot widgets read engine state on demand via
    :meth:`refresh_plots`.

    The window also builds a :class:`TickScheduler` and a :class:`QtCore.QTimer`
    and connects the timer to :meth:`on_tick`. The timer stepping body is a stub
    completed by task 8.2 (see module docstring); this task leaves it minimal so
    8.2 can own the ``steps_for_tick`` / accumulator stepping without colliding
    with layout and wiring code.

    Parameters
    ----------
    parent:
        Optional parent widget (normally ``None`` for a top-level window).
    """

    # Upper bound on engine steps run in a single tick. Protects the event loop
    # from being starved when (speed, dt) demands a very large step count (e.g.
    # speed=1.0, dt=0.1, t_tick=20 -> 200 steps/tick). Excess steps are credited
    # back to the scheduler accumulator and run on later ticks.
    _MAX_STEPS_PER_TICK = 50

    def __init__(self, parent=None) -> None:
        """Build the engine, widgets, layout, wiring, scheduler, and timer."""
        super().__init__(parent)

        self.setWindowTitle("SNN Vector Visualization")

        # --- The single simulation core shared by every widget. -----------
        self._engine = SimulationEngine()

        # --- Build the four plot widgets; keep references for refresh. -----
        # Names chosen so task 8.2's refresh can call update(engine) on each.
        self._net = NetworkDiagram()      # network diagram (Req 8.1)
        self._nplot = PotentialPlot()     # membrane potentials (Req 9.1)
        self._iplot = CurrentPlot()       # input currents (Req 10.1)
        self._vplot = VectorPlot()        # Vector_N arrow + trail (Req 11.1)

        # --- Build the control panels bound to the engine. ----------------
        # Three neuron panels, one per LIF neuron (Req 4.1).
        self._neuron_in1 = NeuronControlPanel(self._engine.input1, title="Input 1")
        self._neuron_in2 = NeuronControlPanel(self._engine.input2, title="Input 2")
        self._neuron_out = NeuronControlPanel(self._engine.output, title="Output")

        # Two input-current panels, one per generator (Req 5.1).
        self._input1 = InputControlPanel(self._engine, "gen1", title="Input 1 current")
        self._input2 = InputControlPanel(self._engine, "gen2", title="Input 2 current")

        # Synaptic weights (Req 6.1).
        self._weights = WeightControls(self._engine, title="Weights")

        # Global simulation controls (dt, speed, pause/resume). Kept as a
        # reference so task 8.2's tick loop can read speed and paused state.
        self._sim_controls = SimControls(self._engine, title="Simulation")

        # Save/load settings to a user-chosen file. On load, every control
        # panel re-syncs from the engine via _on_settings_loaded.
        self._saveload = SaveLoadControls(
            self._engine, on_loaded=self._on_settings_loaded, title="Settings",
        )

        # --- Assemble the layout. -----------------------------------------
        central = self._build_layout()
        self.setCentralWidget(central)

        # --- Timing model + timer (stepping body is task 8.2). ------------
        self._scheduler = TickScheduler()
        # Cache the paused flag and keep it in sync with the control's signal
        # (Req 13.2, 13.3). The speed is read live from the control each tick
        # instead of being cached, so speed edits take effect on the very next
        # tick with no restart (Req 14.6).
        self._paused = self._sim_controls.is_paused()
        self._sim_controls.pauseToggled.connect(self._on_pause_toggled)

        # When the speed changes, clear the scheduler's carried fraction so the
        # new rate takes effect on the very next tick with no leftover backlog
        # from the previous (possibly much faster) speed. Without this, a drop
        # from fast to slow would keep draining accumulated steps and the speed
        # change would feel delayed rather than instantaneous.
        self._sim_controls.speedChanged.connect(self._on_speed_changed)

        # Trail fade is a display-only setting on the Vector_N plot; drive it
        # live from the SimControls slider/field and seed the initial value.
        self._sim_controls.trailFadeChanged.connect(self._vplot.set_trail_fade)
        self._vplot.set_trail_fade(self._sim_controls.trail_fade)

        # Redraw throttle. Stepping happens every tick, but repainting all four
        # plots is expensive (each pushes the whole history buffer), so we only
        # repaint every Nth tick. This keeps the Qt event loop responsive to
        # close/resize events instead of being starved by back-to-back redraws.
        self._ticks_per_redraw = 3
        self._tick_count = 0

        self._setup_timer()

    # ------------------------------------------------------------------
    # Layout construction.
    # ------------------------------------------------------------------
    def _build_layout(self) -> QtWidgets.QWidget:
        """Assemble the plots and controls into a split central widget.

        Returns:
            The central :class:`QtWidgets.QWidget` (a horizontal splitter with
            the 2x2 plot grid on the left and the scrollable control column on
            the right), ready to pass to :meth:`setCentralWidget`.
        """
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_plots())
        splitter.addWidget(self._build_controls())

        # Give the plots the lion's share of the width; the controls stay a
        # readable fixed-ish column that can still be dragged wider.
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_plots(self) -> QtWidgets.QWidget:
        """Lay the four plot widgets out in a 2x2 grid.

        Returns:
            A :class:`QtWidgets.QWidget` containing the network diagram, the
            membrane potential plot, the input current plot, and the Vector_N
            plot in a two-by-two grid.
        """
        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)

        # Top row: network diagram and Vector_N plot.
        grid.addWidget(self._net, 0, 0)
        grid.addWidget(self._vplot, 0, 1)
        # Bottom row: membrane potentials and input currents.
        grid.addWidget(self._nplot, 1, 0)
        grid.addWidget(self._iplot, 1, 1)

        # Equal weight so each quadrant shares the available space.
        for col in (0, 1):
            grid.setColumnStretch(col, 1)
        for row in (0, 1):
            grid.setRowStretch(row, 1)
        return container

    def _build_controls(self) -> QtWidgets.QWidget:
        """Stack the control panels inside a vertical scroll area.

        The many panels can exceed the window height, so they live in a
        :class:`QtWidgets.QScrollArea` that scrolls vertically.

        Returns:
            The :class:`QtWidgets.QScrollArea` hosting the stacked panels.
        """
        panel = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(panel)

        # Neuron parameter panels (Req 4.1).
        column.addWidget(self._neuron_in1)
        column.addWidget(self._neuron_in2)
        column.addWidget(self._neuron_out)
        # Input current panels (Req 5.1).
        column.addWidget(self._input1)
        column.addWidget(self._input2)
        # Weights (Req 6.1) and global sim controls.
        column.addWidget(self._weights)
        column.addWidget(self._sim_controls)
        # Save / load settings to a file.
        column.addWidget(self._saveload)
        # Push everything to the top; empty space collects at the bottom.
        column.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        return scroll

    # ------------------------------------------------------------------
    # Timer setup. The stepping body is completed by task 8.2.
    # ------------------------------------------------------------------
    def _setup_timer(self) -> None:
        """Create and start the wall-clock :class:`QtCore.QTimer`.

        The timer fires at the scheduler's wall-clock cadence
        (``TickScheduler.t_tick`` ms) and is connected to :meth:`on_tick`.
        Task 8.2 fills in the stepping logic inside :meth:`on_tick`; this method
        only establishes the timer object and cadence so that wiring is already
        in place.
        """
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(self._scheduler.t_tick))
        self._timer.timeout.connect(self.on_tick)
        self._timer.start()

    def on_tick(self) -> None:
        """Advance the simulation one wall-clock tick, then refresh the plots.

        Each wall-clock timer tick:

        - **If not paused** (``self._paused`` is ``False``), ask the scheduler
          how many integration steps this tick's slice of simulated time maps
          to -- ``self._scheduler.steps_for_tick(self._sim_controls.speed,
          self._engine.dt)`` -- and run exactly that many ``self._engine.step()``
          calls. The speed is read *live* from :class:`SimControls` each tick,
          so a slow-motion edit takes effect on the very next tick with no
          restart (Req 14.6, 12.4). The scheduler's fractional accumulator
          converts the generally non-integer desired step count into whole
          steps, carrying the remainder forward, so slow motion and
          sub-one-step-per-tick cases (``< 1`` step/tick) advance correctly on
          average (Req 14.6).
        - **If paused**, no steps are taken, so the engine state is frozen and
          retained (Req 13.2).

        After the (possibly zero) stepping, :meth:`refresh_plots` is **always**
        called so the plots re-render the current engine state every tick --
        including while paused, which keeps the frozen frame on screen rather
        than letting it go stale (Req 9.2, 10.2, 11.3, 13.4). Resuming simply
        stops skipping the stepping, so the simulation continues from the
        retained state (Req 13.3).
        """
        if not self._paused:
            n = self._scheduler.steps_for_tick(
                self._sim_controls.speed, self._engine.dt
            )
            # Cap the per-tick step budget so a heavy (speed, dt) combination
            # can never block the event loop. Excess steps are DROPPED (not
            # carried over): carrying them would build an unbounded backlog so
            # that a later speed change wouldn't take effect until the backlog
            # drained -- the exact "speed change isn't instant" lag. Dropping
            # keeps the response immediate; at high speed this just means
            # "run as fast as the redraw budget allows", which is the intended
            # near-real-time behavior. The scheduler's fraction is also cleared
            # on overflow so no residual debt lingers.
            if n > self._MAX_STEPS_PER_TICK:
                n = self._MAX_STEPS_PER_TICK
                self._scheduler.reset()
            for _ in range(n):
                self._engine.step()

        # Throttle the (expensive) redraw: repaint only every Nth tick so the
        # event loop keeps draining window events. When paused we still want the
        # frozen frame visible, which it already is from the last redraw.
        self._tick_count += 1
        if self._tick_count >= self._ticks_per_redraw:
            self._tick_count = 0
            self.refresh_plots()

    def _on_speed_changed(self, speed: float) -> None:
        """Clear the scheduler backlog so a speed change applies immediately.

        The accumulator only needs to carry a sub-one-step fraction between
        ticks; any larger leftover is stale debt from the previous speed.
        Resetting it on every speed change makes the new rate visible on the
        next tick instead of after the old backlog drains.
        """
        self._scheduler.reset()

    def _on_pause_toggled(self, paused: bool) -> None:
        """Sync the cached paused flag with :class:`SimControls` (Req 13.2, 13.3).

        Keeps :attr:`_paused` -- the flag :meth:`on_tick` gates stepping on --
        in step with the control's checkable pause/resume button. When pausing,
        the scheduler's fractional accumulator is reset so that resuming starts
        a fresh slice of simulated time rather than discharging a burst of steps
        accumulated across a long pause (the accumulator would otherwise keep no
        debt here since paused ticks never call ``steps_for_tick``, but resetting
        makes the resume-from-rest behavior explicit and robust to future
        changes).
        """
        self._paused = paused
        if paused:
            self._scheduler.reset()

    # ------------------------------------------------------------------
    # Settings load: re-sync every control panel from the engine.
    # ------------------------------------------------------------------
    def _on_settings_loaded(self) -> None:
        """Refresh all control panels and plots after a settings file loads.

        ``load_engine`` has already applied the file to the engine and reset
        its runtime state. Here we push the engine's new configuration back
        into every control widget so the UI reflects the loaded values, reset
        the tick scheduler so stepping restarts cleanly, and redraw the plots
        (whose history was cleared by the engine reset).
        """
        self._neuron_in1.refresh_from_engine()
        self._neuron_in2.refresh_from_engine()
        self._neuron_out.refresh_from_engine()
        self._input1.refresh_from_engine()
        self._input2.refresh_from_engine()
        self._weights.refresh_from_engine()
        self._sim_controls.refresh_from_engine()
        self._scheduler.reset()
        self.refresh_plots()

    # ------------------------------------------------------------------
    # Plot refresh shared by the tick loop.
    # ------------------------------------------------------------------
    def refresh_plots(self) -> None:
        """Push the current engine state into all four plot widgets.

        Calls ``update(self._engine)`` on the network diagram, the membrane
        potential plot, the input current plot, and the Vector_N plot. The tick
        loop (task 8.2) calls this every tick -- including while paused -- so the
        displayed state always reflects the engine (Req 9.2, 10.2, 11.3, 13.4).
        """
        self._net.update(self._engine)
        self._nplot.update(self._engine)
        self._iplot.update(self._engine)
        self._vplot.update(self._engine)
