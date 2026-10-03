"""Control panels for the SNN vector visualization prototype.

This module holds the Qt control widgets rendered alongside the plots in the
main window. Each panel binds to a live simulation object (a
:class:`~snn_vector_sim.core.lif_neuron.LIFNeuron` or the
:class:`~snn_vector_sim.core.engine.SimulationEngine`) and writes edits
directly through to that object so changes take effect on the next simulation
step with no restart (live update).

Panels:

- :class:`NeuronControlPanel` - bounded spin boxes editing a single
  ``LIFNeuron``'s parameters (Req 4). One is instantiated per neuron; because
  the output neuron's ``tau_m`` also sets the ``N`` decay constant, editing it
  here live-tunes the vector dynamics with no extra wiring (Req 7.3).
- :class:`InputControlPanel` - a combo box selecting the input current type
  (Constant/Pulse) plus magnitude and pulse period/width spin boxes, editing a
  single input generator attribute on the engine live (Req 5).
- :class:`WeightControls` - two spin boxes editing the engine's synaptic
  weights ``w1``/``w2`` live (Req 6).
- :class:`SimControls` - a ``dt`` spin box editing ``engine.dt`` (Req 12),
  plus a GUI-side simulation speed control (slider synced with a numeric
  field, Req 14) and a checkable pause/resume button (Req 13). The speed and
  paused state live on the GUI side and are exposed to the main window via
  the :attr:`SimControls.speed` property, :meth:`SimControls.is_paused`, and
  the ``speedChanged``/``pauseToggled`` Qt signals.

Qt binding isolation
--------------------
The Qt binding import lives in one place (``from PySide6 import ...``) so the
Qt-free ``core`` package stays importable and unit-testable in a headless
environment that lacks PySide6.
"""

from PySide6 import QtCore, QtWidgets

from snn_vector_sim.core.current import (
    ConstantCurrent,
    ManualPulseCurrent,
    OffCurrent,
    PulseCurrent,
)


class NeuronControlPanel(QtWidgets.QGroupBox):
    """A labeled group of spin boxes editing one :class:`LIFNeuron` live.

    The panel exposes a bounded :class:`QtWidgets.QDoubleSpinBox` for each of
    the neuron's parameters (``tau_m``, ``v_rest``, ``v_th``, ``v_reset``,
    ``R``, ``t_refrac``). Each spin box is initialized from the neuron's current
    attribute value, and its ``valueChanged`` signal writes the new value
    straight back onto the bound neuron. The write mutates only this panel's
    target neuron, leaving every other neuron untouched (Req 4.4), and the next
    :meth:`LIFNeuron.step` reads the updated value (Req 4.2, 4.3).

    Strictly-positive parameters (``tau_m`` and ``R``) are enforced through the
    widget minimum, so the UI cannot drive them to zero or negative values.

    Parameters
    ----------
    neuron:
        The :class:`~snn_vector_sim.core.lif_neuron.LIFNeuron` to edit. Its
        attributes are mutated in place.
    title:
        Group box title shown above the controls (e.g. ``"Input 1"``).
    parent:
        Optional parent widget.
    """

    def __init__(self, neuron, title: str = "Neuron", parent=None) -> None:
        super().__init__(title, parent)
        self._neuron = neuron

        form = QtWidgets.QFormLayout(self)

        # tau_m: strictly positive membrane time constant (ms).
        self._tau_m = self._make_spin(
            value=neuron.tau_m, minimum=0.001, maximum=1000.0,
            decimals=3, step=0.1,
        )
        self._tau_m.valueChanged.connect(self._on_tau_m)
        form.addRow("tau_m (ms)", self._tau_m)

        # R: strictly positive membrane resistance (MOhm).
        self._R = self._make_spin(
            value=neuron.R, minimum=0.001, maximum=1000.0,
            decimals=3, step=0.1,
        )
        self._R.valueChanged.connect(self._on_R)
        form.addRow("R (MOhm)", self._R)

        # v_rest: resting potential (mV), may be negative.
        self._v_rest = self._make_spin(
            value=neuron.v_rest, minimum=-200.0, maximum=100.0,
            decimals=2, step=1.0,
        )
        self._v_rest.valueChanged.connect(self._on_v_rest)
        form.addRow("v_rest (mV)", self._v_rest)

        # v_th: spike threshold (mV), may be negative.
        self._v_th = self._make_spin(
            value=neuron.v_th, minimum=-200.0, maximum=100.0,
            decimals=2, step=1.0,
        )
        self._v_th.valueChanged.connect(self._on_v_th)
        form.addRow("v_th (mV)", self._v_th)

        # v_reset: reset potential after a spike (mV), may be negative.
        self._v_reset = self._make_spin(
            value=neuron.v_reset, minimum=-200.0, maximum=100.0,
            decimals=2, step=1.0,
        )
        self._v_reset.valueChanged.connect(self._on_v_reset)
        form.addRow("v_reset (mV)", self._v_reset)

        # t_refrac: refractory duration (ms), zero allowed.
        self._t_refrac = self._make_spin(
            value=neuron.t_refrac, minimum=0.0, maximum=100.0,
            decimals=2, step=0.5,
        )
        self._t_refrac.valueChanged.connect(self._on_t_refrac)
        form.addRow("t_refrac (ms)", self._t_refrac)

    @staticmethod
    def _make_spin(value, minimum, maximum, decimals, step):
        """Build a configured :class:`QtWidgets.QDoubleSpinBox`.

        The range/decimals/step are applied before the initial value so the
        value is not clamped by stale bounds.
        """
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def refresh_from_engine(self) -> None:
        """Re-sync every spin box from the bound neuron (used after a load).

        Signals are blocked so updating the widgets does not write the (now
        stale, mid-update) values back onto the neuron.
        """
        pairs = (
            (self._tau_m, self._neuron.tau_m),
            (self._R, self._neuron.R),
            (self._v_rest, self._neuron.v_rest),
            (self._v_th, self._neuron.v_th),
            (self._v_reset, self._neuron.v_reset),
            (self._t_refrac, self._neuron.t_refrac),
        )
        for spin, value in pairs:
            with QtCore.QSignalBlocker(spin):
                spin.setValue(value)

    # --- Slots: write each edited value straight back onto the neuron. -----
    def _on_tau_m(self, value: float) -> None:
        """Live-update the neuron's membrane time constant."""
        self._neuron.tau_m = value

    def _on_R(self, value: float) -> None:
        """Live-update the neuron's membrane resistance."""
        self._neuron.R = value

    def _on_v_rest(self, value: float) -> None:
        """Live-update the neuron's resting potential."""
        self._neuron.v_rest = value

    def _on_v_th(self, value: float) -> None:
        """Live-update the neuron's spike threshold."""
        self._neuron.v_th = value

    def _on_v_reset(self, value: float) -> None:
        """Live-update the neuron's post-spike reset potential."""
        self._neuron.v_reset = value

    def _on_t_refrac(self, value: float) -> None:
        """Live-update the neuron's refractory duration."""
        self._neuron.t_refrac = value

class InputControlPanel(QtWidgets.QGroupBox):
    """A labeled group selecting and tuning one input current generator live.

    The panel targets a single generator attribute on the simulation engine
    (named by ``gen_attr``, e.g. ``"gen1"``) and lets the user:

    - pick the current type with a :class:`QtWidgets.QComboBox`
      (``"Constant"`` / ``"Pulse"``, Req 5.1);
    - set the magnitude (nA) with a :class:`QtWidgets.QDoubleSpinBox`
      (Req 5.2);
    - set the pulse ``period`` and ``width`` (ms), enabled only in Pulse mode.

    Every edit is applied live (Req 5.5): type changes build a brand-new
    generator and assign it to the engine attribute in a single ``setattr``
    (so a partially-constructed generator is never observable by a concurrent
    step); magnitude/period/width edits write straight onto the current
    generator's attributes. The next engine step reads the updated value with
    no restart.

    Parameters
    ----------
    engine:
        The simulation engine holding the generator attributes. Mutated in
        place via :func:`setattr` when the type is swapped.
    gen_attr:
        Name of the generator attribute on ``engine`` to control
        (e.g. ``"gen1"`` or ``"gen2"``).
    title:
        Group box title shown above the controls (e.g. ``"Input 1"``).
    parent:
        Optional parent widget.
    """

    # Default pulse shape used when switching an existing non-pulse generator
    # to Pulse mode and no prior pulse values are available.
    _DEFAULT_PERIOD = 50.0  # ms
    _DEFAULT_WIDTH = 5.0     # ms

    def __init__(self, engine, gen_attr: str, title: str = "Input", parent=None) -> None:
        super().__init__(title, parent)
        self._engine = engine
        self._gen_attr = gen_attr

        gen = self._generator()

        form = QtWidgets.QFormLayout(self)

        # Type selector. Index order is fixed and must match the swap logic in
        # :meth:`_on_type_changed`: 0=Off, 1=Constant, 2=Pulse, 3=Manual.
        self._type = QtWidgets.QComboBox()
        self._type.addItems(["Off", "Constant", "Pulse", "Manual Fire"])
        form.addRow("Type", self._type)

        # Magnitude (nA): all driven generator types expose a ``.magnitude``.
        self._magnitude = self._make_spin(
            value=getattr(gen, "magnitude", 0.0),
            minimum=-1000.0, maximum=1000.0, decimals=2, step=1.0,
        )
        form.addRow("magnitude (nA)", self._magnitude)

        # Pulse period (ms): strictly positive (Pulse mode only).
        self._period = self._make_spin(
            value=getattr(gen, "period", self._DEFAULT_PERIOD),
            minimum=0.1, maximum=100000.0, decimals=2, step=1.0,
        )
        form.addRow("period (ms)", self._period)

        # Pulse width (ms): zero allowed (Pulse and Manual modes).
        self._width = self._make_spin(
            value=getattr(gen, "width", self._DEFAULT_WIDTH),
            minimum=0.0, maximum=100000.0, decimals=2, step=1.0,
        )
        form.addRow("width (ms)", self._width)

        # Fire button: triggers a single pulse in Manual mode (Req: manual fire).
        self._fire_btn = QtWidgets.QPushButton("Fire")
        self._fire_btn.clicked.connect(self._on_fire)
        form.addRow(self._fire_btn)

        # Initialize the combo and enabled-state to match the current type.
        # Block signals so initialization does not trigger a type swap.
        with QtCore.QSignalBlocker(self._type):
            self._type.setCurrentIndex(self._index_for(gen))
        self._sync_enabled(self._type.currentIndex())

        # Wire live-update slots after initialization.
        self._type.currentIndexChanged.connect(self._on_type_changed)
        self._magnitude.valueChanged.connect(self._on_magnitude)
        self._period.valueChanged.connect(self._on_period)
        self._width.valueChanged.connect(self._on_width)

    # Combo index constants (kept in one place so the mapping is unambiguous).
    _IDX_OFF = 0
    _IDX_CONSTANT = 1
    _IDX_PULSE = 2
    _IDX_MANUAL = 3

    @staticmethod
    def _index_for(gen) -> int:
        """Return the combo index matching an existing generator instance."""
        if isinstance(gen, OffCurrent):
            return InputControlPanel._IDX_OFF
        if isinstance(gen, PulseCurrent):
            return InputControlPanel._IDX_PULSE
        if isinstance(gen, ManualPulseCurrent):
            return InputControlPanel._IDX_MANUAL
        return InputControlPanel._IDX_CONSTANT

    @staticmethod
    def _make_spin(value, minimum, maximum, decimals, step):
        """Build a configured :class:`QtWidgets.QDoubleSpinBox`.

        Range/decimals/step are applied before the initial value so the value
        is not clamped by stale bounds.
        """
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _generator(self):
        """Return the engine's current generator for this panel's attribute."""
        return getattr(self._engine, self._gen_attr)

    def _sync_enabled(self, index: int) -> None:
        """Enable/disable fields to match the selected generator type.

        - magnitude: used by Constant, Pulse, and Manual (disabled for Off).
        - period: Pulse only.
        - width: Pulse and Manual.
        - Fire button: Manual only.
        """
        self._magnitude.setEnabled(index != self._IDX_OFF)
        self._period.setEnabled(index == self._IDX_PULSE)
        self._width.setEnabled(index in (self._IDX_PULSE, self._IDX_MANUAL))
        self._fire_btn.setEnabled(index == self._IDX_MANUAL)

    def refresh_from_engine(self) -> None:
        """Re-sync all widgets from the engine's current generator.

        Called after a settings file is loaded so the panel reflects the newly
        installed generator. Signals are blocked during the update so syncing
        the widgets does not trigger a spurious generator swap.
        """
        gen = self._generator()
        widgets = (self._type, self._magnitude, self._period, self._width)
        blockers = [QtCore.QSignalBlocker(w) for w in widgets]  # noqa: F841
        self._type.setCurrentIndex(self._index_for(gen))
        self._magnitude.setValue(getattr(gen, "magnitude", 0.0))
        self._period.setValue(getattr(gen, "period", self._DEFAULT_PERIOD))
        self._width.setValue(getattr(gen, "width", self._DEFAULT_WIDTH))
        self._sync_enabled(self._type.currentIndex())

    # --- Slots: apply edits live. -----------------------------------------
    def _on_type_changed(self, index: int) -> None:
        """Swap the engine's generator to the selected type (Req 5.1, 5.5).

        The new generator is fully built with the current field values before a
        single ``setattr`` publishes it, so a half-initialized generator is
        never observable by a concurrent step.
        """
        mag = self._magnitude.value()
        if index == self._IDX_OFF:
            new_gen = OffCurrent()
        elif index == self._IDX_PULSE:
            new_gen = PulseCurrent(
                magnitude=mag,
                period=self._period.value(),
                width=self._width.value(),
            )
        elif index == self._IDX_MANUAL:
            new_gen = ManualPulseCurrent(
                magnitude=mag,
                width=self._width.value(),
            )
        else:  # Constant
            new_gen = ConstantCurrent(magnitude=mag)
        setattr(self._engine, self._gen_attr, new_gen)
        self._sync_enabled(index)

    def _on_fire(self) -> None:
        """Trigger a manual pulse if the current generator supports firing."""
        gen = self._generator()
        fire = getattr(gen, "fire", None)
        if callable(fire):
            fire()

    def _on_magnitude(self, value: float) -> None:
        """Live-update the current generator's magnitude (Req 5.2, 5.5)."""
        gen = self._generator()
        if hasattr(gen, "magnitude"):
            gen.magnitude = value

    def _on_period(self, value: float) -> None:
        """Live-update the pulse period if the current generator is a pulse."""
        gen = self._generator()
        if isinstance(gen, PulseCurrent):
            gen.period = value

    def _on_width(self, value: float) -> None:
        """Live-update the pulse/manual width if the generator supports it."""
        gen = self._generator()
        if isinstance(gen, (PulseCurrent, ManualPulseCurrent)):
            gen.width = value


class WeightControls(QtWidgets.QGroupBox):
    """A labeled group of spin boxes editing the two synaptic weights live.

    The panel exposes one :class:`QtWidgets.QDoubleSpinBox` for each synaptic
    weight (``w1`` and ``w2``). Each spin box is initialized from the engine's
    current weight, and its ``valueChanged`` signal writes the new value
    straight back onto the engine (``engine.w1``/``engine.w2``). The next
    engine step then uses the updated weight for subsequent instantaneous
    current jumps and Vector_N impulses with no restart (Req 6.1, 6.2, 6.3).

    Parameters
    ----------
    engine:
        The simulation engine holding the ``w1`` and ``w2`` float attributes,
        mutated in place.
    title:
        Group box title shown above the controls.
    parent:
        Optional parent widget.
    """

    def __init__(self, engine, title: str = "Weights", parent=None) -> None:
        super().__init__(title, parent)
        self._engine = engine

        form = QtWidgets.QFormLayout(self)

        # w1: synaptic weight for Input_Neuron 1. May be negative.
        self._w1 = self._make_spin(
            value=engine.w1, minimum=-100.0, maximum=100.0,
            decimals=2, step=0.1,
        )
        self._w1.valueChanged.connect(self._on_w1)
        form.addRow("w1", self._w1)

        # w2: synaptic weight for Input_Neuron 2. May be negative.
        self._w2 = self._make_spin(
            value=engine.w2, minimum=-100.0, maximum=100.0,
            decimals=2, step=0.1,
        )
        self._w2.valueChanged.connect(self._on_w2)
        form.addRow("w2", self._w2)

        # w_gain: output-drive multiplier. Scales the current delivered to the
        # output neuron (w*gain) without affecting the Vector_N impulses, so
        # the output can be driven to spike while the vector plot keeps its
        # scale. Strictly positive, large range so the output can be pushed.
        self._w_gain = self._make_spin(
            value=engine.w_gain, minimum=0.0, maximum=10000.0,
            decimals=2, step=1.0,
        )
        self._w_gain.valueChanged.connect(self._on_w_gain)
        form.addRow("output gain", self._w_gain)

    @staticmethod
    def _make_spin(value, minimum, maximum, decimals, step):
        """Build a configured :class:`QtWidgets.QDoubleSpinBox`.

        Range/decimals/step are applied before the initial value so the value
        is not clamped by stale bounds.
        """
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def refresh_from_engine(self) -> None:
        """Re-sync the weight/gain spin boxes from the engine (after a load)."""
        for spin, value in (
            (self._w1, self._engine.w1),
            (self._w2, self._engine.w2),
            (self._w_gain, self._engine.w_gain),
        ):
            with QtCore.QSignalBlocker(spin):
                spin.setValue(value)

    # --- Slots: write each edited weight straight back onto the engine. ----
    def _on_w1(self, value: float) -> None:
        """Live-update synaptic weight w1 (Req 6.1, 6.2, 6.3)."""
        self._engine.w1 = value

    def _on_w2(self, value: float) -> None:
        """Live-update synaptic weight w2 (Req 6.1, 6.2, 6.3)."""
        self._engine.w2 = value

    def _on_w_gain(self, value: float) -> None:
        """Live-update the output-drive gain (scales output current only)."""
        self._engine.w_gain = value


class SimControls(QtWidgets.QGroupBox):
    """Simulation-wide controls: timestep, speed, and pause/resume.

    This panel owns three concerns:

    - **Timestep (Req 12).** A :class:`QtWidgets.QDoubleSpinBox` bound to
      ``engine.dt``, initialized from the engine (default 0.1 ms, Req 12.2).
      Edits write ``engine.dt`` directly so the next step uses the new value
      (Req 12.3, 12.4).
    - **Speed (Req 14).** A GUI-side simulation speed in the float range
      ``[0.01, 1.0]`` (design default ``M = 1.0``). The speed does *not* live
      on the engine - it governs the main window's tick scheduler (task 8).
      It is driven by an integer-backed :class:`QtWidgets.QSlider` (Req 14.1)
      kept two-way in sync with an exact :class:`QtWidgets.QDoubleSpinBox`
      (Req 14.2). The sync blocks the partner widget's signals with
      :class:`QtCore.QSignalBlocker` to avoid feedback loops (Req 14.3,
      14.4). The spin box range clamps invalid/out-of-range numeric entry to
      ``[0.01, 1.0]`` and the slider is kept consistent (Req 14.5). The
      current speed is exposed through the read-only :attr:`speed` property
      and the :data:`speedChanged` signal.
    - **Pause/resume (Req 13).** A checkable :class:`QtWidgets.QPushButton`
      whose label toggles between ``"Pause"`` and ``"Resume"``. The paused
      state is exposed through :meth:`is_paused` and the :data:`pauseToggled`
      signal, which the main window consumes to gate its tick scheduler.

    Parameters
    ----------
    engine:
        The simulation engine holding the ``dt`` float attribute.
    title:
        Group box title shown above the controls.
    parent:
        Optional parent widget.
    """

    #: Emitted with the new speed (float in ``[0.01, 1.0]``) whenever it changes.
    speedChanged = QtCore.Signal(float)
    #: Emitted with the new paused state (``True`` when paused) on each toggle.
    pauseToggled = QtCore.Signal(bool)
    #: Emitted with the new trail-fade strength (float in ``[0, 1]``).
    trailFadeChanged = QtCore.Signal(float)

    # Speed range and slider scaling. The slider is integer-only, so the
    # float speed is encoded as ``slider_value / _SPEED_SCALE``. The minimum is
    # 0.001 (very slow motion); the scale is 1000 so the integer slider spans
    # 1..1000 with fine resolution across the whole range.
    _SPEED_MIN = 0.001
    _SPEED_MAX = 1.0
    _SPEED_SCALE = 1000  # slider range 1..1000 -> speed 0.001..1.000

    def __init__(self, engine, title: str = "Simulation", parent=None) -> None:
        super().__init__(title, parent)
        self._engine = engine

        # GUI-side state. Speed defaults to the design value M = 1.0.
        self._speed: float = self._SPEED_MAX
        self._paused: bool = False

        form = QtWidgets.QFormLayout(self)

        # --- Timestep dt (Req 12). ----------------------------------------
        self._dt = QtWidgets.QDoubleSpinBox()
        self._dt.setDecimals(3)
        self._dt.setRange(0.01, 10.0)
        self._dt.setSingleStep(0.01)
        self._dt.setValue(engine.dt)
        self._dt.valueChanged.connect(self._on_dt)
        form.addRow("dt (ms)", self._dt)

        # --- Speed slider + numeric field (Req 14). -----------------------
        self._speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._speed_slider.setRange(
            int(round(self._SPEED_MIN * self._SPEED_SCALE)),  # 1
            int(round(self._SPEED_MAX * self._SPEED_SCALE)),  # 100
        )
        self._speed_slider.setValue(self._speed_to_slider(self._speed))
        self._speed_slider.valueChanged.connect(self._on_speed_slider)
        form.addRow("speed (slider)", self._speed_slider)

        self._speed_spin = QtWidgets.QDoubleSpinBox()
        self._speed_spin.setDecimals(3)
        self._speed_spin.setRange(self._SPEED_MIN, self._SPEED_MAX)
        self._speed_spin.setSingleStep(0.001)
        self._speed_spin.setValue(self._speed)
        self._speed_spin.valueChanged.connect(self._on_speed_spin)
        form.addRow("speed", self._speed_spin)

        # --- Trail fade: how fast the red vector trail decays with age. ---
        # GUI/display-only (not an engine or saved setting). 0 = long lingering
        # tail, 1 = fades out quickly. Default 0.5. Encoded on an integer slider
        # 0..100 and mirrored by a numeric field.
        self._trail_fade = 0.5
        self._fade_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._fade_slider.setRange(0, 100)
        self._fade_slider.setValue(int(round(self._trail_fade * 100)))
        self._fade_slider.valueChanged.connect(self._on_fade_slider)
        form.addRow("trail fade (slider)", self._fade_slider)

        self._fade_spin = QtWidgets.QDoubleSpinBox()
        self._fade_spin.setDecimals(2)
        self._fade_spin.setRange(0.0, 1.0)
        self._fade_spin.setSingleStep(0.05)
        self._fade_spin.setValue(self._trail_fade)
        self._fade_spin.valueChanged.connect(self._on_fade_spin)
        form.addRow("trail fade", self._fade_spin)

        # --- Pause / resume (Req 13). -------------------------------------
        self._pause_btn = QtWidgets.QPushButton("Pause")
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        form.addRow(self._pause_btn)

    def refresh_from_engine(self) -> None:
        """Re-sync the dt spin box from the engine (after a load).

        Speed and paused state are GUI-side and not part of saved settings, so
        they are left as-is.
        """
        with QtCore.QSignalBlocker(self._dt):
            self._dt.setValue(self._engine.dt)

    # --- Speed <-> slider conversion helpers. -----------------------------
    def _speed_to_slider(self, speed: float) -> int:
        """Convert a float speed in ``[0.01, 1.0]`` to an integer slider tick."""
        return int(round(speed * self._SPEED_SCALE))

    def _slider_to_speed(self, tick: int) -> float:
        """Convert an integer slider tick to a float speed in ``[0.01, 1.0]``."""
        return tick / self._SPEED_SCALE

    @staticmethod
    def _clamp_speed(speed: float) -> float:
        """Clamp an arbitrary speed into the valid ``[0.01, 1.0]`` range (Req 14.5)."""
        return min(SimControls._SPEED_MAX, max(SimControls._SPEED_MIN, speed))

    def _set_speed(self, speed: float) -> None:
        """Store a clamped speed and emit :data:`speedChanged` if it changed.

        The caller is responsible for keeping the widget that originated the
        change coherent; this only updates the stored value and notifies
        listeners (the main window's tick scheduler).
        """
        speed = self._clamp_speed(speed)
        if speed != self._speed:
            self._speed = speed
            self.speedChanged.emit(speed)

    # --- Public accessors consumed by the main window (task 8). -----------
    @property
    def speed(self) -> float:
        """The current GUI-side simulation speed, a float in ``[0.01, 1.0]``."""
        return self._speed

    def is_paused(self) -> bool:
        """Return ``True`` while the simulation is paused (Req 13)."""
        return self._paused

    @property
    def trail_fade(self) -> float:
        """The current trail-fade strength, a float in ``[0, 1]``."""
        return self._trail_fade

    # --- Slots. -----------------------------------------------------------
    def _on_dt(self, value: float) -> None:
        """Live-update the engine's fixed timestep (Req 12.3, 12.4)."""
        self._engine.dt = value

    def _set_trail_fade(self, fade: float) -> None:
        """Store the trail-fade value and emit :data:`trailFadeChanged`."""
        fade = max(0.0, min(1.0, fade))
        if fade != self._trail_fade:
            self._trail_fade = fade
            self.trailFadeChanged.emit(fade)

    def _on_fade_slider(self, tick: int) -> None:
        """Apply a slider-driven trail fade and mirror it onto the spin box."""
        fade = tick / 100.0
        with QtCore.QSignalBlocker(self._fade_spin):
            self._fade_spin.setValue(fade)
        self._set_trail_fade(fade)

    def _on_fade_spin(self, value: float) -> None:
        """Apply a spin-box-driven trail fade and mirror it onto the slider."""
        with QtCore.QSignalBlocker(self._fade_slider):
            self._fade_slider.setValue(int(round(value * 100)))
        self._set_trail_fade(value)

    def _on_speed_slider(self, tick: int) -> None:
        """Apply a slider-driven speed and mirror it onto the spin box (Req 14.3).

        The spin box's signals are blocked during the mirror so the update
        does not loop back into :meth:`_on_speed_spin`.
        """
        speed = self._slider_to_speed(tick)
        with QtCore.QSignalBlocker(self._speed_spin):
            self._speed_spin.setValue(speed)
        self._set_speed(speed)

    def _on_speed_spin(self, value: float) -> None:
        """Apply a spin-box-driven speed and mirror it onto the slider (Req 14.4).

        The spin box range already clamps out-of-range entry to
        ``[0.01, 1.0]`` (Req 14.5); the slider is kept consistent with the
        clamped value. The slider's signals are blocked during the mirror so
        the update does not loop back into :meth:`_on_speed_slider`.
        """
        speed = self._clamp_speed(value)
        with QtCore.QSignalBlocker(self._speed_slider):
            self._speed_slider.setValue(self._speed_to_slider(speed))
        self._set_speed(speed)

    def _on_pause_toggled(self, checked: bool) -> None:
        """Update the paused state, button label, and emit :data:`pauseToggled` (Req 13)."""
        self._paused = checked
        self._pause_btn.setText("Resume" if checked else "Pause")
        self.pauseToggled.emit(checked)


class SaveLoadControls(QtWidgets.QGroupBox):
    """Save and load the engine's settings to/from a user-chosen JSON file.

    Two buttons drive a native file dialog (no terminal involved):

    - **Save** opens a "save as" dialog so the user names the file and picks
      where it goes, then writes the current engine configuration there.
    - **Load** opens an "open" dialog, reads a settings file, applies it to the
      engine, and runs the ``on_loaded`` callback so every control panel and
      plot re-syncs to the loaded values.

    Errors (bad path, malformed JSON) are reported in a message box rather than
    crashing the app.

    Parameters
    ----------
    engine:
        The simulation engine to snapshot / restore.
    on_loaded:
        Zero-argument callable invoked after a successful load, used by the
        main window to refresh the control widgets from the engine.
    title:
        Group box title.
    parent:
        Optional parent widget.
    """

    _FILE_FILTER = "SNN settings (*.json);;All files (*)"

    def __init__(self, engine, on_loaded=None, title: str = "Settings", parent=None) -> None:
        super().__init__(title, parent)
        self._engine = engine
        self._on_loaded = on_loaded

        row = QtWidgets.QHBoxLayout(self)
        self._save_btn = QtWidgets.QPushButton("Save...")
        self._load_btn = QtWidgets.QPushButton("Load...")
        self._save_btn.clicked.connect(self._on_save)
        self._load_btn.clicked.connect(self._on_load)
        row.addWidget(self._save_btn)
        row.addWidget(self._load_btn)

    def _on_save(self) -> None:
        """Prompt for a destination file and write the engine settings there."""
        # Imported lazily so core persistence stays decoupled from the GUI import
        # path and so a missing GUI toolkit never blocks headless core use.
        from snn_vector_sim.core.persistence import save_engine

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save settings", "settings.json", self._FILE_FILTER,
        )
        if not path:
            return  # user cancelled
        try:
            save_engine(self._engine, path)
        except OSError as exc:
            self._error("Could not save settings", str(exc))

    def _on_load(self) -> None:
        """Prompt for a settings file, apply it, and refresh the controls."""
        from snn_vector_sim.core.persistence import load_engine

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load settings", "", self._FILE_FILTER,
        )
        if not path:
            return  # user cancelled
        try:
            load_engine(self._engine, path)
        except (OSError, ValueError) as exc:
            # ValueError covers json.JSONDecodeError (a ValueError subclass).
            self._error("Could not load settings", str(exc))
            return
        if self._on_loaded is not None:
            self._on_loaded()

    def _error(self, title: str, message: str) -> None:
        """Show a modal warning dialog describing a save/load failure."""
        QtWidgets.QMessageBox.warning(self, title, message)
