"""Simulation engine state: the vector ``N`` and rolling history buffers.

This module owns the Qt-free simulation state objects used by the vector
visualization prototype:

- :class:`VectorN` holds the central 2D vector ``N = {N_1, N_2}`` (Req 7, 11).
- :class:`HistoryBuffers` holds bounded rolling windows of the quantities the
  GUI plots each tick: simulated time, the three membrane potentials, the two
  input currents, and the ``(N_1, N_2)`` tip trail (Req 9.1, 10.1, 11.2).

The ``SimulationEngine`` (which steps the neurons, decays ``N``, and pushes
samples into :class:`HistoryBuffers`) is implemented in a later task; this
module intentionally provides only the state containers it will use.

Units convention (mV-ms-nA-MOhm), matching ``lif_neuron.py`` and ``current.py``:
- Time: milliseconds (ms)
- Membrane potential (v_in1, v_in2, v_out): millivolts (mV)
- Input currents (I1, I2): nanoamperes (nA)
- Vector_N components (n1, n2): dimensionless accumulator units
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np

from snn_vector_sim.core.lif_neuron import LIFNeuron
from snn_vector_sim.core.current import (
    ConstantCurrent,
    CurrentGenerator,
    PulseCurrent,
)


# Default bounded capacity for the time-series history buffers. Keeping this
# fixed makes memory use and per-tick redraw cost constant regardless of how
# long the simulation runs (design: ~2000-5000 samples).
DEFAULT_MAXLEN = 3000

# Default number of recent vector tips kept for the fading History_Trail
# (Req 11.2). The trail is usually shorter than the full time-series window so
# the fade stays visible rather than smearing across the whole run.
DEFAULT_TRAIL_LEN = 400


@dataclass
class VectorN:
    """The central 2D vector ``N = {N_1, N_2}`` visualized by the prototype.

    Each component ``n_i`` is driven by Input_Neuron ``i``'s spike train,
    decays with the Output_Neuron's ``tau_m``, and is reset to the origin when
    the Output_Neuron spikes (Req 7). The components are plain attributes so the
    engine mutates them directly each timestep.
    """

    n1: float = 0.0  # component N_1, driven by Input_Neuron 1
    n2: float = 0.0  # component N_2, driven by Input_Neuron 2

    def reset(self) -> None:
        """Set both components back to the origin (``n1 = n2 = 0``).

        Called when the Output_Neuron spikes so the vector snaps to the origin
        (Req 7.6).
        """
        self.n1 = 0.0
        self.n2 = 0.0


class HistoryBuffers:
    """Bounded rolling windows of plotted simulation quantities.

    Each buffer is a fixed-length :class:`collections.deque`; once full, pushing
    a new sample automatically discards the oldest one. This keeps memory and
    per-tick redraw cost constant for an arbitrarily long run.

    Buffers held:
    - ``time`` : simulated time stamps (ms).
    - ``v_in1``, ``v_in2``, ``v_out`` : the three membrane potentials (mV, Req 9.1).
    - ``I1``, ``I2`` : the two injected input currents (nA, Req 10.1).
    - ``trail`` : the last ``trail_len`` ``(n1, n2)`` tip positions for the
      Vector_N History_Trail (Req 11.2).

    The time-series buffers share one length (``maxlen``); the trail has its own
    (usually shorter) length so the fading trail stays legible.
    """

    def __init__(self, maxlen: int = DEFAULT_MAXLEN,
                 trail_len: int = DEFAULT_TRAIL_LEN) -> None:
        """Create empty buffers.

        Args:
            maxlen: Maximum number of time-series samples retained. Must be a
                positive integer.
            trail_len: Maximum number of recent ``(n1, n2)`` tip points kept for
                the History_Trail. Must be a positive integer.
        """
        if maxlen <= 0:
            raise ValueError("maxlen must be a positive integer")
        if trail_len <= 0:
            raise ValueError("trail_len must be a positive integer")

        self.maxlen = maxlen
        self.trail_len = trail_len

        # Time-series windows (one shared length).
        self.time: Deque[float] = deque(maxlen=maxlen)
        self.v_in1: Deque[float] = deque(maxlen=maxlen)
        self.v_in2: Deque[float] = deque(maxlen=maxlen)
        self.v_out: Deque[float] = deque(maxlen=maxlen)
        self.I1: Deque[float] = deque(maxlen=maxlen)
        self.I2: Deque[float] = deque(maxlen=maxlen)

        # Vector tip trail (its own, usually shorter, length).
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=trail_len)

    def push(self, t: float, v_in1: float, v_in2: float, v_out: float,
             I1: float, I2: float, n1: float, n2: float) -> None:
        """Append one timestep's sample to every buffer.

        The oldest sample in each full buffer is dropped automatically. The
        ``(n1, n2)`` pair is appended to the trail as a single tip point.

        Args:
            t: Simulated time of this sample (ms).
            v_in1: Input_Neuron 1 membrane potential (mV).
            v_in2: Input_Neuron 2 membrane potential (mV).
            v_out: Output_Neuron membrane potential (mV).
            I1: Current injected into Input_Neuron 1 (nA).
            I2: Current injected into Input_Neuron 2 (nA).
            n1: Vector_N component ``N_1``.
            n2: Vector_N component ``N_2``.
        """
        self.time.append(t)
        self.v_in1.append(v_in1)
        self.v_in2.append(v_in2)
        self.v_out.append(v_out)
        self.I1.append(I1)
        self.I2.append(I2)
        self.trail.append((n1, n2))

    def clear(self) -> None:
        """Empty every buffer (used when the engine resets)."""
        self.time.clear()
        self.v_in1.clear()
        self.v_in2.clear()
        self.v_out.clear()
        self.I1.clear()
        self.I2.clear()
        self.trail.clear()

    def potentials(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(time, v_in1, v_in2, v_out)`` as NumPy arrays for plotting.

        Supports the Membrane_Potential time-series graphic (Req 9.1).
        """
        return (
            np.fromiter(self.time, dtype=float, count=len(self.time)),
            np.fromiter(self.v_in1, dtype=float, count=len(self.v_in1)),
            np.fromiter(self.v_in2, dtype=float, count=len(self.v_in2)),
            np.fromiter(self.v_out, dtype=float, count=len(self.v_out)),
        )

    def currents(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(time, I1, I2)`` as NumPy arrays for plotting.

        Supports the Input_Current time-series graphic (Req 10.1).
        """
        return (
            np.fromiter(self.time, dtype=float, count=len(self.time)),
            np.fromiter(self.I1, dtype=float, count=len(self.I1)),
            np.fromiter(self.I2, dtype=float, count=len(self.I2)),
        )

    def trail_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the History_Trail tips as ``(n1_values, n2_values)`` arrays.

        The arrays are ordered oldest-to-newest, matching the fade from faint
        (old) to bright (new) in the Vector_N graphic (Req 11.2). Both arrays
        are empty when no samples have been pushed yet.
        """
        count = len(self.trail)
        n1 = np.empty(count, dtype=float)
        n2 = np.empty(count, dtype=float)
        for i, (a, b) in enumerate(self.trail):
            n1[i] = a
            n2[i] = b
        return n1, n2

    def __len__(self) -> int:
        """Number of time-series samples currently stored."""
        return len(self.time)


@dataclass
class StepResult:
    """Snapshot of one timestep's outcome, returned by :meth:`SimulationEngine.step`.

    Carries everything the GUI and the unit tests need to inspect a single
    step without reaching into the engine's internal objects: the three
    membrane potentials, the two injected input currents, the Vector_N
    components after the step, and the three spike flags for this step.

    Units convention (mV-ms-nA-MOhm), matching the rest of ``core``:
    - Membrane potentials (``v_in1``, ``v_in2``, ``v_out``): millivolts (mV).
    - Input currents (``I1``, ``I2``): nanoamperes (nA).
    - Vector_N components (``n1``, ``n2``): dimensionless accumulator units.
    - Spike flags (``s_in1``, ``s_in2``, ``s_out``): ``True`` if that neuron
      spiked on this step.
    """

    v_in1: float   # Input_Neuron 1 membrane potential (mV) after the step
    v_in2: float   # Input_Neuron 2 membrane potential (mV) after the step
    v_out: float   # Output_Neuron membrane potential (mV) after the step
    I1: float      # current injected into Input_Neuron 1 this step (nA)
    I2: float      # current injected into Input_Neuron 2 this step (nA)
    n1: float      # Vector_N component N_1 after the step
    n2: float      # Vector_N component N_2 after the step
    s_in1: bool    # did Input_Neuron 1 spike this step
    s_in2: bool    # did Input_Neuron 2 spike this step
    s_out: bool    # did the Output_Neuron spike this step


class SimulationEngine:
    """The Qt-free simulation core: two inputs, one output, and Vector_N.

    Owns the fixed two-input / one-output network topology (Req 1), the two
    live-switchable input current generators (Req 5), the two synaptic weights
    (Req 6), the central :class:`VectorN` state (Req 7), the integration
    timestep ``dt`` and simulated time ``t`` (Req 12), and the rolling
    :class:`HistoryBuffers` the GUI plots.

    All parameters are plain attributes (or attributes on the owned neurons and
    generators) so GUI controls can mutate them directly; the next call to
    :meth:`step` reads the updated values (live update, Req 4/5/6).

    Units convention (mV-ms-nA-MOhm) as elsewhere in ``core``.
    """

    def __init__(self) -> None:
        """Construct the network and state with the design defaults.

        Builds two input :class:`LIFNeuron`s (``tau_m=10``) and one output
        neuron (``tau_m=15``) per the design Data Models table, two pulse
        current generators, unit synaptic weights, an origin ``N``, the default
        ``dt`` of 0.1 ms, ``t=0``, and bounded history buffers.
        """
        # Two input neurons (shared defaults, tau_m = 10 ms).
        self.input1 = LIFNeuron(
            tau_m=10.0, R=10.0, v_rest=-65.0, v_th=-50.0,
            v_reset=-70.0, t_refrac=2.0,
        )
        self.input2 = LIFNeuron(
            tau_m=10.0, R=10.0, v_rest=-65.0, v_th=-50.0,
            v_reset=-70.0, t_refrac=2.0,
        )
        # One output neuron (tau_m = 15 ms; also drives N decay, Req 7.2/7.3).
        self.output = LIFNeuron(
            tau_m=15.0, R=10.0, v_rest=-65.0, v_th=-50.0,
            v_reset=-70.0, t_refrac=3.0,
        )

        # One live-switchable current generator per input (Req 5).
        self.gen1: CurrentGenerator = PulseCurrent(
            magnitude=20.0, period=50.0, width=5.0,
        )
        self.gen2: CurrentGenerator = PulseCurrent(
            magnitude=20.0, period=50.0, width=5.0,
        )

        # Synaptic weights: scale both the output current jump and the N
        # impulse for the matching input (Req 6).
        self.w1 = 1.0
        self.w2 = 1.0

        # Output-drive gain: a live multiplier applied ONLY to the weighted
        # current delivered to the output neuron, so the output can be made to
        # spike more readily by raising the gain WITHOUT changing the Vector_N
        # impulses (which stay at the raw w1/w2). This keeps the vector plot
        # scale unchanged while still letting the output fire.
        self.w_gain = 1.0

        # Central vector state (Req 7).
        self.N = VectorN()

        # Timing: fixed integration step and simulated time (Req 12).
        self.dt = 0.1   # ms
        self.t = 0.0    # ms

        # Rolling windows the GUI plots each tick (Req 9/10/11).
        self.history = HistoryBuffers()

    def step(self) -> StepResult:
        """Advance the whole network by one ``dt`` and return a :class:`StepResult`.

        The ordering is strict and correctness-relevant (Req 3, Req 7); it is
        held fixed so the within-step interactions are deterministic:

        1. Compute this step's injected input currents from the generators.
        2. Step both input neurons, capturing their spike flags.
        3. Decay ``N`` first, by the exact exponential ``exp(-dt / tau_out)``
           with ``tau_out = output.tau_m`` (Req 7.2, 7.3, 7.5). Exact decay
           keeps ``N`` stable even when ``tau_out`` is small relative to ``dt``.
        4. Add ``w_i`` to ``N_i`` as an instantaneous impulse for each input
           spike this step (Req 7.4).
        5. Apply ``w1*s1 + w2*s2`` as a one-step instantaneous current jump to
           the output neuron; simultaneous input spikes sum (Req 3.1, 3.3).
        6. If the output neuron spiked, reset ``N`` to the origin last, so an
           output spike always wins over the same-step input impulses (Req 7.6).
        7. Advance simulated time and push this step's sample to history.

        Returns:
            A :class:`StepResult` with the post-step potentials, the injected
            currents, the ``N`` components, and the three spike flags.
        """
        # 1. Input currents for this step (nA).
        I1 = self.gen1.value(self.t, self.dt)
        I2 = self.gen2.value(self.t, self.dt)

        # 2. Step both input neurons; capture spikes for this step.
        s1 = self.input1.step(self.dt, I1)
        s2 = self.input2.step(self.dt, I2)

        # 3. Decay N FIRST, before applying this step's impulses (Req 7.5).
        #    Exact exponential decay with tau = output.tau_m (Req 7.2/7.3).
        factor = math.exp(-self.dt / self.output.tau_m)
        self.N.n1 *= factor
        self.N.n2 *= factor

        # 4. Instantaneous input-spike impulses on N (Req 7.4).
        if s1:
            self.N.n1 += self.w1
        if s2:
            self.N.n2 += self.w2

        # 5. Instantaneous weighted current jump to the output neuron;
        #    simultaneous spikes sum (Req 3.1/3.3). The output-drive gain scales
        #    ONLY this current, not the N impulses above, so raising the gain
        #    makes the output spike more easily without changing the vector
        #    plot's scale.
        I_out = (self.w1 * s1 + self.w2 * s2) * self.w_gain
        s_out = self.output.step(self.dt, I_out)

        # 6. Output spike resets N last, overriding same-step impulses (Req 7.6).
        if s_out:
            self.N.reset()

        # 7. Advance time and record the sample for the plots.
        self.t += self.dt
        self.history.push(
            self.t,
            self.input1.v, self.input2.v, self.output.v,
            I1, I2,
            self.N.n1, self.N.n2,
        )

        return StepResult(
            v_in1=self.input1.v,
            v_in2=self.input2.v,
            v_out=self.output.v,
            I1=I1,
            I2=I2,
            n1=self.N.n1,
            n2=self.N.n2,
            s_in1=s1,
            s_in2=s2,
            s_out=s_out,
        )

    def reset(self) -> None:
        """Reset the whole simulation to a fresh start.

        Resets all three neurons' runtime state, snaps ``N`` to the origin,
        rewinds simulated time to zero, and clears the history buffers so the
        plots start empty. Current generators are reset to their initial phase
        where applicable (they are time-driven, so rewinding ``t`` already
        restarts their pattern; any generator exposing a ``reset`` is also
        invoked for completeness).
        """
        self.input1.reset_state()
        self.input2.reset_state()
        self.output.reset_state()
        self.N.reset()
        self.t = 0.0
        self.history.clear()

        # Reset any generator that exposes a reset hook. The stock generators
        # are purely time-driven (no internal phase state), so rewinding t
        # already restarts their pattern; this stays correct if a future
        # stateful generator is swapped in live.
        for gen in (self.gen1, self.gen2):
            reset = getattr(gen, "reset", None)
            if callable(reset):
                reset()
