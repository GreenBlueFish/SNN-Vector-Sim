# Design Document

## Overview

This document describes the design of a pure-Python graphical spiking neural network (SNN) prototype whose purpose is to explore representing a small network's activity as a 2D vector `N = {N_1, N_2}`. The network is a fixed topology of two input Leaky Integrate-and-Fire (LIF) neurons feeding one output LIF neuron through individually weighted, instantaneous synapses.

The application is a single-process, single-threaded, timer-driven simulation with a live GUI. Every meaningful quantity (per-neuron LIF parameters, input currents, synaptic weights, dt, speed) is adjustable while the simulation runs. The GUI renders four live visualizations: a network diagram, membrane potentials over time, input currents over time, and the vector `N` as an arrow with a history trail.

### Goals

- Faithful, inspectable LIF dynamics for all three neurons (Req 2).
- Instantaneous weighted synaptic current jumps to the output neuron (Req 3).
- Vector `N` dynamics driven by input spike impulses, decaying with the output neuron's `tau_m`, reset to origin on output spikes (Req 7).
- Fully live controls with no stop/restart required (Req 4, 5, 6).
- Smooth real-time plotting with pause/resume, adjustable dt, and slow-motion speed (Req 9–14).

### Non-Goals

- Networks other than 2-input/1-output (fixed by Req 1).
- Decaying/alpha synaptic currents (explicitly instantaneous per Req 3).
- Persistence, data export, or multi-session management.

## Technology Choices

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.9+ | Required (Req 15). |
| GUI + plotting | **PyQtGraph** on **PySide6** (Qt) | PyQtGraph is purpose-built for fast, continuously updating scientific plots and shares the Qt event loop with native control widgets (sliders, spin boxes, combo boxes, buttons). This lets four plots plus a dense control panel live in one responsive window. |
| Numerics | NumPy | Simple vector state and rolling buffers. |
| Timing | `QtCore.QTimer` | Drives the simulation step + render from the Qt event loop, keeping everything single-threaded. |

### Considered alternative

- **Matplotlib + widgets**: pure Python and simple, but redrawing four live axes via `FuncAnimation` at interactive rates is sluggish and its slider/button widgets are coarse. Rejected for the real-time, control-heavy requirement.
- **Threaded engine**: a background thread stepping the engine while Qt renders. Rejected as unnecessary for a network this small; a single QTimer stepping N integration steps per tick is simpler and avoids cross-thread state races on live-edited parameters.

> PySide6 is assumed. PyQt5 works with trivial import changes; the code isolates the Qt import so the binding can be swapped in one place.

## Units Convention

To keep the LIF math clean and avoid unit confusion, the simulation uses a **consistent normalized/biophysical mV–ms–nA–MΩ convention**:

- Time: **milliseconds (ms)**. `dt` default `0.1 ms` (Req 12).
- Membrane potential: **millivolts (mV)**.
- `tau_m`: **ms**.
- Membrane resistance `R`: **MΩ**.
- Current `I`: **nA** (so `R·I` is in mV, consistent with the voltage equation).
- Weights `w_i`: dimensionless gain applied to current jumps (nA delivered per presynaptic spike) and to `N` impulses.

The exact scale is not physically critical for this exploratory tool; what matters is internal consistency so `R·I` has voltage units.

## Architecture

```mermaid
graph TD
    subgraph Engine["Simulation core (no Qt)"]
        CG1["CurrentGenerator (input 1)"]
        CG2["CurrentGenerator (input 2)"]
        IN1["LIFNeuron: input 1"]
        IN2["LIFNeuron: input 2"]
        OUT["LIFNeuron: output"]
        VEC["VectorN state (N1, N2)"]
        ENG["SimulationEngine (owns neurons, weights, N, history buffers)"]
        CG1 --> ENG
        CG2 --> ENG
        ENG --> IN1
        ENG --> IN2
        ENG --> OUT
        ENG --> VEC
    end

    subgraph UI["GUI (PySide6 + PyQtGraph)"]
        TIMER["QTimer tick"]
        CTRL["Control panel widgets"]
        NETPLOT["Network diagram"]
        VPLOT["Membrane potential plot"]
        IPLOT["Input current plot"]
        NPLOT["Vector N plot + trail"]
    end

    TIMER -->|step(dt) x stepsPerTick| ENG
    CTRL -->|live writes| ENG
    ENG -->|history buffers| VPLOT
    ENG -->|history buffers| IPLOT
    ENG -->|N1,N2 + trail| NPLOT
    ENG -->|weights| NETPLOT
    TIMER -->|refresh| VPLOT & IPLOT & NPLOT & NETPLOT
```

The simulation core has **no Qt dependency** so it can be unit-tested headlessly. The GUI layer owns the timer, reads engine state for rendering, and writes control values straight into engine/neuron attributes.

### Module layout

```
snn_vector_sim/
├── core/
│   ├── lif_neuron.py        # LIFNeuron
│   ├── current.py           # CurrentGenerator, ConstantCurrent, PulseCurrent
│   └── engine.py            # SimulationEngine, VectorN, history buffers
├── gui/
│   ├── main_window.py       # MainWindow: layout, timer, wiring
│   ├── controls.py          # NeuronControlPanel, InputControlPanel, SimControls
│   └── plots.py             # NetworkDiagram, PotentialPlot, CurrentPlot, VectorPlot
├── app.py                   # entry point: build engine + window, start Qt loop
├── tests/
│   ├── test_lif_neuron.py
│   ├── test_current.py
│   └── test_engine.py
└── requirements.txt
```

## Components and Interfaces

### LIFNeuron (`core/lif_neuron.py`)

Holds one neuron's LIF state and parameters and advances it one timestep.

```python
@dataclass
class LIFNeuron:
    tau_m: float = 10.0        # ms  (>0)
    R: float = 10.0            # MΩ  (>0)
    v_rest: float = -65.0      # mV
    v_th: float = -50.0        # mV  (> v_reset, > v_rest sensible)
    v_reset: float = -70.0     # mV
    t_refrac: float = 2.0      # ms  (>=0)
    # runtime state
    v: float = field(init=False)        # current membrane potential
    refrac_left: float = 0.0            # ms remaining in refractory
    spiked: bool = False                # did it spike on the last step

    def reset_state(self): ...          # v = v_rest, refrac_left = 0

    def step(self, dt: float, i_input: float) -> bool:
        """Advance by dt (ms) with input current i_input (nA).
        Returns True if the neuron spiked this step."""
```

`step` semantics (Req 2):

1. If `refrac_left > 0`: hold `v = v_reset`, decrement `refrac_left` by `dt`, set `spiked = False`, return `False` (no integration, no spike).
2. Otherwise integrate forward-Euler: `dv = (-(v - v_rest) + R * i_input) / tau_m * dt; v += dv`.
3. If `v >= v_th`: set `v = v_reset`, `refrac_left = t_refrac`, `spiked = True`, return `True`.
4. Else `spiked = False`, return `False`.

Parameters are plain attributes so GUI controls mutate them directly and the next `step` reads the new values (Req 4.2, live update).

### Current generators (`core/current.py`)

Produce the externally injected current for an input neuron each timestep (Req 5). Live-switchable per input.

```python
class CurrentGenerator(Protocol):
    def value(self, t: float, dt: float) -> float: ...   # returns nA for this step

@dataclass
class ConstantCurrent:
    magnitude: float = 0.0
    def value(self, t, dt): return self.magnitude

@dataclass
class PulseCurrent:
    magnitude: float = 20.0
    period: float = 50.0      # ms between pulse onsets
    width: float = 5.0        # ms each pulse stays on
    def value(self, t, dt):
        phase = t % self.period
        return self.magnitude if phase < self.width else 0.0
```

- Constant injects `magnitude` every step (Req 5.3).
- Pulse injects `magnitude` for `width` ms within each `period`, else `0` (Req 5.4). A repeating pulse train is more useful than a one-shot for watching patterns; `period >= width`.
- The engine stores one generator per input; swapping type or editing fields is a direct attribute write applied on the next step (Req 5.5).

### VectorN and SimulationEngine (`core/engine.py`)

```python
@dataclass
class VectorN:
    n1: float = 0.0
    n2: float = 0.0
    def reset(self): self.n1 = 0.0; self.n2 = 0.0

class SimulationEngine:
    def __init__(self):
        self.input1 = LIFNeuron(...)
        self.input2 = LIFNeuron(...)
        self.output = LIFNeuron(...)
        self.gen1: CurrentGenerator = ConstantCurrent(0.0)
        self.gen2: CurrentGenerator = ConstantCurrent(0.0)
        self.w1 = 1.0
        self.w2 = 1.0
        self.N = VectorN()
        self.dt = 0.1          # ms (Req 12.2)
        self.t = 0.0           # ms simulated time
        self.history = HistoryBuffers(maxlen=...)

    def step(self) -> StepResult: ...
    def reset(self): ...
```

`step()` executes **one timestep** with this strict ordering (ordering matters for correctness, Req 3, Req 7):

```mermaid
sequenceDiagram
    participant E as Engine.step()
    participant G as Generators
    participant I1 as Input1
    participant I2 as Input2
    participant O as Output
    participant N as VectorN

    E->>G: I1 = gen1.value(t,dt); I2 = gen2.value(t,dt)
    E->>I1: s1 = input1.step(dt, I1)
    E->>I2: s2 = input2.step(dt, I2)
    Note over E,N: decay N first (one dt of exp decay)
    E->>N: n1 *= exp(-dt/tau_out); n2 *= exp(-dt/tau_out)
    alt s1 spiked
        E->>N: n1 += w1
    end
    alt s2 spiked
        E->>N: n2 += w2
    end
    E->>O: I_out = w1*s1 + w2*s2  (instantaneous jump, nA for this step)
    E->>O: s_out = output.step(dt, I_out)
    alt s_out spiked
        E->>N: N.reset()  (set n1=n2=0)
    end
    E->>E: t += dt; push history
```

Rationale for the order:

- Input neurons step first so their spikes are known for this timestep.
- `N` **decays before** applying this step's impulses (Req 7.5, deterministic within-step order). Decay uses the exact exponential `exp(-dt/tau_out)` where `tau_out = output.tau_m` (Req 7.2, 7.3) — exact decay keeps `N` stable even if `tau_out` is small relative to `dt`.
- Input spike `i` adds `w_i` to `N_i` as an instantaneous impulse (Req 7.4).
- The output neuron receives `w1*s1 + w2*s2` as a one-step instantaneous current jump; simultaneous input spikes sum (Req 3.1, 3.3).
- If the output neuron spikes this step, `N` is reset to the origin last (Req 7.6), so an output spike always wins over the same-step impulses.

`StepResult` carries `(v_in1, v_in2, v_out, I1, I2, n1, n2, s_in1, s_in2, s_out)` for the GUI/tests.

#### History buffers

`HistoryBuffers` wraps fixed-length `collections.deque` (or preallocated NumPy ring buffers) for: simulated time, `v` of all three neurons, `I1`, `I2`, and the `(n1, n2)` trail. A bounded length (e.g. last ~5–10 seconds of simulated time, capped by sample count) keeps memory and redraw cost constant. The vector trail keeps the last K tip points for the fading trail.

### GUI layer

#### MainWindow (`gui/main_window.py`)

Owns the engine, the QTimer, all widgets, and the wiring. Layout (Qt splitters/grid):

- Left/top: **plots** — network diagram, membrane potential plot, input current plot, vector `N` plot.
- Right/bottom: **control panel** — three neuron panels, two input panels, global sim controls.

Timer loop (one tick):

```python
def on_tick(self):
    if not self.paused:
        for _ in range(self.steps_per_tick):
            self.engine.step()
    self.refresh_plots()     # always refresh so paused state still renders (Req 13.4)
```

#### Plots (`gui/plots.py`)

- **NetworkDiagram**: a `pg.GraphicsLayoutWidget`/`ViewBox` with three node markers (input1, input2, output) and two edges. Edge labels show live `w1`, `w2` (Req 8). Optionally flash a node when it spikes.
- **PotentialPlot**: one `PlotWidget` with three curves (input1, input2, output), updated via `setData` from the history buffers each tick (Req 9). Threshold lines optional.
- **CurrentPlot**: one `PlotWidget` with two curves `I1`, `I2` (Req 10).
- **VectorPlot**: a `PlotWidget` on the `N_1`–`N_2` plane showing (a) an arrow from origin to `(n1, n2)` using `pg.ArrowItem` + a line, and (b) a fading `History_Trail` of recent tips as a line/scatter whose alpha ramps from faint (old) to bright (new) (Req 11). Axes auto-range or fixed symmetric range.

Efficient updates: `PlotDataItem`s are created once; each tick calls `setData` with the current buffer contents. The arrow position/trail are recomputed each tick. No new items are allocated per frame.

#### Controls (`gui/controls.py`)

- **NeuronControlPanel** (x3): labeled `QDoubleSpinBox` (optionally paired with a slider) for `tau_m`, `v_rest`, `v_th`, `v_reset`, `R`, `t_refrac` (Req 4.1). Each widget's `valueChanged` writes directly to the corresponding `LIFNeuron` attribute (Req 4.2–4.4). The output neuron's `tau_m` control thereby also changes the `N` decay constant (Req 7.3) with no extra wiring.
- **InputControlPanel** (x2): a `QComboBox` for type (Constant/Pulse); a magnitude spin box; pulse `period` and `width` spin boxes enabled only in Pulse mode (Req 5). Changing type swaps `engine.genN` to a new generator preserving magnitude; editing fields writes through.
- **WeightControls**: two spin boxes for `w1`, `w2` writing `engine.w1/.w2` (Req 6).
- **SimControls**:
  - `dt` spin box → `engine.dt` (Req 12.3–12.4), bounded (e.g. 0.01–10 ms).
  - **Speed**: a slider (integer-backed, mapped to 0.01–1.0) and a synced `QDoubleSpinBox`. Both edit a single `speed` value; each updates the other with signals blocked to avoid loops (Req 14.3–14.4). Out-of-range/invalid numeric entry is clamped/rejected and the last valid value retained (Req 14.5 intent).
  - **Pause/Resume**: a checkable `QPushButton` toggling `self.paused` (Req 13).

### dt vs. speed (timing model)

Two independent knobs:

- **`dt`** is the integration step — numerical fidelity (Req 12). Smaller `dt` = more accurate dynamics, more steps to cover the same simulated time.
- **`speed`** is a **slow-motion multiplier** `M ∈ [0.01, 1.0]` controlling how much *simulated* time elapses per unit *wall-clock* time (Req 14).

The timer fires at a fixed wall-clock interval `T_tick` (e.g. 20 ms → ~50 fps). Target simulated time advanced per tick is `M · T_tick`. Steps per tick:

```
steps_per_tick = max(1, round(M * T_tick / dt))
```

So at `M = 1.0`, `T_tick = 20 ms`, `dt = 0.1 ms` → 200 steps/tick (near real time). At `M = 0.1` → 20 steps/tick (10× slow motion). An accumulator carries the fractional remainder across ticks so average speed stays accurate (Req 14.6). Changing `dt` or `M` just changes `steps_per_tick` on the next tick — no restart (Req 12.4, 14.6). Pause sets `steps_per_tick` effectively to 0 while rendering continues (Req 13).

> If very small `M` with small `dt` would make a tick drop below 1 step, the accumulator defers stepping across ticks so arbitrarily slow motion is possible without starving the UI.

## Data Models

### Default parameters (working demo)

| Parameter | Input neurons | Output neuron | Notes |
|---|---|---|---|
| `tau_m` | 10 ms | 15 ms | Output `tau_m` also drives `N` decay |
| `R` | 10 MΩ | 10 MΩ | `R·I` in mV |
| `v_rest` | -65 mV | -65 mV | |
| `v_th` | -50 mV | -50 mV | |
| `v_reset` | -70 mV | -70 mV | |
| `t_refrac` | 2 ms | 3 ms | |

| Engine field | Default |
|---|---|
| `w1`, `w2` | 1.0, 1.0 (weights scale both output current jump and N impulse) |
| `gen1`, `gen2` | PulseCurrent(magnitude=20 nA, period=50 ms, width=5 ms) |
| `dt` | 0.1 ms |
| `speed` M | 1.0 |
| `T_tick` | 20 ms |
| history length | ~2000–5000 samples (bounded) |

With `R=10`, a 20 nA input drives `R·I = 200` mV of drive scaled by the leak, pushing input neurons across threshold periodically — a visible, tunable spiking regime out of the box.

## Error Handling

The sim must keep running under live edits (Req 4.3, 5.5, 6.3). Strategy:

- **Bounds at the widget**: spin boxes enforce ranges — `tau_m` and `R` have a strictly positive minimum (e.g. `0.001`), `dt ∈ [0.01, 10]`, `speed ∈ [0.01, 1.0]`, `t_refrac >= 0`. This structurally prevents divide-by-zero in the LIF update and the `N` decay.
- **Ordering sanity**: `v_th`, `v_reset`, `v_rest` are not hard-constrained against each other (a researcher may want odd configs), but a non-blocking visual warning is shown if `v_th <= v_reset` (which would spike every step). The sim still runs.
- **Speed field validation**: out-of-range or malformed numeric entry is clamped to the valid range (or reverted to the last valid value) and the slider re-synced; the simulation is not interrupted (Req 14.5 intent).
- **Generator swap safety**: switching current type constructs the new generator with current magnitude before replacing the reference, so no step sees a partially configured generator.

Because invalid values are prevented at the input boundary, the engine core can assume valid parameters and stays branch-light in the hot loop.

## Testing Strategy

Testing is proportionate to a single-user research prototype: the **Qt-free core is unit-tested**; the GUI is smoke/manually tested.

### Unit tests (pytest, headless)

`test_lif_neuron.py`
- Sub-threshold: with zero input, `v` decays from an offset toward `v_rest` with the expected `tau_m` (within tolerance).
- Spiking: a constant supra-threshold current produces a spike; on spike `v` is set to `v_reset` and `step` returns `True`.
- Refractory: no integration/spiking while `refrac_left > 0`; integration resumes after `t_refrac`.
- Steady state: constant sub-threshold current converges to `v_rest + R·I`.

`test_current.py`
- Constant returns `magnitude` for all `t`.
- Pulse returns `magnitude` only within `width` of each `period`, `0` otherwise; verified across several periods.

`test_engine.py`
- Input spike impulse: forcing input1 to spike adds exactly `w1` to `N_1`.
- Decay: with no spikes, `N_i` decays by `exp(-dt/tau_out)` per step; changing `output.tau_m` changes the decay rate.
- Output reset: forcing the output neuron to spike sets `N` to `(0, 0)` in that step, overriding same-step input impulses.
- Simultaneous input spikes: output current equals `w1 + w2` that step.
- Within-step order: decay-before-impulse verified numerically.
- Timing model: `steps_per_tick` and the accumulator produce the expected average simulated-time advance for representative `(M, dt, T_tick)`.

### Manual / smoke testing (GUI)

- App launches and shows all four plots plus the control panel.
- Live edits to each neuron's parameters, both input generators, both weights, `dt`, and speed take effect without stalling or restarting.
- Pause freezes dynamics while plots keep rendering the frozen state; resume continues.
- Reducing speed visibly slows the animation; the slider and numeric field stay in sync.
- The `N` arrow grows on input spikes and snaps to the origin on output spikes, with a visible fading trail.

## Dependencies and Running

`requirements.txt`:

```
PySide6>=6.5
pyqtgraph>=0.13
numpy>=1.23
```

Run:

```
pip install -r requirements.txt
python -m snn_vector_sim.app
```

(Tests: `pip install pytest` then `pytest snn_vector_sim/tests`.)

## Requirements Traceability

| Requirement | Design coverage |
|---|---|
| 1 Fixed topology | `SimulationEngine` fixes two inputs + one output and two weights; no API to change structure. |
| 2 LIF dynamics | `LIFNeuron.step` (integration, threshold, reset, refractory). |
| 3 Instantaneous synaptic current | Engine sets `I_out = w1*s1 + w2*s2` one-step jump; simultaneous spikes sum. |
| 4 Live LIF params | `NeuronControlPanel` writes `LIFNeuron` attributes read next step. |
| 5 Live input currents | `ConstantCurrent`/`PulseCurrent` + `InputControlPanel` type/magnitude/pulse controls. |
| 6 Live weights | `WeightControls` → `engine.w1/.w2`, used for jumps and N impulses. |
| 7 Vector N dynamics | Engine step order: decay by `exp(-dt/tau_out)`, `+w_i` on input spike, reset on output spike; `tau_out = output.tau_m`. |
| 8 Network graphic | `NetworkDiagram` with nodes, edges, live weight labels. |
| 9 Membrane potential graphic | `PotentialPlot`, 3 curves, updated each tick. |
| 10 Input current graphic | `CurrentPlot`, 2 curves, updated each tick. |
| 11 Vector N graphic | `VectorPlot` arrow + fading history trail. |
| 12 dt control | `engine.dt` + `dt` spin box, default 0.1 ms. |
| 13 Pause/resume | `paused` flag; stepping halts, rendering continues. |
| 14 Speed control | Slider + synced numeric field; slow-motion multiplier via `steps_per_tick` + accumulator. |
| 15 Pure Python | Python + PySide6/PyQtGraph. |
