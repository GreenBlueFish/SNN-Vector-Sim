# Implementation Plan

- [x] 1. Scaffold project structure and dependencies
  - Create the `snn_vector_sim/` package with `core/`, `gui/`, and `tests/` subpackages and `__init__.py` files.
  - Write `requirements.txt` with `PySide6>=6.5`, `pyqtgraph>=0.13`, `numpy>=1.23`.
  - Create an empty `app.py` entry-point stub.
  - _Requirements: 15.1, 15.2_

- [x] 2. Implement the LIF neuron core
  - [x] 2.1 Implement `LIFNeuron` in `core/lif_neuron.py`
    - Define the dataclass with parameters (`tau_m`, `R`, `v_rest`, `v_th`, `v_reset`, `t_refrac`) and runtime state (`v`, `refrac_left`, `spiked`), initializing `v` to `v_rest`.
    - Implement `reset_state()` and `step(dt, i_input) -> bool` with forward-Euler integration, threshold detection, reset, and refractory hold per the design ordering.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [x] 2.2 Write unit tests for `LIFNeuron`
    - Test sub-threshold decay toward `v_rest`, steady state at `v_rest + R·I`, spiking on supra-threshold current with reset to `v_reset`, and refractory hold then resumption.
    - Run the tests and confirm they pass.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3. Implement current generators
  - [x] 3.1 Implement generators in `core/current.py`
    - Define the `CurrentGenerator` protocol/interface and `ConstantCurrent(magnitude)` and `PulseCurrent(magnitude, period, width)` with a `value(t, dt)` method.
    - _Requirements: 5.3, 5.4_
  - [x] 3.2 Write unit tests for current generators
    - Test constant output across time and pulse on/off behavior across several periods.
    - Run the tests and confirm they pass.
    - _Requirements: 5.3, 5.4_

- [x] 4. Implement the simulation engine and vector N dynamics
  - [x] 4.1 Implement `VectorN` and history buffers in `core/engine.py`
    - Implement `VectorN` with `n1`, `n2`, and `reset()`.
    - Implement `HistoryBuffers` using bounded deques/ring buffers for time, three membrane potentials, two input currents, and the N tip trail.
    - _Requirements: 7.6, 9.1, 10.1, 11.2_
  - [x] 4.2 Implement `SimulationEngine` with the ordered step
    - Construct two input neurons, one output neuron, two generators, weights `w1`/`w2`, `N`, `dt`, and `t` with the design defaults.
    - Implement `step()` with the strict ordering: compute input currents, step input neurons, decay `N` by `exp(-dt/output.tau_m)`, add `w_i` impulse per input spike, apply `w1*s1 + w2*s2` instantaneous jump to the output neuron, reset `N` on output spike, advance `t`, and push history. Return a `StepResult`.
    - Implement `reset()`.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 4.3 Write unit tests for the engine
    - Test: input spike adds `w_i` to `N_i`; N decays by `exp(-dt/tau_out)` with no spikes and the rate tracks `output.tau_m`; output spike resets N to `(0,0)` overriding same-step impulses; simultaneous input spikes sum to `w1+w2` output current; decay-before-impulse ordering.
    - Run the tests and confirm they pass.
    - _Requirements: 3.1, 3.3, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 5. Implement the timing model in the engine/driver
  - Add a `steps_per_tick` computation `max(1, round(M * T_tick / dt))` with a fractional accumulator so slow-motion and sub-one-step-per-tick cases advance correctly over time, independent of `dt`.
  - Write a unit test verifying average simulated-time advance for representative `(M, dt, T_tick)` including a very small `M`.
  - Run the test and confirm it passes.
  - _Requirements: 12.1, 12.4, 14.5, 14.6_

- [x] 6. Build the plot widgets
  - [x] 6.1 Implement `NetworkDiagram` in `gui/plots.py`
    - Draw two input nodes, one output node, and two edges with live weight labels; expose an `update(engine)` method.
    - _Requirements: 8.1, 8.2_
  - [x] 6.2 Implement `PotentialPlot` and `CurrentPlot`
    - `PotentialPlot`: one plot with three `PlotDataItem` curves for the three membrane potentials, updated via `setData` from history buffers.
    - `CurrentPlot`: one plot with two curves for the two input currents.
    - _Requirements: 9.1, 9.2, 10.1, 10.2_
  - [x] 6.3 Implement `VectorPlot`
    - Draw the N arrow from origin to `(n1, n2)` and a fading history trail of recent tips in the `N_1`–`N_2` plane; update each tick.
    - _Requirements: 11.1, 11.2, 11.3_

- [x] 7. Build the control panels
  - [x] 7.1 Implement `NeuronControlPanel` in `gui/controls.py`
    - Create a reusable panel with bounded spin boxes for `tau_m`, `v_rest`, `v_th`, `v_reset`, `R`, `t_refrac` whose `valueChanged` writes directly to a target `LIFNeuron`. Enforce `tau_m`/`R` strictly positive via widget minimums. Instantiate one per neuron.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 7.3_
  - [x] 7.2 Implement `InputControlPanel`
    - Combo box for Constant/Pulse that swaps `engine.genN` preserving magnitude; magnitude spin box; pulse `period`/`width` spin boxes enabled only in Pulse mode. All edits applied live.
    - _Requirements: 5.1, 5.2, 5.5_
  - [x] 7.3 Implement `WeightControls` and `SimControls`
    - `WeightControls`: two spin boxes writing `engine.w1`/`engine.w2`.
    - `SimControls`: `dt` spin box (bounded) writing `engine.dt`; speed slider + synced numeric field (0.01–1.0) with signal-blocked two-way sync and clamping of invalid entries; checkable pause/resume button.
    - _Requirements: 6.1, 6.2, 6.3, 12.2, 12.3, 12.4, 13.1, 14.1, 14.2, 14.3, 14.4, 14.5_

- [x] 8. Assemble the main window and wire the timer
  - [x] 8.1 Implement `MainWindow` in `gui/main_window.py`
    - Lay out the four plots and the control panels; construct one engine; instantiate control panels bound to the engine's neurons and fields.
    - _Requirements: 4.1, 5.1, 6.1, 8.1, 9.1, 10.1, 11.1_
  - [x] 8.2 Implement the QTimer tick loop
    - On each tick, if not paused run `steps_per_tick` engine steps using the accumulator, then always refresh all plots so paused state keeps rendering the frozen state.
    - _Requirements: 9.2, 10.2, 11.3, 12.4, 13.2, 13.3, 13.4, 14.6_

- [x] 9. Wire the entry point and verify end to end
  - Complete `app.py` to build the engine + `MainWindow`, start the Qt event loop, and run via `python -m snn_vector_sim.app`.
  - Run the full unit-test suite and confirm all core tests pass.
  - Launch the app and smoke-test: all four graphics render; live edits to neurons, inputs, weights, dt, and speed take effect without restart; pause freezes dynamics while rendering continues; slowing speed visibly slows motion with slider/field synced; the N arrow grows on input spikes and resets on output spikes with a visible trail.
  - _Requirements: 9.2, 10.2, 11.3, 13.2, 13.3, 13.4, 14.6, 15.1, 15.2_
