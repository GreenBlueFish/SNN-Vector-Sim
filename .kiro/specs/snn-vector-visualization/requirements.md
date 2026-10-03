# Requirements Document

## Introduction

This feature is a pure-Python graphical prototype for exploring a spiking neural network (SNN) idea: representing the activity of a small network as a 2D vector. The network has a fixed topology of exactly two input neurons feeding a single output neuron through individually weighted synapses. All three neurons use Leaky Integrate-and-Fire (LIF) dynamics with instantaneous synaptic current jumps.

The core purpose is to visualize a 2D vector `N = {N_1, N_2}` as an arrow (with a history trail) in an `N_1` vs `N_2` plane. Each component `N_i` is driven by the spike train of input neuron `i`, decays with the output neuron's membrane time constant, and resets to the origin whenever the output neuron spikes.

The prototype is a single-user research tool. Every meaningful quantity (per-neuron LIF parameters, input currents, synaptic weights) is adjustable live while the simulation runs. The user can pause, resume, and slow the simulation to observe spike-driven patterns as they happen. The GUI framework is left to design discretion, prioritizing responsive live controls and real-time plotting.

## Glossary

- **System**: The complete Python SNN simulation application, including the simulation engine and the graphical user interface.
- **Simulation_Engine**: The component that advances the network state by fixed timesteps and computes neuron dynamics.
- **GUI**: The graphical user interface component that renders visualizations and exposes live controls.
- **LIF_Neuron**: A neuron governed by Leaky Integrate-and-Fire dynamics.
- **Input_Neuron**: One of the two LIF neurons that receive injected input current. Indexed `i ∈ {1, 2}`.
- **Output_Neuron**: The single LIF neuron that receives weighted spikes from both input neurons.
- **Membrane_Potential**: The time-varying membrane voltage `V` of a neuron.
- **tau_m**: Membrane time constant of a neuron, governing the leak rate of its membrane potential.
- **Resting_Potential**: The membrane potential a neuron leaks toward in the absence of input.
- **Threshold**: The membrane potential value at or above which a neuron emits a spike.
- **Reset_Potential**: The membrane potential a neuron is set to immediately after it spikes.
- **Membrane_Resistance**: The resistance parameter `R` (or capacitance equivalent) scaling input current's effect on membrane potential.
- **Refractory_Period**: A duration after a spike during which a neuron does not integrate input or spike.
- **Spike**: A discrete event emitted by a neuron when its membrane potential reaches threshold.
- **Spike_Train**: The sequence of spike events `s_i(t)` emitted by neuron `i` over time.
- **Synaptic_Weight**: A scalar (`w_1` or `w_2`) scaling the current jump delivered to the Output_Neuron when the corresponding Input_Neuron spikes.
- **Instantaneous_Current_Jump**: A synaptic model in which an incoming spike causes an immediate step change in the receiving neuron's input, with no decaying synaptic current.
- **Input_Current**: An externally injected current delivered to an Input_Neuron, of type pulse or constant.
- **Pulse_Current**: An Input_Current applied as a brief, bounded-duration injection.
- **Constant_Current**: An Input_Current held at a fixed magnitude continuously.
- **Vector_N**: The 2D vector `N = {N_1, N_2}` that is the central object of visualization.
- **N_i**: Component `i` of Vector_N, driven by Input_Neuron `i`'s Spike_Train.
- **N_Plane**: The 2D Cartesian plane with axes `N_1` and `N_2` in which Vector_N is drawn.
- **History_Trail**: The rendered trace of recent Vector_N tip positions over time.
- **dt**: The fixed simulation timestep in milliseconds.
- **Simulation_Speed**: The ratio controlling how fast simulated time advances relative to wall-clock time.

## Requirements

### Requirement 1: Fixed Network Topology

**User Story:** As a researcher, I want a fixed two-input, one-output network topology, so that I can study vector dynamics without changing the network structure.

#### Acceptance Criteria

1. THE System SHALL instantiate exactly two Input_Neurons and exactly one Output_Neuron.
2. THE System SHALL connect each Input_Neuron to the Output_Neuron through that Input_Neuron's own Synaptic_Weight.
3. THE System SHALL associate Synaptic_Weight `w_1` with Input_Neuron 1 and Synaptic_Weight `w_2` with Input_Neuron 2.
4. THE System SHALL keep the network topology unchanged for the entire duration of a simulation session.

### Requirement 2: LIF Neuron Dynamics

**User Story:** As a researcher, I want all neurons to follow Leaky Integrate-and-Fire dynamics, so that spiking behavior is biologically motivated and consistent across the network.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL compute each LIF_Neuron's Membrane_Potential using Leaky Integrate-and-Fire dynamics parameterized by that neuron's tau_m, Resting_Potential, Threshold, Reset_Potential, and Membrane_Resistance.
2. WHILE a LIF_Neuron is not in its Refractory_Period, THE Simulation_Engine SHALL integrate that neuron's input current into its Membrane_Potential each timestep.
3. WHEN a LIF_Neuron's Membrane_Potential reaches or exceeds that neuron's Threshold, THE Simulation_Engine SHALL emit a Spike for that neuron.
4. WHEN a LIF_Neuron emits a Spike, THE Simulation_Engine SHALL set that neuron's Membrane_Potential to that neuron's Reset_Potential.
5. WHILE a LIF_Neuron is within its Refractory_Period after a Spike, THE Simulation_Engine SHALL hold that neuron's Membrane_Potential at its Reset_Potential and prevent that neuron from integrating input or emitting a Spike.

### Requirement 3: Instantaneous Synaptic Current to Output Neuron

**User Story:** As a researcher, I want input spikes to cause instantaneous current jumps in the output neuron, so that synaptic transmission is modeled as impulses rather than decaying currents.

#### Acceptance Criteria

1. WHEN Input_Neuron `i` emits a Spike, THE Simulation_Engine SHALL apply an Instantaneous_Current_Jump scaled by Synaptic_Weight `w_i` to the Output_Neuron at the timestep of that Spike.
2. THE Simulation_Engine SHALL model synaptic input to the Output_Neuron as instantaneous current jumps without any decaying synaptic current term.
3. WHEN two Input_Neurons emit a Spike within the same timestep, THE Simulation_Engine SHALL apply the sum of both weighted Instantaneous_Current_Jumps to the Output_Neuron for that timestep.

### Requirement 4: Live-Adjustable Per-Neuron LIF Parameters

**User Story:** As a researcher, I want to adjust every LIF parameter of each neuron individually while the simulation runs, so that I can observe how parameter changes affect spiking and vector dynamics in real time.

#### Acceptance Criteria

1. THE GUI SHALL expose live controls for tau_m, Resting_Potential, Threshold, Reset_Potential, Membrane_Resistance, and Refractory_Period for each of the three LIF_Neurons individually.
2. WHEN the user changes any LIF parameter control for a LIF_Neuron, THE Simulation_Engine SHALL apply the new parameter value to that specific LIF_Neuron on the next computed timestep.
3. WHILE the simulation is running, THE System SHALL accept LIF parameter changes without requiring the simulation to stop or restart.
4. WHEN the user changes a LIF_Neuron's parameter, THE System SHALL leave the other two LIF_Neurons' parameters unchanged.

### Requirement 5: Live-Adjustable Input Currents

**User Story:** As a researcher, I want to inject pulse or constant currents into each input neuron and adjust them live, so that I can drive the network with different stimulation patterns.

#### Acceptance Criteria

1. THE GUI SHALL expose, for each Input_Neuron, a control to select the Input_Current type as either Pulse_Current or Constant_Current.
2. THE GUI SHALL expose, for each Input_Neuron, a control to set the magnitude of that Input_Neuron's Input_Current.
3. WHEN the user selects Constant_Current for an Input_Neuron, THE Simulation_Engine SHALL inject the configured magnitude into that Input_Neuron every timestep.
4. WHEN the user selects Pulse_Current for an Input_Neuron, THE Simulation_Engine SHALL inject the configured magnitude into that Input_Neuron for the configured pulse duration and inject zero current otherwise.
5. WHEN the user changes an Input_Neuron's Input_Current type or magnitude, THE Simulation_Engine SHALL apply the change to that Input_Neuron on the next computed timestep without stopping the simulation.

### Requirement 6: Live-Adjustable Synaptic Weights

**User Story:** As a researcher, I want to adjust the two synaptic weights live, so that I can see how weighting the inputs reshapes the output neuron's behavior and the vector.

#### Acceptance Criteria

1. THE GUI SHALL expose live controls for Synaptic_Weight `w_1` and Synaptic_Weight `w_2`.
2. WHEN the user changes a Synaptic_Weight, THE Simulation_Engine SHALL use the new weight value for subsequent Instantaneous_Current_Jumps and subsequent Vector_N updates starting from the next computed timestep.
3. WHILE the simulation is running, THE System SHALL accept Synaptic_Weight changes without requiring the simulation to stop or restart.

### Requirement 7: Vector N Dynamics

**User Story:** As a researcher, I want the vector N to evolve from input spike trains and decay with the output neuron's time constant, so that the vector represents weighted, leaky accumulation of input activity.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL compute each component `N_i` of Vector_N using the dynamics `tau * dN_i/dt = -N_i + w_i * s_i(t)`, where `s_i(t)` is Input_Neuron `i`'s Spike_Train.
2. THE Simulation_Engine SHALL use the Output_Neuron's current tau_m as the `tau` in the Vector_N dynamics.
3. WHEN the user changes the Output_Neuron's tau_m, THE Simulation_Engine SHALL use the updated tau_m value as the decay time constant for Vector_N on the next computed timestep.
4. WHEN Input_Neuron `i` emits a Spike, THE Simulation_Engine SHALL increase component `N_i` by Synaptic_Weight `w_i` at the timestep of that Spike as an instantaneous impulse.
5. WHILE no Input_Neuron `i` Spike occurs, THE Simulation_Engine SHALL decay component `N_i` toward zero according to the Vector_N dynamics each timestep.
6. WHEN the Output_Neuron emits a Spike, THE Simulation_Engine SHALL set both `N_1` and `N_2` to zero at the timestep of that Spike.

### Requirement 8: Network Topology Graphic

**User Story:** As a researcher, I want a diagram of the network, so that I can see the two inputs, the output, and their weighted connections at a glance.

#### Acceptance Criteria

1. THE GUI SHALL display a graphic depicting the two Input_Neurons, the Output_Neuron, and the weighted connection from each Input_Neuron to the Output_Neuron.
2. THE GUI SHALL display the current value of each Synaptic_Weight on or adjacent to its corresponding connection in the network graphic.

### Requirement 9: Membrane Potential Graphic

**User Story:** As a researcher, I want to see membrane potentials over time for all three neurons, so that I can correlate spiking with vector behavior.

#### Acceptance Criteria

1. THE GUI SHALL display a time-series graphic of the Membrane_Potential of each of the three LIF_Neurons.
2. WHILE the simulation is running, THE GUI SHALL update the Membrane_Potential graphic as new timesteps are computed.

### Requirement 10: Input Current Graphic

**User Story:** As a researcher, I want to see the injected input currents over time, so that I can relate stimulation to resulting spikes and vector dynamics.

#### Acceptance Criteria

1. THE GUI SHALL display a time-series graph of the Input_Current injected into each Input_Neuron.
2. WHILE the simulation is running, THE GUI SHALL update the Input_Current graph as new timesteps are computed.

### Requirement 11: Vector N Graphic

**User Story:** As a researcher, I want to see the vector N as an arrow with a history trail in the N_1 vs N_2 plane, so that I can observe how the vector moves and resets over time.

#### Acceptance Criteria

1. THE GUI SHALL display Vector_N as a 2D arrow in the N_Plane, with the arrow drawn from the origin to the point `(N_1, N_2)`.
2. THE GUI SHALL display a History_Trail of recent Vector_N tip positions in the N_Plane.
3. WHILE the simulation is running, THE GUI SHALL update the Vector_N arrow and the History_Trail as new timesteps are computed.

### Requirement 12: Simulation Timestep Control

**User Story:** As a researcher, I want a fixed, adjustable timestep, so that I can control the numerical resolution of the simulation.

#### Acceptance Criteria

1. THE Simulation_Engine SHALL advance the network state using a fixed timestep dt.
2. THE System SHALL use a default dt of 0.1 milliseconds.
3. THE GUI SHALL expose a control to set the value of dt.
4. WHEN the user changes dt, THE Simulation_Engine SHALL use the new dt value for subsequent timestep computations.

### Requirement 13: Pause and Resume Control

**User Story:** As a researcher, I want to pause and resume the simulation, so that I can stop at any point to inspect the state.

#### Acceptance Criteria

1. THE GUI SHALL provide a control to pause and resume the simulation.
2. WHEN the user activates the pause control while the simulation is running, THE Simulation_Engine SHALL stop advancing the network state while retaining the current state.
3. WHEN the user activates the resume control while the simulation is paused, THE Simulation_Engine SHALL continue advancing the network state from the retained state.
4. WHILE the simulation is paused, THE GUI SHALL continue to display the retained network state.

### Requirement 14: Simulation Speed Control

**User Story:** As a researcher, I want to slow the simulation using a slider and a numeric field, so that I can observe spike-driven patterns in slow motion as they happen.

#### Acceptance Criteria

1. THE GUI SHALL provide a slider control to set the Simulation_Speed.
2. THE GUI SHALL provide a numeric input field to set an exact Simulation_Speed value.
3. WHEN the user sets the Simulation_Speed through the slider, THE GUI SHALL update the numeric input field to the same value.
4. WHEN the user sets the Simulation_Speed through the numeric input field, THE GUI SHALL update the slider to the same value.
5. WHEN the user sets a Simulation_Speed slower than real time, THE Simulation_Engine SHALL advance simulated time slower than wall-clock time by the configured ratio.
6. WHEN the user changes the Simulation_Speed, THE Simulation_Engine SHALL apply the new speed without stopping or restarting the simulation.

### Requirement 15: Pure Python Implementation

**User Story:** As a researcher, I want the prototype implemented in pure Python, so that I can run and modify it in a single-language environment.

#### Acceptance Criteria

1. THE System SHALL be implemented in Python.
2. THE System SHALL use a Python GUI framework that supports live interactive controls and real-time plotting.
