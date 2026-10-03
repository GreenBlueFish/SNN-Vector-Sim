"""Leaky Integrate-and-Fire (LIF) neuron model.

Implements a single LIF neuron that holds its own parameters and runtime
state and advances one timestep at a time via forward-Euler integration.

Units convention (mV-ms-nA-MOhm):
- Time / tau_m / t_refrac: milliseconds (ms)
- Membrane potential (v, v_rest, v_th, v_reset): millivolts (mV)
- Membrane resistance R: megaohms (MOhm)
- Input current i_input: nanoamperes (nA), so R * i_input is in mV
"""

from dataclasses import dataclass, field


@dataclass
class LIFNeuron:
    """A single Leaky Integrate-and-Fire neuron.

    Parameters are plain attributes so GUI controls can mutate them directly;
    the next call to :meth:`step` reads the updated values (live update).
    """

    # Parameters
    tau_m: float = 10.0        # ms   membrane time constant (>0)
    R: float = 10.0            # MOhm membrane resistance (>0)
    v_rest: float = -65.0      # mV   resting potential
    v_th: float = -50.0        # mV   spike threshold
    v_reset: float = -70.0     # mV   reset potential after a spike
    t_refrac: float = 2.0      # ms   refractory duration (>=0)

    # Runtime state
    v: float = field(init=False)       # mV current membrane potential
    refrac_left: float = 0.0           # ms remaining in refractory period
    spiked: bool = False               # did it spike on the last step

    def __post_init__(self) -> None:
        # Initialize membrane potential to the resting potential.
        self.v = self.v_rest

    def reset_state(self) -> None:
        """Reset runtime state: v -> v_rest, clear refractory and spike flag."""
        self.v = self.v_rest
        self.refrac_left = 0.0
        self.spiked = False

    def step(self, dt: float, i_input: float) -> bool:
        """Advance the neuron by ``dt`` (ms) with input current ``i_input`` (nA).

        Returns ``True`` if the neuron spiked on this step, else ``False``.

        Ordering (per design):
        1. If in refractory period, hold v at v_reset, decrement the refractory
           timer, do not integrate or spike.
        2. Otherwise integrate forward-Euler.
        3. If v >= v_th, emit a spike: set v to v_reset, start refractory.
        4. Else no spike.
        """
        # 1. Refractory hold: no integration, no spike.
        if self.refrac_left > 0:
            self.v = self.v_reset
            self.refrac_left -= dt
            self.spiked = False
            return False

        # 2. Forward-Euler integration.
        dv = (-(self.v - self.v_rest) + self.R * i_input) / self.tau_m * dt
        self.v += dv

        # 3. Threshold detection -> spike, reset, enter refractory.
        if self.v >= self.v_th:
            self.v = self.v_reset
            self.refrac_left = self.t_refrac
            self.spiked = True
            return True

        # 4. No spike.
        self.spiked = False
        return False
