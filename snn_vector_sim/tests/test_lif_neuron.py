"""Unit tests for :class:`snn_vector_sim.core.lif_neuron.LIFNeuron`.

Covers the design-specified test cases for LIF dynamics:
- Sub-threshold decay toward ``v_rest`` with the expected ``tau_m``.
- Steady state at ``v_rest + R * I`` under constant sub-threshold current.
- Spiking on supra-threshold current with reset to ``v_reset``.
- Refractory hold then resumption of integration after ``t_refrac``.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
"""

import math

import pytest

from snn_vector_sim.core.lif_neuron import LIFNeuron


# ---------------------------------------------------------------------------
# Construction / reset
# ---------------------------------------------------------------------------

def test_initializes_v_to_v_rest():
    neuron = LIFNeuron(v_rest=-65.0)
    assert neuron.v == pytest.approx(-65.0)
    assert neuron.refrac_left == 0.0
    assert neuron.spiked is False


def test_reset_state_restores_v_rest_and_clears_flags():
    neuron = LIFNeuron(v_rest=-65.0, t_refrac=2.0)
    # Drive it into a spike so state is dirtied.
    neuron.step(dt=1.0, i_input=1000.0)
    assert neuron.refrac_left > 0  # entered refractory

    neuron.reset_state()
    assert neuron.v == pytest.approx(-65.0)
    assert neuron.refrac_left == 0.0
    assert neuron.spiked is False


# ---------------------------------------------------------------------------
# Sub-threshold decay (Req 2.1, 2.2)
# ---------------------------------------------------------------------------

def test_subthreshold_decay_toward_v_rest_with_expected_tau():
    """With zero input, v decays from an offset toward v_rest.

    The closed-form solution of dv/dt = -(v - v_rest)/tau_m with zero input is
    v(t) = v_rest + (v0 - v_rest) * exp(-t / tau_m). Forward-Euler with a small
    dt should track this within tolerance.
    """
    tau_m = 10.0
    v_rest = -65.0
    v0 = -55.0  # offset above rest, still below threshold
    dt = 0.01

    neuron = LIFNeuron(
        tau_m=tau_m, R=10.0, v_rest=v_rest, v_th=-50.0, v_reset=-70.0, t_refrac=0.0
    )
    neuron.v = v0

    # Advance for one time constant worth of simulated time.
    n_steps = int(round(tau_m / dt))
    for _ in range(n_steps):
        spiked = neuron.step(dt=dt, i_input=0.0)
        assert spiked is False  # must never spike sub-threshold

    expected = v_rest + (v0 - v_rest) * math.exp(-1.0)
    assert neuron.v == pytest.approx(expected, abs=0.05)
    # And it should be strictly between the start and rest (monotonic decay).
    assert v_rest < neuron.v < v0


def test_subthreshold_decay_is_monotonic_toward_rest():
    neuron = LIFNeuron(tau_m=10.0, v_rest=-65.0, v_th=-50.0, t_refrac=0.0)
    neuron.v = -55.0
    prev = neuron.v
    for _ in range(100):
        neuron.step(dt=0.1, i_input=0.0)
        assert neuron.v <= prev + 1e-9  # non-increasing while above rest
        assert neuron.v >= -65.0        # never overshoots below rest
        prev = neuron.v


# ---------------------------------------------------------------------------
# Steady state (Req 2.1, 2.2)
# ---------------------------------------------------------------------------

def test_steady_state_converges_to_v_rest_plus_R_times_I():
    """Constant sub-threshold current converges to v_rest + R * I."""
    R = 10.0
    v_rest = -65.0
    i_input = 1.0  # R*I = 10 mV -> steady state -55 mV, below threshold -50
    dt = 0.01

    neuron = LIFNeuron(
        tau_m=10.0, R=R, v_rest=v_rest, v_th=-50.0, v_reset=-70.0, t_refrac=0.0
    )

    for _ in range(20000):  # many tau_m worth of time
        spiked = neuron.step(dt=dt, i_input=i_input)
        assert spiked is False

    expected = v_rest + R * i_input
    assert neuron.v == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Spiking + reset (Req 2.3, 2.4)
# ---------------------------------------------------------------------------

def test_supra_threshold_current_produces_spike_and_resets():
    """A constant supra-threshold current produces a spike; on spike v is set
    to v_reset and step returns True."""
    v_reset = -70.0
    neuron = LIFNeuron(
        tau_m=10.0, R=10.0, v_rest=-65.0, v_th=-50.0, v_reset=v_reset, t_refrac=2.0
    )

    # Strong current drives v across threshold within a few steps.
    spiked = False
    for _ in range(1000):
        spiked = neuron.step(dt=0.1, i_input=100.0)
        if spiked:
            break

    assert spiked is True
    assert neuron.spiked is True
    assert neuron.v == pytest.approx(v_reset)
    assert neuron.refrac_left == pytest.approx(2.0)


def test_spike_fires_when_potential_reaches_threshold():
    """Spike condition is v >= v_th (reaches or exceeds)."""
    neuron = LIFNeuron(
        tau_m=10.0, R=10.0, v_rest=-65.0, v_th=-50.0, v_reset=-70.0, t_refrac=1.0
    )
    # Place v just below threshold; a single step with current pushes it over.
    neuron.v = -50.5
    spiked = neuron.step(dt=1.0, i_input=10.0)
    assert spiked is True
    assert neuron.v == pytest.approx(-70.0)


# ---------------------------------------------------------------------------
# Refractory hold then resumption (Req 2.5)
# ---------------------------------------------------------------------------

def test_refractory_holds_at_reset_and_blocks_integration_and_spikes():
    """While refrac_left > 0, v is held at v_reset, no integration, no spike."""
    v_reset = -70.0
    t_refrac = 2.0
    dt = 0.1
    neuron = LIFNeuron(
        tau_m=10.0, R=10.0, v_rest=-65.0, v_th=-50.0, v_reset=v_reset,
        t_refrac=t_refrac,
    )

    # Place v just below threshold, then force a spike to enter refractory.
    neuron.v = -50.5
    assert neuron.step(dt=dt, i_input=100.0) is True
    assert neuron.refrac_left == pytest.approx(t_refrac)

    # During refractory, even a huge current must not integrate or spike,
    # and v must stay pinned at v_reset.
    held_steps = int(round(t_refrac / dt)) - 1
    for _ in range(held_steps):
        spiked = neuron.step(dt=dt, i_input=100.0)
        assert spiked is False
        assert neuron.v == pytest.approx(v_reset)
        assert neuron.refrac_left > 0


def test_integration_resumes_after_refractory_period():
    """After t_refrac elapses, the neuron integrates input again."""
    v_reset = -70.0
    t_refrac = 2.0
    dt = 0.1
    neuron = LIFNeuron(
        tau_m=10.0, R=10.0, v_rest=-65.0, v_th=-50.0, v_reset=v_reset,
        t_refrac=t_refrac,
    )

    # Place v just below threshold, then spike -> refractory.
    neuron.v = -50.5
    assert neuron.step(dt=dt, i_input=100.0) is True

    # Step through the whole refractory window.
    n_refrac_steps = int(round(t_refrac / dt))
    for _ in range(n_refrac_steps):
        neuron.step(dt=dt, i_input=0.0)

    assert neuron.refrac_left <= 0

    # Now integration should resume: with a positive sub-threshold current,
    # v moves away from v_reset.
    v_before = neuron.v
    neuron.step(dt=dt, i_input=1.0)
    assert neuron.v != pytest.approx(v_before)
    assert neuron.v > v_reset  # driven upward by positive current
