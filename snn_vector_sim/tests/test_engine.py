"""Unit tests for the engine state containers.

Covers :class:`VectorN` and :class:`HistoryBuffers` from
:mod:`snn_vector_sim.core.engine`. The ``SimulationEngine`` itself is
implemented and tested in a later task; these tests target only the state
objects introduced in this task.

Validates: Requirements 7.6, 9.1, 10.1, 11.2
"""

import numpy as np
import pytest

from snn_vector_sim.core.engine import (
    DEFAULT_MAXLEN,
    DEFAULT_TRAIL_LEN,
    HistoryBuffers,
    VectorN,
)


# ---------------------------------------------------------------------------
# VectorN
# ---------------------------------------------------------------------------

def test_vectorn_defaults_to_origin():
    v = VectorN()
    assert v.n1 == pytest.approx(0.0)
    assert v.n2 == pytest.approx(0.0)


def test_vectorn_accepts_initial_components():
    v = VectorN(n1=1.5, n2=-2.0)
    assert v.n1 == pytest.approx(1.5)
    assert v.n2 == pytest.approx(-2.0)


def test_vectorn_reset_sets_both_to_zero():
    """reset() snaps the vector back to the origin (Req 7.6)."""
    v = VectorN(n1=3.0, n2=4.0)
    v.reset()
    assert v.n1 == pytest.approx(0.0)
    assert v.n2 == pytest.approx(0.0)


def test_vectorn_components_are_mutable():
    v = VectorN()
    v.n1 += 1.0
    v.n2 += 2.5
    assert v.n1 == pytest.approx(1.0)
    assert v.n2 == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# HistoryBuffers: construction
# ---------------------------------------------------------------------------

def test_history_defaults():
    h = HistoryBuffers()
    assert h.maxlen == DEFAULT_MAXLEN
    assert h.trail_len == DEFAULT_TRAIL_LEN
    assert len(h) == 0


@pytest.mark.parametrize("bad_maxlen", [0, -1])
def test_history_rejects_nonpositive_maxlen(bad_maxlen):
    with pytest.raises(ValueError):
        HistoryBuffers(maxlen=bad_maxlen)


@pytest.mark.parametrize("bad_trail", [0, -5])
def test_history_rejects_nonpositive_trail_len(bad_trail):
    with pytest.raises(ValueError):
        HistoryBuffers(trail_len=bad_trail)


# ---------------------------------------------------------------------------
# HistoryBuffers: push and retrieval
# ---------------------------------------------------------------------------

def _push_sample(h, i):
    """Push a deterministic sample indexed by i."""
    h.push(
        t=i * 0.1,
        v_in1=-65.0 + i,
        v_in2=-64.0 + i,
        v_out=-63.0 + i,
        I1=1.0 * i,
        I2=2.0 * i,
        n1=0.5 * i,
        n2=-0.5 * i,
    )


def test_push_grows_all_buffers_together():
    h = HistoryBuffers(maxlen=100, trail_len=100)
    for i in range(5):
        _push_sample(h, i)
    assert len(h) == 5

    t, v1, v2, vo = h.potentials()
    assert t.shape == (5,)
    assert v1.shape == (5,)
    assert v2.shape == (5,)
    assert vo.shape == (5,)

    tc, i1, i2 = h.currents()
    assert tc.shape == (5,)
    assert i1.shape == (5,)
    assert i2.shape == (5,)


def test_potentials_values_match_pushed():
    """potentials() returns the pushed membrane potentials in order (Req 9.1)."""
    h = HistoryBuffers(maxlen=100)
    for i in range(4):
        _push_sample(h, i)
    t, v1, v2, vo = h.potentials()
    assert np.allclose(t, [0.0, 0.1, 0.2, 0.3])
    assert np.allclose(v1, [-65.0, -64.0, -63.0, -62.0])
    assert np.allclose(v2, [-64.0, -63.0, -62.0, -61.0])
    assert np.allclose(vo, [-63.0, -62.0, -61.0, -60.0])


def test_currents_values_match_pushed():
    """currents() returns the pushed input currents in order (Req 10.1)."""
    h = HistoryBuffers(maxlen=100)
    for i in range(4):
        _push_sample(h, i)
    t, i1, i2 = h.currents()
    assert np.allclose(t, [0.0, 0.1, 0.2, 0.3])
    assert np.allclose(i1, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(i2, [0.0, 2.0, 4.0, 6.0])


def test_trail_points_values_match_pushed():
    """trail_points() returns the (n1, n2) tips oldest-to-newest (Req 11.2)."""
    h = HistoryBuffers(maxlen=100, trail_len=100)
    for i in range(4):
        _push_sample(h, i)
    n1, n2 = h.trail_points()
    assert np.allclose(n1, [0.0, 0.5, 1.0, 1.5])
    assert np.allclose(n2, [0.0, -0.5, -1.0, -1.5])


def test_trail_points_empty_when_no_samples():
    h = HistoryBuffers()
    n1, n2 = h.trail_points()
    assert n1.shape == (0,)
    assert n2.shape == (0,)


# ---------------------------------------------------------------------------
# HistoryBuffers: bounded behavior
# ---------------------------------------------------------------------------

def test_timeseries_bounded_by_maxlen():
    """Time-series buffers drop oldest samples once full."""
    maxlen = 10
    h = HistoryBuffers(maxlen=maxlen, trail_len=1000)
    for i in range(maxlen + 25):
        _push_sample(h, i)
    assert len(h) == maxlen

    t, v1, v2, vo = h.potentials()
    assert t.shape == (maxlen,)
    # Oldest retained sample is index (total - maxlen).
    first_index = (maxlen + 25) - maxlen
    assert t[0] == pytest.approx(first_index * 0.1)
    assert t[-1] == pytest.approx((maxlen + 25 - 1) * 0.1)


def test_trail_bounded_independently_of_maxlen():
    """The trail has its own shorter bound than the time-series buffers."""
    h = HistoryBuffers(maxlen=1000, trail_len=5)
    for i in range(20):
        _push_sample(h, i)
    # Time-series kept all 20; trail kept only the last 5.
    assert len(h) == 20
    n1, n2 = h.trail_points()
    assert n1.shape == (5,)
    # Last 5 tips correspond to indices 15..19.
    assert np.allclose(n1, [0.5 * i for i in range(15, 20)])
    assert np.allclose(n2, [-0.5 * i for i in range(15, 20)])


# ---------------------------------------------------------------------------
# HistoryBuffers: clear
# ---------------------------------------------------------------------------

def test_clear_empties_all_buffers():
    h = HistoryBuffers(maxlen=100, trail_len=100)
    for i in range(10):
        _push_sample(h, i)
    assert len(h) == 10
    h.clear()
    assert len(h) == 0
    t, v1, v2, vo = h.potentials()
    assert t.shape == (0,)
    n1, n2 = h.trail_points()
    assert n1.shape == (0,)


# ===========================================================================
# SimulationEngine (task 4.3)
# ===========================================================================
#
# These tests target the ordered step() of SimulationEngine: the Vector_N
# impulse/decay/reset dynamics (Req 7) and the instantaneous weighted current
# jump to the output neuron (Req 3).
#
# Validates: Requirements 3.1, 3.3, 7.2, 7.3, 7.4, 7.5, 7.6

import math

from snn_vector_sim.core.engine import SimulationEngine, StepResult
from snn_vector_sim.core.current import ConstantCurrent


# ---------------------------------------------------------------------------
# Helpers for deterministic spike control
# ---------------------------------------------------------------------------

def _make_quiet_engine():
    """An engine whose generators inject zero current, so nothing spikes
    unless we force it. This isolates the N dynamics under test."""
    eng = SimulationEngine()
    eng.gen1 = ConstantCurrent(0.0)
    eng.gen2 = ConstantCurrent(0.0)
    eng.input1.reset_state()
    eng.input2.reset_state()
    eng.output.reset_state()
    eng.N.reset()
    eng.t = 0.0
    eng.history.clear()
    return eng


def _arm_input_to_spike(neuron):
    """Set a neuron's membrane potential just below threshold so that with the
    tiny leak-only dynamics (zero input current) it will NOT spike on its own,
    then nudge it to exactly threshold so the next integrated step spikes.

    We set v slightly above v_th so that even after one forward-Euler leak step
    (which pulls v back toward v_rest) it still crosses threshold this step.
    """
    # Put v comfortably above threshold; the leak over one small dt cannot pull
    # it back below v_th in a single step, guaranteeing a spike this step.
    neuron.v = neuron.v_th + 50.0


def _disarm(neuron):
    """Hold a neuron safely at rest so it does not spike."""
    neuron.v = neuron.v_rest


# ---------------------------------------------------------------------------
# Input spike impulse: forcing input i to spike adds exactly w_i to N_i (Req 7.4)
# ---------------------------------------------------------------------------

def test_input1_spike_adds_w1_to_n1():
    """Forcing input1 to spike adds exactly w1 to N_1 (Req 7.4).

    N starts at the origin, so after decay (of zero) + impulse, N_1 == w1 and
    N_2 stays 0.
    """
    eng = _make_quiet_engine()
    eng.w1 = 2.5
    eng.w2 = 1.0
    _arm_input_to_spike(eng.input1)
    _disarm(eng.input2)
    _disarm(eng.output)

    result = eng.step()

    assert result.s_in1 is True
    assert result.s_in2 is False
    # N_1 == decay(0) + w1 == w1; N_2 untouched.
    assert eng.N.n1 == pytest.approx(2.5)
    assert eng.N.n2 == pytest.approx(0.0)


def test_input2_spike_adds_w2_to_n2():
    """Forcing input2 to spike adds exactly w2 to N_2 (Req 7.4)."""
    eng = _make_quiet_engine()
    eng.w1 = 1.0
    eng.w2 = 3.0
    _disarm(eng.input1)
    _arm_input_to_spike(eng.input2)
    _disarm(eng.output)

    result = eng.step()

    assert result.s_in1 is False
    assert result.s_in2 is True
    assert eng.N.n1 == pytest.approx(0.0)
    assert eng.N.n2 == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Decay: with no spikes, N_i decays by exp(-dt/tau_out) per step (Req 7.5),
# and the rate tracks output.tau_m (Req 7.2, 7.3)
# ---------------------------------------------------------------------------

def test_no_spike_decays_by_exp_dt_over_tau_out():
    """With no spikes, each N_i is scaled by exp(-dt/output.tau_m) per step."""
    eng = _make_quiet_engine()
    eng.N.n1 = 1.0
    eng.N.n2 = -0.5
    _disarm(eng.input1)
    _disarm(eng.input2)
    _disarm(eng.output)

    factor = math.exp(-eng.dt / eng.output.tau_m)
    result = eng.step()

    assert result.s_in1 is False
    assert result.s_in2 is False
    assert result.s_out is False
    assert eng.N.n1 == pytest.approx(1.0 * factor)
    assert eng.N.n2 == pytest.approx(-0.5 * factor)


def test_decay_rate_tracks_output_tau_m():
    """Changing output.tau_m changes the N decay rate (Req 7.3).

    A larger tau_out decays more slowly (factor closer to 1); a smaller tau_out
    decays faster. The engine reads the live tau_m on the next step with no
    extra wiring (Req 7.2).
    """
    # Slow decay (large tau_m).
    slow = _make_quiet_engine()
    slow.output.tau_m = 100.0
    slow.N.n1 = 1.0
    _disarm(slow.input1)
    _disarm(slow.input2)
    _disarm(slow.output)
    slow.step()
    slow_factor = math.exp(-slow.dt / 100.0)
    assert slow.N.n1 == pytest.approx(slow_factor)

    # Fast decay (small tau_m) from the same starting value.
    fast = _make_quiet_engine()
    fast.output.tau_m = 1.0
    fast.N.n1 = 1.0
    _disarm(fast.input1)
    _disarm(fast.input2)
    _disarm(fast.output)
    fast.step()
    fast_factor = math.exp(-fast.dt / 1.0)
    assert fast.N.n1 == pytest.approx(fast_factor)

    # Faster decay leaves less of the original value than slower decay.
    assert fast.N.n1 < slow.N.n1


def test_decay_compounds_over_multiple_steps():
    """Over k steps with no spikes, N_i == N0 * exp(-dt/tau_out)**k (Req 7.5)."""
    eng = _make_quiet_engine()
    eng.N.n1 = 2.0
    _disarm(eng.input1)
    _disarm(eng.input2)
    _disarm(eng.output)

    factor = math.exp(-eng.dt / eng.output.tau_m)
    steps = 20
    for _ in range(steps):
        # Keep the neurons quiet every step (leak-only on zero current keeps
        # them at rest anyway, but be explicit about intent).
        _disarm(eng.input1)
        _disarm(eng.input2)
        _disarm(eng.output)
        eng.step()

    assert eng.N.n1 == pytest.approx(2.0 * (factor ** steps))


# ---------------------------------------------------------------------------
# Within-step order: decay-before-impulse (Req 7.5)
# ---------------------------------------------------------------------------

def test_decay_applied_before_impulse():
    """A spiking step applies decay to the PRE-EXISTING N first, then adds the
    undecayed impulse: N_i == old_N_i * exp(-dt/tau_out) + w_i (Req 7.5)."""
    eng = _make_quiet_engine()
    eng.w1 = 1.0
    old_n1 = 0.8
    eng.N.n1 = old_n1
    _arm_input_to_spike(eng.input1)
    _disarm(eng.input2)
    _disarm(eng.output)

    factor = math.exp(-eng.dt / eng.output.tau_m)
    result = eng.step()

    assert result.s_in1 is True
    # Decay hits the old value; the impulse is added undecayed on top.
    assert eng.N.n1 == pytest.approx(old_n1 * factor + eng.w1)


def test_impulse_is_not_decayed_within_the_spiking_step():
    """The within-step impulse itself is NOT scaled by the decay factor: if it
    were decayed, N_1 would be old*factor + w1*factor, not old*factor + w1."""
    eng = _make_quiet_engine()
    eng.w1 = 1.0
    old_n1 = 1.0
    eng.N.n1 = old_n1
    _arm_input_to_spike(eng.input1)
    _disarm(eng.input2)
    _disarm(eng.output)

    factor = math.exp(-eng.dt / eng.output.tau_m)
    eng.step()

    decayed_impulse_value = old_n1 * factor + eng.w1 * factor
    correct_value = old_n1 * factor + eng.w1
    assert eng.N.n1 == pytest.approx(correct_value)
    assert eng.N.n1 != pytest.approx(decayed_impulse_value)


# ---------------------------------------------------------------------------
# Output reset: output spike sets N to (0,0), overriding same-step impulses (Req 7.6)
# ---------------------------------------------------------------------------

def test_output_spike_resets_n_to_origin():
    """When the output neuron spikes, N is reset to (0, 0) that step (Req 7.6)."""
    eng = _make_quiet_engine()
    eng.N.n1 = 1.5
    eng.N.n2 = -2.0
    _disarm(eng.input1)
    _disarm(eng.input2)
    _arm_input_to_spike(eng.output)

    result = eng.step()

    assert result.s_out is True
    assert eng.N.n1 == pytest.approx(0.0)
    assert eng.N.n2 == pytest.approx(0.0)


def test_output_spike_overrides_same_step_input_impulses():
    """An output spike wins over same-step input impulses: even when both
    inputs spike (adding w1, w2 to N) in the same step the output spikes, N
    ends at the origin (Req 7.6)."""
    eng = _make_quiet_engine()
    eng.w1 = 5.0
    eng.w2 = 7.0
    _arm_input_to_spike(eng.input1)
    _arm_input_to_spike(eng.input2)
    _arm_input_to_spike(eng.output)

    result = eng.step()

    assert result.s_in1 is True
    assert result.s_in2 is True
    assert result.s_out is True
    # Impulses were added then wiped by the reset -> origin.
    assert eng.N.n1 == pytest.approx(0.0)
    assert eng.N.n2 == pytest.approx(0.0)
    assert result.n1 == pytest.approx(0.0)
    assert result.n2 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Simultaneous input spikes sum to w1 + w2 output current (Req 3.1, 3.3)
# ---------------------------------------------------------------------------

def test_simultaneous_input_spikes_sum_output_current():
    """When both inputs spike in the same step, the output neuron receives
    I_out = w1 + w2 as an instantaneous current jump (Req 3.1, 3.3).

    We verify via the output neuron's membrane response. With the output armed
    at rest and refractory clear, its one-step forward-Euler update is
    dv = (-(v - v_rest) + R * I_out) / tau_m * dt. At v == v_rest the leak term
    is zero, so dv == R * I_out / tau_m * dt, which lets us back out I_out.
    """
    eng = _make_quiet_engine()
    eng.w1 = 2.0
    eng.w2 = 3.0
    _arm_input_to_spike(eng.input1)
    _arm_input_to_spike(eng.input2)
    # Output starts exactly at rest so the leak term vanishes this step and it
    # stays sub-threshold (small current, no spike) for a clean readout.
    eng.output.v = eng.output.v_rest
    eng.output.refrac_left = 0.0

    v_rest = eng.output.v_rest
    R = eng.output.R
    tau_m = eng.output.tau_m
    dt = eng.dt

    result = eng.step()

    assert result.s_in1 is True
    assert result.s_in2 is True
    # Expected output current is the summed weighted jump.
    expected_I_out = eng.w1 + eng.w2
    # Recover I_out from the membrane change: dv = R * I_out / tau_m * dt.
    observed_dv = eng.output.v - v_rest
    observed_I_out = observed_dv * tau_m / (R * dt)
    assert observed_I_out == pytest.approx(expected_I_out)


def test_single_input_spike_output_current_is_that_weight():
    """One input spiking delivers only its own weight as output current
    (sanity complement to the summing case, Req 3.1)."""
    eng = _make_quiet_engine()
    eng.w1 = 4.0
    eng.w2 = 9.0
    _arm_input_to_spike(eng.input1)
    _disarm(eng.input2)
    eng.output.v = eng.output.v_rest
    eng.output.refrac_left = 0.0

    v_rest = eng.output.v_rest
    R = eng.output.R
    tau_m = eng.output.tau_m
    dt = eng.dt

    result = eng.step()

    assert result.s_in1 is True
    assert result.s_in2 is False
    observed_dv = eng.output.v - v_rest
    observed_I_out = observed_dv * tau_m / (R * dt)
    assert observed_I_out == pytest.approx(eng.w1)


def test_no_input_spikes_output_current_is_zero():
    """With no input spikes, the output neuron receives zero synaptic current
    and (at rest) does not move (Req 3.1 contrapositive)."""
    eng = _make_quiet_engine()
    _disarm(eng.input1)
    _disarm(eng.input2)
    eng.output.v = eng.output.v_rest
    eng.output.refrac_left = 0.0

    result = eng.step()

    assert result.s_in1 is False
    assert result.s_in2 is False
    # At rest with zero current, dv == 0 so v stays at rest.
    assert eng.output.v == pytest.approx(eng.output.v_rest)
