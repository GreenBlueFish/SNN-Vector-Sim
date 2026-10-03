"""Unit tests for the input current generators.

Covers the design-specified test cases for :mod:`snn_vector_sim.core.current`:
- :class:`ConstantCurrent` returns its magnitude for all ``t`` (Req 5.3).
- :class:`PulseCurrent` returns its magnitude only within ``width`` of each
  ``period`` and ``0`` otherwise, verified across several periods (Req 5.4).

Validates: Requirements 5.3, 5.4
"""

import pytest

from snn_vector_sim.core.current import (
    ConstantCurrent,
    CurrentGenerator,
    PulseCurrent,
)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_generators_satisfy_current_generator_protocol():
    assert isinstance(ConstantCurrent(), CurrentGenerator)
    assert isinstance(PulseCurrent(), CurrentGenerator)


# ---------------------------------------------------------------------------
# ConstantCurrent (Req 5.3)
# ---------------------------------------------------------------------------

def test_constant_returns_magnitude_for_all_t():
    """ConstantCurrent injects its magnitude every step regardless of time."""
    magnitude = 12.5
    gen = ConstantCurrent(magnitude=magnitude)
    dt = 0.1
    # Sample a long span of simulated time, including t=0 and large t.
    for i in range(0, 100000, 7):
        t = i * dt
        assert gen.value(t=t, dt=dt) == pytest.approx(magnitude)


def test_constant_default_magnitude_is_zero():
    gen = ConstantCurrent()
    assert gen.value(t=0.0, dt=0.1) == pytest.approx(0.0)
    assert gen.value(t=1234.5, dt=0.1) == pytest.approx(0.0)


def test_constant_handles_negative_magnitude():
    gen = ConstantCurrent(magnitude=-5.0)
    assert gen.value(t=0.0, dt=0.1) == pytest.approx(-5.0)
    assert gen.value(t=999.0, dt=0.1) == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# PulseCurrent (Req 5.4)
# ---------------------------------------------------------------------------

def test_pulse_on_within_width_at_start_of_first_period():
    """At t=0 (period onset) the pulse is on."""
    gen = PulseCurrent(magnitude=20.0, period=50.0, width=5.0)
    assert gen.value(t=0.0, dt=0.1) == pytest.approx(20.0)


def test_pulse_off_after_width_within_period():
    """Just past the width, within the same period, the pulse is off."""
    gen = PulseCurrent(magnitude=20.0, period=50.0, width=5.0)
    # width=5 -> on for phase in [0,5); off for phase in [5,50).
    assert gen.value(t=5.0, dt=0.1) == pytest.approx(0.0)
    assert gen.value(t=10.0, dt=0.1) == pytest.approx(0.0)
    assert gen.value(t=49.9, dt=0.1) == pytest.approx(0.0)


def test_pulse_on_off_behavior_across_several_periods():
    """Verify the on/off pattern repeats across multiple periods.

    For each of several periods k, the pulse must be on for simulated times
    within ``[k*period, k*period + width)`` and off for the remainder of the
    period ``[k*period + width, (k+1)*period)``.
    """
    magnitude = 20.0
    period = 50.0
    width = 5.0
    gen = PulseCurrent(magnitude=magnitude, period=period, width=width)

    n_periods = 6
    for k in range(n_periods):
        base = k * period

        # Points that must be ON (phase strictly less than width).
        for phase in (0.0, 0.1, 2.5, width - 0.1):
            t = base + phase
            assert gen.value(t=t, dt=0.1) == pytest.approx(magnitude), (
                f"expected ON at period {k}, phase {phase}, t={t}"
            )

        # Points that must be OFF (phase in [width, period)).
        for phase in (width, width + 0.1, period / 2.0, period - 0.1):
            t = base + phase
            assert gen.value(t=t, dt=0.1) == pytest.approx(0.0), (
                f"expected OFF at period {k}, phase {phase}, t={t}"
            )


def test_pulse_boundary_at_width_is_off():
    """The width boundary is exclusive: phase == width is off."""
    gen = PulseCurrent(magnitude=7.0, period=20.0, width=4.0)
    # Across several periods, exactly at the width boundary -> off.
    for k in range(4):
        t = k * 20.0 + 4.0
        assert gen.value(t=t, dt=0.1) == pytest.approx(0.0)


def test_pulse_onset_boundary_at_period_is_on():
    """Each new period onset (phase == 0) turns the pulse on again."""
    gen = PulseCurrent(magnitude=7.0, period=20.0, width=4.0)
    for k in range(4):
        t = k * 20.0
        assert gen.value(t=t, dt=0.1) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# OffCurrent and ManualPulseCurrent (added features)
# ---------------------------------------------------------------------------

from snn_vector_sim.core.current import OffCurrent, ManualPulseCurrent


def test_off_current_is_always_zero():
    gen = OffCurrent()
    for i in range(0, 1000, 13):
        assert gen.value(t=i * 0.1, dt=0.1) == pytest.approx(0.0)
    assert isinstance(gen, CurrentGenerator)


def test_manual_pulse_is_zero_until_fired():
    gen = ManualPulseCurrent(magnitude=20.0, width=5.0)
    # Never fired -> no current no matter how long we wait.
    for i in range(100):
        assert gen.value(t=i * 0.1, dt=0.1) == pytest.approx(0.0)


def test_manual_pulse_fires_for_width_then_stops():
    gen = ManualPulseCurrent(magnitude=20.0, width=5.0)
    gen.fire()
    dt = 0.1
    on_steps = 0
    # width=5 ms at dt=0.1 -> about 50 steps on.
    for _ in range(200):
        v = gen.value(t=0.0, dt=dt)
        if v > 0:
            assert v == pytest.approx(20.0)
            on_steps += 1
        else:
            break
    assert on_steps == pytest.approx(50, abs=1)
    # After the pulse, it stays off until fired again.
    assert gen.value(t=0.0, dt=dt) == pytest.approx(0.0)


def test_manual_pulse_refire_restarts_the_pulse():
    gen = ManualPulseCurrent(magnitude=10.0, width=2.0)
    gen.fire()
    gen.value(t=0.0, dt=1.0)   # consumes 1 ms, 1 ms left
    gen.fire()                 # re-arm back to full width
    # Should now deliver another full ~2 ms (2 steps at dt=1.0).
    assert gen.value(t=0.0, dt=1.0) == pytest.approx(10.0)
    assert gen.value(t=0.0, dt=1.0) == pytest.approx(10.0)
    assert gen.value(t=0.0, dt=1.0) == pytest.approx(0.0)


def test_manual_pulse_satisfies_protocol():
    assert isinstance(ManualPulseCurrent(), CurrentGenerator)
