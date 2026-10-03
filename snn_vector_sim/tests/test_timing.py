"""Unit tests for the wall-clock-to-simulation timing model.

Covers :class:`TickScheduler` from :mod:`snn_vector_sim.core.timing`, which
maps fixed wall-clock timer ticks onto engine integration steps using a
fractional accumulator (the slow-motion speed control).

The central property under test is that the long-run **average simulated-time
advance per tick** equals the target ``M * T_tick`` regardless of ``dt`` and
even for arbitrarily small ``M`` (sub-one-step-per-tick), with the accumulator
carrying fractional remainders across ticks. Live changes to ``M`` or ``dt``
take effect on the next tick without resetting the accumulator (Req 12.4,
14.6).

Validates: Requirements 12.1, 12.4, 14.5, 14.6
"""

import math

import pytest

from snn_vector_sim.core.timing import DEFAULT_T_TICK, TickScheduler


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_default_t_tick():
    sched = TickScheduler()
    assert sched.t_tick == pytest.approx(DEFAULT_T_TICK)


@pytest.mark.parametrize("bad_t_tick", [0.0, -1.0])
def test_rejects_nonpositive_t_tick(bad_t_tick):
    with pytest.raises(ValueError):
        TickScheduler(t_tick=bad_t_tick)


def test_steps_for_tick_rejects_nonpositive_dt():
    sched = TickScheduler()
    with pytest.raises(ValueError):
        sched.steps_for_tick(M=1.0, dt=0.0)


def test_steps_for_tick_rejects_negative_M():
    sched = TickScheduler()
    with pytest.raises(ValueError):
        sched.steps_for_tick(M=-0.1, dt=0.1)


# ---------------------------------------------------------------------------
# Average simulated-time advance for representative (M, dt, T_tick) combos
# ---------------------------------------------------------------------------
#
# Over many ticks, total_steps * dt should approximate
# total_ticks * M * T_tick (the target simulated-time advance), independent of
# dt. The accumulator makes the error bounded by less than one step's worth of
# dt total, so the per-tick average converges as the tick count grows.

@pytest.mark.parametrize(
    "M, dt, t_tick",
    [
        (1.0, 0.1, 20.0),    # near real time: desired 200 steps/tick
        (0.1, 0.1, 20.0),    # 10x slow motion: desired 20 steps/tick
        (0.5, 0.25, 20.0),   # desired 40 steps/tick
        (1.0, 1.0, 20.0),    # desired 20 steps/tick
        (0.37, 0.1, 16.0),   # non-integer desired steps/tick (59.2)
        (0.01, 0.5, 20.0),   # very small M: desired 0.4 steps/tick (<1)
    ],
)
def test_average_sim_time_advance_matches_target(M, dt, t_tick):
    sched = TickScheduler(t_tick=t_tick)
    ticks = 10000

    total_steps = 0
    for _ in range(ticks):
        total_steps += sched.steps_for_tick(M, dt)

    advanced_sim_time = total_steps * dt
    target_sim_time = ticks * M * t_tick

    # The accumulator guarantees the discrepancy is bounded by one undispatched
    # fractional step (< dt of simulated time) across the whole run, so the
    # per-tick average error vanishes as ticks grows.
    assert abs(advanced_sim_time - target_sim_time) < dt
    # Per-tick average advance is within a hair of the target.
    avg_per_tick = advanced_sim_time / ticks
    assert avg_per_tick == pytest.approx(M * t_tick, abs=dt / ticks + 1e-9)


# ---------------------------------------------------------------------------
# Very small M: sub-one-step-per-tick slow motion (Req 14.5)
# ---------------------------------------------------------------------------

def test_very_small_M_runs_fewer_than_one_step_per_tick_on_average():
    """M=0.001, dt=0.1, T_tick=20 -> desired 0.2 steps/tick, i.e. ~1 step
    every 5 ticks. Some ticks must run 0 steps, yet the long-run average
    advance stays correct (arbitrarily slow motion, Req 14.5)."""
    M, dt, t_tick = 0.001, 0.1, 20.0
    desired_per_tick = M * t_tick / dt  # 0.2
    assert desired_per_tick < 1.0

    sched = TickScheduler(t_tick=t_tick)
    ticks = 10000
    per_tick_counts = [sched.steps_for_tick(M, dt) for _ in range(ticks)]

    total_steps = sum(per_tick_counts)

    # Some ticks genuinely dispatch zero steps (the accumulator defers).
    assert any(c == 0 for c in per_tick_counts)
    # No tick runs more than one step at this tiny rate.
    assert max(per_tick_counts) <= 1

    # Long-run average matches the sub-one target within one carried fraction.
    expected_total = ticks * desired_per_tick
    assert abs(total_steps - expected_total) < 1.0
    avg = total_steps / ticks
    assert avg == pytest.approx(desired_per_tick, abs=1e-3)


def test_sub_one_step_dispatches_whole_steps_only():
    """Even at a fractional desired rate, each tick returns an integer count
    (never a partial step)."""
    sched = TickScheduler(t_tick=20.0)
    for _ in range(50):
        n = sched.steps_for_tick(M=0.002, dt=0.1)  # desired 0.4 steps/tick (<1)
        assert isinstance(n, int)
        assert n >= 0


# ---------------------------------------------------------------------------
# Accumulator carries fractional remainder correctly across ticks
# ---------------------------------------------------------------------------

def test_accumulator_carries_fraction_between_ticks():
    """desired = 0.5 steps/tick -> the pattern settles to one step every two
    ticks (0, 1, 0, 1, ...), proving the remainder carries forward."""
    # M * T_tick / dt = 0.5  with T_tick=20, dt=40, M=1.0
    sched = TickScheduler(t_tick=20.0)
    counts = [sched.steps_for_tick(M=1.0, dt=40.0) for _ in range(8)]
    assert counts == [0, 1, 0, 1, 0, 1, 0, 1]


def test_near_real_time_dispatches_expected_whole_steps():
    """desired exactly 200 steps/tick dispatches 200 every tick with no drift."""
    sched = TickScheduler(t_tick=20.0)
    for _ in range(100):
        assert sched.steps_for_tick(M=1.0, dt=0.1) == 200


def test_fractional_desired_averages_to_target_over_known_window():
    """desired = 1.5 steps/tick -> over 2 ticks dispatches 3 total (1 then 2,
    repeating), averaging exactly 1.5."""
    # M * T_tick / dt = 1.5 with T_tick=15, dt=10, M=1.0
    sched = TickScheduler(t_tick=15.0)
    counts = [sched.steps_for_tick(M=1.0, dt=10.0) for _ in range(6)]
    assert sum(counts) == 9           # 6 ticks * 1.5
    assert counts == [1, 2, 1, 2, 1, 2]


# ---------------------------------------------------------------------------
# Changing M or dt mid-run adjusts the rate without resetting incorrectly
# (Req 12.4, 14.6)
# ---------------------------------------------------------------------------

def test_changing_M_mid_run_adjusts_rate_live():
    """Switching M between ticks changes the dispatched rate immediately while
    preserving the carried fraction (no reset, Req 14.6)."""
    sched = TickScheduler(t_tick=20.0)
    dt = 0.1

    # Phase A: M=1.0 -> 200 steps/tick.
    a = [sched.steps_for_tick(M=1.0, dt=dt) for _ in range(50)]
    assert all(n == 200 for n in a)

    # Phase B: drop to M=0.1 -> 20 steps/tick, effective immediately next tick.
    b = [sched.steps_for_tick(M=0.1, dt=dt) for _ in range(50)]
    assert all(n == 20 for n in b)


def test_changing_dt_mid_run_adjusts_rate_live():
    """Switching dt between ticks changes steps/tick for the same M, keeping
    simulated-time advance per tick (M*T_tick) consistent (Req 12.4)."""
    M, t_tick = 1.0, 20.0
    sched = TickScheduler(t_tick=t_tick)

    # dt=0.1 -> 200 steps/tick; each tick advances M*T_tick = 20 ms sim time.
    a = [sched.steps_for_tick(M, dt=0.1) for _ in range(30)]
    assert all(n == 200 for n in a)
    assert all(n * 0.1 == pytest.approx(M * t_tick) for n in a)

    # dt=0.2 -> 100 steps/tick; still advances the same 20 ms sim time per tick.
    b = [sched.steps_for_tick(M, dt=0.2) for _ in range(30)]
    assert all(n == 100 for n in b)
    assert all(n * 0.2 == pytest.approx(M * t_tick) for n in b)


def test_changing_params_preserves_carried_fraction():
    """A fraction accrued before a parameter change is not discarded; it is
    carried into the computation under the new parameters (no implicit reset)."""
    sched = TickScheduler(t_tick=20.0)

    # One tick at desired 0.5 steps/tick: dispatches 0, carries 0.5.
    first = sched.steps_for_tick(M=1.0, dt=40.0)  # 20/40 = 0.5
    assert first == 0

    # Next tick switch to desired 0.6 steps/tick: accumulator 0.5 + 0.6 = 1.1
    # -> dispatches 1, carries 0.1. If the fraction had been reset, we'd only
    # have 0.6 and dispatch 0.
    second = sched.steps_for_tick(M=0.6, dt=20.0)  # 0.6*20/20 = 0.6
    assert second == 1


# ---------------------------------------------------------------------------
# reset() clears the accumulator
# ---------------------------------------------------------------------------

def test_reset_clears_accumulator():
    sched = TickScheduler(t_tick=20.0)
    # Accrue a fraction without dispatching.
    sched.steps_for_tick(M=1.0, dt=40.0)  # desired 0.5, dispatch 0, carry 0.5
    sched.reset()
    # After reset, a fresh desired-0.5 tick again dispatches 0 (no carried 0.5
    # that would have combined to 1.0 and dispatched a step).
    assert sched.steps_for_tick(M=1.0, dt=40.0) == 0
