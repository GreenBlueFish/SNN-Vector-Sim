"""Wall-clock-to-simulation timing model (the slow-motion speed mapping).

This Qt-free module maps the GUI's fixed wall-clock timer ticks onto a number
of engine integration steps per tick, implementing the slow-motion speed
control (Req 14) independently of the integration step ``dt`` (Req 12).

Two independent knobs drive the mapping:

- ``dt``: the engine integration step (ms) -- numerical fidelity (Req 12).
- ``M``: the slow-motion speed multiplier in ``[0.01, 1.0]`` -- how much
  *simulated* time elapses per unit *wall-clock* time (Req 14).

The timer fires at a fixed wall-clock interval ``T_tick`` (default 20 ms).
The target simulated time to advance per tick is ``M * T_tick``, so the
desired (generally fractional) number of integration steps per tick is::

    desired = M * T_tick / dt

A naive ``max(1, round(desired))`` floors at one step per tick, which both
distorts the average speed for slow motion and makes arbitrarily slow motion
(``desired < 1``) impossible. Instead this module keeps a **fractional
accumulator**: each tick it adds ``desired`` to the accumulator, dispatches
``floor(accumulator)`` whole steps, and carries the remainder to the next
tick. Over many ticks the dispatched steps average out to exactly ``desired``
per tick regardless of ``dt``, and ``desired < 1`` simply means some ticks run
zero steps (Req 14.5, 14.6). Changing ``M`` or ``dt`` just changes the amount
added next tick -- no restart is needed (Req 12.4, 14.6).

Units convention (ms) matches the rest of ``core``.
"""

import math

# Default wall-clock timer interval in milliseconds (~50 ticks/second). The GUI
# QTimer fires at this cadence; the scheduler maps each tick onto engine steps.
DEFAULT_T_TICK = 20.0


class TickScheduler:
    """Accumulator-based mapping from wall-clock ticks to engine steps.

    Each wall-clock tick, :meth:`steps_for_tick` is asked how many integration
    steps the engine should run. It accumulates the desired fractional step
    count ``M * T_tick / dt`` and returns ``floor`` of the running total,
    carrying the fractional remainder forward so the long-run average advance
    is exactly ``M * T_tick`` of simulated time per tick -- independent of
    ``dt`` and valid for arbitrarily small ``M`` (Req 14.5, 14.6).

    The scheduler is stateless with respect to ``M`` and ``dt``: they are
    passed in per tick (or read from stored defaults), so live edits take
    effect on the very next tick with no restart (Req 12.4, 14.6).
    """

    def __init__(self, t_tick: float = DEFAULT_T_TICK) -> None:
        """Create a scheduler for a given wall-clock tick interval.

        Args:
            t_tick: Wall-clock timer interval in milliseconds. Must be
                strictly positive.
        """
        if t_tick <= 0:
            raise ValueError("t_tick must be a positive number of milliseconds")
        self.t_tick = t_tick
        # Carried fractional step count between ticks. Starts empty so the very
        # first tick dispatches floor(desired) with no spurious head start.
        self._accumulator = 0.0

    def steps_for_tick(self, M: float, dt: float) -> int:
        """Return how many engine steps to run for the current wall-clock tick.

        Accumulates the desired fractional step count ``M * T_tick / dt`` for
        this tick, dispatches the whole-number part, and keeps the remainder
        for the next tick. The result is always a non-negative integer; it may
        be ``0`` when very slow motion (small ``M``) with a small ``dt`` puts
        the per-tick desire below one step (Req 14.5).

        Args:
            M: Slow-motion speed multiplier (``[0.01, 1.0]`` in normal use).
                Must be non-negative.
            dt: Engine integration step in milliseconds. Must be strictly
                positive.

        Returns:
            The number of integration steps to run this tick (``>= 0``).
        """
        if dt <= 0:
            raise ValueError("dt must be a positive number of milliseconds")
        if M < 0:
            raise ValueError("M must be non-negative")

        # Desired (fractional) steps for this tick's slice of simulated time.
        self._accumulator += M * self.t_tick / dt

        # Dispatch whole steps; carry the fractional remainder to the next tick
        # so the long-run average advance stays exact regardless of dt.
        whole = math.floor(self._accumulator)
        self._accumulator -= whole
        return int(whole)

    def credit(self, steps: float) -> None:
        """Return undispatched steps to the accumulator for a later tick.

        Used by the GUI when it caps the number of steps it will run in a single
        tick (to keep the event loop responsive): the steps it chose not to run
        this tick are credited back here so they are dispatched on following
        ticks, preserving the long-run average simulated-time advance.

        Args:
            steps: Number of steps to carry forward. Must be non-negative.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self._accumulator += steps

    def reset(self) -> None:
        """Clear the carried fractional remainder.

        Called when the simulation is reset so a fresh run starts with an empty
        accumulator and no leftover step debt/credit from a previous run.
        """
        self._accumulator = 0.0
