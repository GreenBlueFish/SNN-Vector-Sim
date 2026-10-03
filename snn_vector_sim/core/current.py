"""Input current generators for the input neurons.

Each generator produces the externally injected current (in nA) delivered to
an input neuron on a given timestep. Generators are live-switchable per input:
swapping the type or editing a field is a direct attribute write that takes
effect on the next step.

Units convention (mV-ms-nA-MOhm):
- Time (t, dt, period, width): milliseconds (ms)
- magnitude and returned current: nanoamperes (nA)
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class CurrentGenerator(Protocol):
    """Protocol for an input current source.

    Implementations return the injected current (nA) for the current step
    given the simulated time ``t`` (ms) and timestep ``dt`` (ms).
    """

    def value(self, t: float, dt: float) -> float:
        """Return the injected current (nA) for this timestep."""
        ...


@dataclass
class ConstantCurrent:
    """A current held at a fixed magnitude on every timestep.

    Injects ``magnitude`` nA every step regardless of time (Req 5.3).
    """

    magnitude: float = 0.0  # nA injected every step

    def value(self, t: float, dt: float) -> float:
        """Return ``magnitude`` for all ``t``."""
        return self.magnitude


@dataclass
class PulseCurrent:
    """A repeating pulse train of current.

    Injects ``magnitude`` nA for ``width`` ms within each ``period`` ms window
    and ``0`` nA otherwise (Req 5.4). A repeating pulse train is more useful
    than a one-shot for observing patterns; ``period >= width`` is expected.
    """

    magnitude: float = 20.0  # nA injected while the pulse is on
    period: float = 50.0     # ms between pulse onsets
    width: float = 5.0       # ms each pulse stays on

    def value(self, t: float, dt: float) -> float:
        """Return ``magnitude`` when within a pulse window, else ``0``."""
        phase = t % self.period
        return self.magnitude if phase < self.width else 0.0


@dataclass
class OffCurrent:
    """A disabled current source that injects nothing.

    Always returns ``0`` nA, used to turn an input off entirely while keeping
    the input neuron in the network (it simply receives no external drive).
    Exposes a ``magnitude`` attribute (always ignored) so the GUI can treat all
    generator types uniformly.
    """

    magnitude: float = 0.0  # present for a uniform GUI interface; unused

    def value(self, t: float, dt: float) -> float:
        """Return ``0`` for all ``t`` (the input is off)."""
        return 0.0


@dataclass
class ManualPulseCurrent:
    """A single pulse emitted only when :meth:`fire` is called.

    Unlike :class:`PulseCurrent` (which repeats on a fixed period), this source
    stays at ``0`` nA until the user triggers it via :meth:`fire`. Each trigger
    arms a pulse of ``magnitude`` nA lasting ``width`` ms; :meth:`value`
    delivers that current and counts the remaining on-time down by ``dt`` each
    step until the pulse ends. Firing again re-arms (restarts) the pulse.

    Because the "fire" event comes from a GUI button on the main thread and
    :meth:`value` is called from the same thread's tick loop, no locking is
    needed.
    """

    magnitude: float = 20.0  # nA injected while the manual pulse is on
    width: float = 5.0       # ms the pulse stays on after each fire
    _remaining: float = 0.0  # ms of on-time left for the current pulse

    def fire(self) -> None:
        """Arm (or re-arm) a single pulse of ``width`` ms of on-time."""
        self._remaining = self.width

    def value(self, t: float, dt: float) -> float:
        """Return ``magnitude`` while a fired pulse is active, else ``0``.

        Each call consumes ``dt`` ms of the remaining on-time, so a pulse armed
        by :meth:`fire` lasts about ``width`` ms regardless of ``dt``.
        """
        if self._remaining > 0.0:
            self._remaining -= dt
            return self.magnitude
        return 0.0
