"""Save and load simulation settings to/from a plain JSON snapshot.

This module converts a :class:`~snn_vector_sim.core.engine.SimulationEngine`'s
*configuration* (not its live runtime state) to a JSON-serializable dict and
back. The snapshot captures the three neurons' parameters, both input current
generators (by type and their parameters), the two synaptic weights, the output
gain, and the integration timestep ``dt``.

Runtime state -- membrane potentials, the vector ``N``, simulated time ``t``,
and the history buffers -- is intentionally *not* saved. Loading a file applies
the saved configuration to an existing engine and resets its runtime state so
the run restarts cleanly from the loaded settings.

The format is a small, human-readable JSON document with a ``version`` field so
future changes can be detected. All numbers are plain floats; generator types
are stored by a short string tag.
"""

import json
from typing import Any, Dict

from snn_vector_sim.core.current import (
    ConstantCurrent,
    ManualPulseCurrent,
    OffCurrent,
    PulseCurrent,
)

#: Bumped if the on-disk schema changes incompatibly.
SETTINGS_VERSION = 1

# Neuron parameter names persisted (runtime state like ``v`` is excluded).
_NEURON_PARAMS = ("tau_m", "R", "v_rest", "v_th", "v_reset", "t_refrac")

# String tags identifying each generator type on disk.
_GEN_OFF = "off"
_GEN_CONSTANT = "constant"
_GEN_PULSE = "pulse"
_GEN_MANUAL = "manual"


def _neuron_to_dict(neuron) -> Dict[str, float]:
    """Return a neuron's persisted parameters as a dict."""
    return {name: float(getattr(neuron, name)) for name in _NEURON_PARAMS}


def _apply_neuron_dict(neuron, data: Dict[str, Any]) -> None:
    """Apply persisted parameters onto an existing neuron, then reset its state.

    Unknown keys are ignored and missing keys leave the neuron's current value
    untouched, so partial or older files load without error.
    """
    for name in _NEURON_PARAMS:
        if name in data:
            setattr(neuron, name, float(data[name]))
    neuron.reset_state()


def _generator_to_dict(gen) -> Dict[str, Any]:
    """Serialize a current generator to a ``{"type": ..., ...}`` dict."""
    if isinstance(gen, OffCurrent):
        return {"type": _GEN_OFF}
    if isinstance(gen, PulseCurrent):
        return {
            "type": _GEN_PULSE,
            "magnitude": float(gen.magnitude),
            "period": float(gen.period),
            "width": float(gen.width),
        }
    if isinstance(gen, ManualPulseCurrent):
        return {
            "type": _GEN_MANUAL,
            "magnitude": float(gen.magnitude),
            "width": float(gen.width),
        }
    # Default/fallback: treat anything else as a constant with its magnitude.
    return {
        "type": _GEN_CONSTANT,
        "magnitude": float(getattr(gen, "magnitude", 0.0)),
    }


def _generator_from_dict(data: Dict[str, Any]):
    """Build a fresh current generator from its serialized dict.

    Unknown or missing type tags fall back to a zero :class:`ConstantCurrent`
    so a malformed entry degrades gracefully rather than raising.
    """
    gen_type = data.get("type", _GEN_CONSTANT)
    if gen_type == _GEN_OFF:
        return OffCurrent()
    if gen_type == _GEN_PULSE:
        return PulseCurrent(
            magnitude=float(data.get("magnitude", 20.0)),
            period=float(data.get("period", 50.0)),
            width=float(data.get("width", 5.0)),
        )
    if gen_type == _GEN_MANUAL:
        return ManualPulseCurrent(
            magnitude=float(data.get("magnitude", 20.0)),
            width=float(data.get("width", 5.0)),
        )
    return ConstantCurrent(magnitude=float(data.get("magnitude", 0.0)))


def engine_to_dict(engine) -> Dict[str, Any]:
    """Capture an engine's configuration as a JSON-serializable dict."""
    return {
        "version": SETTINGS_VERSION,
        "neurons": {
            "input1": _neuron_to_dict(engine.input1),
            "input2": _neuron_to_dict(engine.input2),
            "output": _neuron_to_dict(engine.output),
        },
        "generators": {
            "gen1": _generator_to_dict(engine.gen1),
            "gen2": _generator_to_dict(engine.gen2),
        },
        "weights": {
            "w1": float(engine.w1),
            "w2": float(engine.w2),
            "w_gain": float(engine.w_gain),
        },
        "dt": float(engine.dt),
    }


def apply_dict_to_engine(engine, data: Dict[str, Any]) -> None:
    """Apply a settings dict onto an existing engine and reset its runtime.

    Missing sections are skipped so partial files still load. After applying
    the configuration, :meth:`SimulationEngine.reset` is called so the run
    restarts cleanly from the loaded settings (clears N, history, and time).
    """
    neurons = data.get("neurons", {})
    if "input1" in neurons:
        _apply_neuron_dict(engine.input1, neurons["input1"])
    if "input2" in neurons:
        _apply_neuron_dict(engine.input2, neurons["input2"])
    if "output" in neurons:
        _apply_neuron_dict(engine.output, neurons["output"])

    gens = data.get("generators", {})
    if "gen1" in gens:
        engine.gen1 = _generator_from_dict(gens["gen1"])
    if "gen2" in gens:
        engine.gen2 = _generator_from_dict(gens["gen2"])

    weights = data.get("weights", {})
    if "w1" in weights:
        engine.w1 = float(weights["w1"])
    if "w2" in weights:
        engine.w2 = float(weights["w2"])
    if "w_gain" in weights:
        engine.w_gain = float(weights["w_gain"])

    if "dt" in data:
        engine.dt = float(data["dt"])

    # Restart cleanly from the loaded configuration.
    engine.reset()


def save_engine(engine, path: str) -> None:
    """Write an engine's configuration to ``path`` as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(engine_to_dict(engine), fh, indent=2)


def load_engine(engine, path: str) -> None:
    """Read a settings file from ``path`` and apply it to ``engine``."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    apply_dict_to_engine(engine, data)
