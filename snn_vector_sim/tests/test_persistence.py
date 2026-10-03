"""Unit tests for settings serialization (:mod:`snn_vector_sim.core.persistence`).

Covers the configuration round-trip: capturing an engine's neuron parameters,
generators (all four types), weights, output gain, and ``dt`` to a dict/file
and restoring them onto a fresh engine.
"""

import json

import pytest

from snn_vector_sim.core.engine import SimulationEngine
from snn_vector_sim.core.current import (
    ConstantCurrent,
    ManualPulseCurrent,
    OffCurrent,
    PulseCurrent,
)
from snn_vector_sim.core.persistence import (
    apply_dict_to_engine,
    engine_to_dict,
    load_engine,
    save_engine,
)


def _configure(engine):
    """Set a distinctive, non-default configuration on an engine."""
    engine.input1.tau_m = 7.5
    engine.input1.v_th = -48.0
    engine.input2.R = 12.0
    engine.output.tau_m = 22.0
    engine.output.t_refrac = 4.0
    engine.w1 = 2.5
    engine.w2 = -1.5
    engine.w_gain = 40.0
    engine.dt = 0.25
    engine.gen1 = PulseCurrent(magnitude=33.0, period=60.0, width=7.0)
    engine.gen2 = ManualPulseCurrent(magnitude=18.0, width=3.0)


def test_round_trip_through_dict_preserves_configuration():
    src = SimulationEngine()
    _configure(src)

    data = engine_to_dict(src)

    dst = SimulationEngine()
    apply_dict_to_engine(dst, data)

    # Neuron params.
    assert dst.input1.tau_m == pytest.approx(7.5)
    assert dst.input1.v_th == pytest.approx(-48.0)
    assert dst.input2.R == pytest.approx(12.0)
    assert dst.output.tau_m == pytest.approx(22.0)
    assert dst.output.t_refrac == pytest.approx(4.0)
    # Weights + gain + dt.
    assert dst.w1 == pytest.approx(2.5)
    assert dst.w2 == pytest.approx(-1.5)
    assert dst.w_gain == pytest.approx(40.0)
    assert dst.dt == pytest.approx(0.25)
    # Generators by type and params.
    assert isinstance(dst.gen1, PulseCurrent)
    assert dst.gen1.magnitude == pytest.approx(33.0)
    assert dst.gen1.period == pytest.approx(60.0)
    assert dst.gen1.width == pytest.approx(7.0)
    assert isinstance(dst.gen2, ManualPulseCurrent)
    assert dst.gen2.magnitude == pytest.approx(18.0)
    assert dst.gen2.width == pytest.approx(3.0)


@pytest.mark.parametrize(
    "gen, gen_type",
    [
        (OffCurrent(), OffCurrent),
        (ConstantCurrent(magnitude=5.0), ConstantCurrent),
        (PulseCurrent(magnitude=9.0, period=40.0, width=4.0), PulseCurrent),
        (ManualPulseCurrent(magnitude=11.0, width=6.0), ManualPulseCurrent),
    ],
)
def test_each_generator_type_round_trips(gen, gen_type):
    src = SimulationEngine()
    src.gen1 = gen
    dst = SimulationEngine()
    apply_dict_to_engine(dst, engine_to_dict(src))
    assert isinstance(dst.gen1, gen_type)


def test_save_and_load_file(tmp_path):
    src = SimulationEngine()
    _configure(src)
    path = tmp_path / "my settings.json"

    save_engine(src, str(path))
    # The file is valid, human-readable JSON with a version tag.
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["version"] >= 1
    assert raw["generators"]["gen2"]["type"] == "manual"

    dst = SimulationEngine()
    load_engine(dst, str(path))
    assert dst.w_gain == pytest.approx(40.0)
    assert isinstance(dst.gen2, ManualPulseCurrent)


def test_apply_resets_runtime_state():
    """Applying settings restarts the run: N at origin, t=0, history cleared."""
    src = SimulationEngine()
    _configure(src)
    dst = SimulationEngine()
    # Dirty dst's runtime state before loading.
    for _ in range(50):
        dst.step()
    assert dst.t > 0

    apply_dict_to_engine(dst, engine_to_dict(src))
    assert dst.t == pytest.approx(0.0)
    assert dst.N.n1 == pytest.approx(0.0)
    assert dst.N.n2 == pytest.approx(0.0)
    assert len(dst.history) == 0


def test_partial_dict_leaves_missing_fields_untouched():
    """A sparse settings dict only overrides the keys it contains."""
    dst = SimulationEngine()
    original_w2 = dst.w2
    apply_dict_to_engine(dst, {"weights": {"w1": 9.0}})
    assert dst.w1 == pytest.approx(9.0)
    assert dst.w2 == pytest.approx(original_w2)
