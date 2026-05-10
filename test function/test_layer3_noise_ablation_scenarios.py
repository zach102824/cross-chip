"""Unit tests for noise ablation scenario presets (no OGM / no quantum runs)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "scripts"))
from layer3_noise_ablation import (  # noqa: E402
    DEFAULT_SCENARIO_IDS,
    build_noise_ablation_scenario,
)
from main_cursor_lib import (  # noqa: E402
    DEFAULT_AMP_DAMP_GAMMA,
    DEFAULT_DEPOL_PROB,
    DEFAULT_LEAKAGE_APPROX_PROB,
    DEFAULT_PHASE_DAMP_GAMMA,
)


def test_all_default_scenario_ids_build() -> None:
    for sid in DEFAULT_SCENARIO_IDS:
        cfg, apply_ro = build_noise_ablation_scenario(sid)
        assert set(cfg.keys()) == {
            "amp_damp_gamma",
            "phase_damp_gamma",
            "depol_prob",
            "leakage_approx_prob",
            "high_cz_multiplier",
        }
        assert isinstance(apply_ro, bool)


def test_readout_only_silent_simulator() -> None:
    cfg, apply_ro = build_noise_ablation_scenario("readout_only")
    assert apply_ro is True
    np.testing.assert_allclose(cfg["amp_damp_gamma"], 0.0)
    np.testing.assert_allclose(cfg["depol_prob"], 0.0)


def test_amp_only_readout_off() -> None:
    cfg, apply_ro = build_noise_ablation_scenario("amp_only")
    assert apply_ro is False
    assert cfg["amp_damp_gamma"] == DEFAULT_AMP_DAMP_GAMMA
    assert cfg["phase_damp_gamma"] == 0.0


def test_amp_readout_pair() -> None:
    cfg, apply_ro = build_noise_ablation_scenario("amp_readout")
    assert apply_ro is True
    assert cfg["amp_damp_gamma"] == DEFAULT_AMP_DAMP_GAMMA
    assert cfg["depol_prob"] == 0.0


def test_phase_depol_pair() -> None:
    cfg, apply_ro = build_noise_ablation_scenario("phase_depol")
    assert apply_ro is False
    assert cfg["phase_damp_gamma"] == DEFAULT_PHASE_DAMP_GAMMA
    assert cfg["depol_prob"] == DEFAULT_DEPOL_PROB


def test_leakage_only() -> None:
    cfg, apply_ro = build_noise_ablation_scenario("leakage_only")
    assert apply_ro is False
    assert cfg["leakage_approx_prob"] == DEFAULT_LEAKAGE_APPROX_PROB
    assert cfg["amp_damp_gamma"] == 0.0


def test_unknown_scenario_raises() -> None:
    try:
        build_noise_ablation_scenario("not_a_real_scenario")
    except ValueError as e:
        assert "Unknown" in str(e)
    else:
        raise AssertionError("expected ValueError")
