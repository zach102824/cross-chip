"""Smoke tests for layer-3 parameter sweep generation (no OGM / no mitigation runs)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "scripts"))
import layer3_mitigation_sweep as l3  # noqa: E402


def test_generate_layer3_param_sets_shape_and_baseline() -> None:
    sets, sigma_used = l3.generate_layer3_param_sets(master_seed=42, num_sets=10)
    assert sets.shape == (10, 21)
    assert sigma_used.shape == (10,)
    np.testing.assert_allclose(sets[0], l3.BASELINE_LAYER3_VQE)
    assert sigma_used[0] == 0.0


def test_sigma_schedule_monotonic_increasing() -> None:
    _, sigma_used = l3.generate_layer3_param_sets(master_seed=7, num_sets=10)
    tail = sigma_used[1:]
    assert np.all(np.diff(tail) > 0), "perturbation sigmas should increase"


def test_baseline_vector_length_matches_layers() -> None:
    assert len(l3.BASELINE_LAYER3_VQE) == 21


def test_generate_layer3_param_sets_hundred_rows() -> None:
    sets, sigma_used = l3.generate_layer3_param_sets(master_seed=0, num_sets=100)
    assert sets.shape == (100, 21)
    assert sigma_used.shape == (100,)
    np.testing.assert_allclose(sets[0], l3.BASELINE_LAYER3_VQE)


def test_load_layer3_baseline_npy_roundtrip(tmp_path: Path) -> None:
    v = np.arange(21, dtype=float)
    p = tmp_path / "b.npy"
    np.save(p, v)
    got = l3.load_layer3_baseline_npy(p)
    np.testing.assert_allclose(got, v)


def test_load_layer3_baseline_npy_rejects_wrong_len(tmp_path: Path) -> None:
    np.save(tmp_path / "bad.npy", np.zeros(5))
    with pytest.raises(ValueError, match="21 parameters"):
        l3.load_layer3_baseline_npy(tmp_path / "bad.npy")


def test_load_layer3_baseline_npy_missing() -> None:
    with pytest.raises(FileNotFoundError):
        l3.load_layer3_baseline_npy(Path("/nonexistent/baseline.npy"))


def test_generate_with_custom_baseline_and_sigma_bounds() -> None:
    custom = np.linspace(-0.1, 0.1, 21)
    sets, sigma_used = l3.generate_layer3_param_sets(
        master_seed=1,
        num_sets=4,
        baseline=custom,
        sigma_min=0.1,
        sigma_max=1.0,
    )
    np.testing.assert_allclose(sets[0], custom)
    assert sigma_used[0] == 0.0
    assert sigma_used[1] == pytest.approx(0.1)
    n_pert = 3
    assert sigma_used.shape == (4,)
    expected_sigmas = np.logspace(np.log10(0.1), np.log10(1.0), n_pert)
    np.testing.assert_allclose(sigma_used[1:], expected_sigmas)
