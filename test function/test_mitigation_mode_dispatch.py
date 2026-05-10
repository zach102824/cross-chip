from __future__ import annotations

from pathlib import Path
import sys

import cirq
import numpy as np
import pytest
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shot_measurement import VALID_MITIGATION_MODES, run_mitigation


def _small_problem():
    q0, q1 = cirq.LineQubit.range(2)
    th = sympy.Symbol("th_0_0")
    ph = sympy.Symbol("ph_0_1")
    circuit = cirq.Circuit(
        [
            cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=th / sympy.pi).on(q0),
            cirq.ZPowGate(exponent=-ph / (2 * sympy.pi)).on(q1),
            cirq.CZ(q0, q1),
        ]
    )
    qubits = [q0, q1]
    target_params = {th: 0.6, ph: 0.7}
    target_resolver = dict(target_params)
    observable = 0.6 * cirq.Z(q0) + 0.4 * cirq.X(q1) - 0.2 * cirq.Z(q0) * cirq.Z(q1)
    return circuit, qubits, [th, ph], target_params, target_resolver, observable


def _common_kwargs():
    base_noise_cfg = dict(
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
    )
    shot_cfg = dict(
        num_shots=1500,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        sampling_seed=23,
    )
    zne_cfg = dict(noise_scales=[1.0, 2.0, 3.0], fit_order=1)
    cdr_cfg = dict(
        num_circuits=4,
        t_max=1,
        min_snap_fraction=0.0,
        seed=11,
        cdr_fit_scope="total_energy",
    )
    return base_noise_cfg, shot_cfg, zne_cfg, cdr_cfg


def test_invalid_mode_raises() -> None:
    circuit, qubits, symbols, target_params, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, _, _ = _common_kwargs()
    with pytest.raises(ValueError, match="mitigation_mode"):
        run_mitigation(
            "no-such-mode",
            ansatz_circuit=circuit,
            observable_h=obs,
            qubits=qubits,
            target_resolver=target_resolver,
            target_params=target_params,
            symbols=symbols,
            base_noise_cfg=base_noise_cfg,
            shot_cfg=shot_cfg,
        )


def test_valid_mode_set() -> None:
    assert set(VALID_MITIGATION_MODES) == {"none", "zne", "cdr", "both"}


def test_mode_none_returns_baseline_only() -> None:
    circuit, qubits, _, _, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, _, _ = _common_kwargs()
    out = run_mitigation(
        "none",
        ansatz_circuit=circuit,
        observable_h=obs,
        qubits=qubits,
        target_resolver=target_resolver,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
    )
    assert out["mode"] == "none"
    assert "unmit_target" in out
    assert "rem_target" in out
    forbidden = {
        "trace_zne", "shot_zne", "cdr_models",
        "cdr_unmit_corrected", "cdr_rem_corrected",
        "per_scale_cdr_unmit", "zne_of_cdr_unmit_target",
    }
    assert not (forbidden & out.keys())


def test_mode_zne_returns_zne_keys_only() -> None:
    circuit, qubits, _, _, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, zne_cfg, _ = _common_kwargs()
    out = run_mitigation(
        "zne",
        ansatz_circuit=circuit,
        observable_h=obs,
        qubits=qubits,
        target_resolver=target_resolver,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        zne_cfg=zne_cfg,
    )
    assert "trace_zne" in out
    assert "shot_zne" in out
    assert "cdr_unmit_corrected" not in out
    assert "zne_of_cdr_unmit_target" not in out
    assert len(out["trace_zne"]["trace_energies"]) == 3
    assert len(out["shot_zne"]["shot_unmitigated_energies"]) == 3


def test_mode_cdr_returns_cdr_keys_only() -> None:
    circuit, qubits, symbols, target_params, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, _, cdr_cfg = _common_kwargs()
    out = run_mitigation(
        "cdr",
        ansatz_circuit=circuit,
        observable_h=obs,
        qubits=qubits,
        target_resolver=target_resolver,
        target_params=target_params,
        symbols=symbols,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        cdr_cfg=cdr_cfg,
    )
    assert "cdr_unmit_corrected" in out
    assert "cdr_rem_corrected" in out
    assert "cdr_models" in out
    coeffs_unmit = out["cdr_models"]["coeffs_unmit_to_exact"]
    assert len(coeffs_unmit) == 2
    assert "trace_zne" not in out
    assert "zne_of_cdr_unmit_target" not in out


def test_mode_both_returns_per_scale_and_extrapolated_keys() -> None:
    circuit, qubits, symbols, target_params, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, zne_cfg, cdr_cfg = _common_kwargs()
    out = run_mitigation(
        "both",
        ansatz_circuit=circuit,
        observable_h=obs,
        qubits=qubits,
        target_resolver=target_resolver,
        target_params=target_params,
        symbols=symbols,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        zne_cfg=zne_cfg,
        cdr_cfg=cdr_cfg,
    )
    assert "trace_zne" in out
    assert "shot_zne" in out
    assert "per_scale_cdr_unmit" in out
    assert "per_scale_cdr_rem" in out
    assert "zne_of_cdr_unmit_target" in out
    assert "zne_of_cdr_rem_target" in out
    assert len(out["per_scale_cdr_unmit"]) == len(zne_cfg["noise_scales"])
    assert len(out["per_scale_cdr_rem"]) == len(zne_cfg["noise_scales"])
    assert np.isfinite(float(out["zne_of_cdr_unmit_target"]))
    assert np.isfinite(float(out["zne_of_cdr_rem_target"]))


def test_mode_both_per_scale_models_are_independent_lists() -> None:
    circuit, qubits, symbols, target_params, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, zne_cfg, cdr_cfg = _common_kwargs()
    out = run_mitigation(
        "both",
        ansatz_circuit=circuit,
        observable_h=obs,
        qubits=qubits,
        target_resolver=target_resolver,
        target_params=target_params,
        symbols=symbols,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        zne_cfg=zne_cfg,
        cdr_cfg=cdr_cfg,
    )
    per_scale_models = out["per_scale_cdr_models"]
    assert len(per_scale_models) == len(zne_cfg["noise_scales"])
    t_remaining_lists = [m["training_t_remaining"] for m in per_scale_models]
    for tr in t_remaining_lists[1:]:
        assert tr == t_remaining_lists[0], (
            "Snap pattern is scale-independent; per-scale t_remaining "
            "should be identical across scales."
        )


def test_mode_both_reduces_to_cdr_when_zne_scales_one_and_zero_order() -> None:
    """With zne_scales=[1.0] and fit_order=0, zne_of_cdr_*_target should match
    the per-scale CDR-corrected target value."""
    circuit, qubits, symbols, target_params, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, _, cdr_cfg = _common_kwargs()
    zne_cfg = dict(noise_scales=[1.0], fit_order=0)

    out = run_mitigation(
        "both",
        ansatz_circuit=circuit,
        observable_h=obs,
        qubits=qubits,
        target_resolver=target_resolver,
        target_params=target_params,
        symbols=symbols,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        zne_cfg=zne_cfg,
        cdr_cfg=cdr_cfg,
    )
    assert abs(
        float(out["zne_of_cdr_unmit_target"]) - float(out["per_scale_cdr_unmit"][0])
    ) <= 1e-9
    assert abs(
        float(out["zne_of_cdr_rem_target"]) - float(out["per_scale_cdr_rem"][0])
    ) <= 1e-9


def test_mode_zne_requires_zne_cfg() -> None:
    circuit, qubits, _, _, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, _, _ = _common_kwargs()
    with pytest.raises(ValueError, match="zne_cfg"):
        run_mitigation(
            "zne",
            ansatz_circuit=circuit,
            observable_h=obs,
            qubits=qubits,
            target_resolver=target_resolver,
            base_noise_cfg=base_noise_cfg,
            shot_cfg=shot_cfg,
        )


def test_mode_cdr_default_per_pauli_model_keys() -> None:
    circuit, qubits, symbols, target_params, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, _, _ = _common_kwargs()
    cdr_cfg = dict(
        num_circuits=4,
        t_max=1,
        min_snap_fraction=0.0,
        seed=11,
    )
    out = run_mitigation(
        "cdr",
        ansatz_circuit=circuit,
        observable_h=obs,
        qubits=qubits,
        target_resolver=target_resolver,
        target_params=target_params,
        symbols=symbols,
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        cdr_cfg=cdr_cfg,
    )
    assert out["cdr_fit_scope"] == "per_pauli"
    m = out["cdr_models"]
    assert m["fit_scope"] == "per_pauli"
    assert "coeffs_unmit_to_exact_per_term" in m
    assert len(m["coeffs_unmit_to_exact_per_term"]) == 3


def test_mode_cdr_requires_cdr_cfg() -> None:
    circuit, qubits, _, _, target_resolver, obs = _small_problem()
    base_noise_cfg, shot_cfg, _, _ = _common_kwargs()
    with pytest.raises(ValueError, match="cdr_cfg"):
        run_mitigation(
            "cdr",
            ansatz_circuit=circuit,
            observable_h=obs,
            qubits=qubits,
            target_resolver=target_resolver,
            base_noise_cfg=base_noise_cfg,
            shot_cfg=shot_cfg,
        )
